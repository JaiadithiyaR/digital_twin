#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/nr-module.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("NdtOranNetwork");

// ============================================================
// CONFIGURATION
// ============================================================

static constexpr double DEFAULT_SIMULATION_TIME = 500.0;
static constexpr double TELEMETRY_INTERVAL = 1.0;

static constexpr uint32_t NUM_UE = 2;
static constexpr uint32_t NUM_GNB = 1;

static constexpr double CENTRAL_FREQUENCY = 3.5e9;
static constexpr double BANDWIDTH = 20e6;

static constexpr uint32_t PACKET_SIZE = 1000;

// 1000 bytes every 1 ms = 8 Mbps per UE
static constexpr double CONFIGURED_OFFERED_LOAD_MBPS = 8.0;

// Used for invalid/unavailable radio KPI values.
// This is much safer for ML/DT processing than NaN.
static constexpr double INVALID_METRIC = -999.0;

// ============================================================
// GLOBAL STOP CONTROL
// ============================================================

volatile std::sig_atomic_t g_stopSignal = 0;

void
SignalHandler(int signalNumber)
{
    if (signalNumber == SIGINT || signalNumber == SIGTERM)
    {
        g_stopSignal = 1;
    }
}

// ============================================================
// RADIO TELEMETRY
// ============================================================

struct UeRadioTelemetry
{
    double rsrpDbm = INVALID_METRIC;
    double rsrqDb = INVALID_METRIC;
    double sinrDb = INVALID_METRIC;

    uint8_t cqi = 0;
    uint8_t mcs = 0;
    uint8_t ri = 0;

    uint16_t cellId = 0;

    bool hasRsrp = false;
    bool hasRsrq = false;
    bool hasSinr = false;
    bool hasCqi = false;
};

struct ResourceTelemetry
{
    uint64_t rbActiveCount = 0;
    uint64_t rbSamples = 0;

    uint64_t slotSamples = 0;

    double scheduledUeSum = 0.0;

    double symbolUtilizationSum = 0.0;

    uint32_t lastAvailableRb = 0;
    uint32_t lastAvailableSymbols = 0;

    uint16_t cellId = 0;
    uint16_t bwpId = 0;
};

struct PreviousFlowStats
{
    uint64_t txPackets = 0;
    uint64_t rxPackets = 0;

    uint64_t txBytes = 0;
    uint64_t rxBytes = 0;

    Time delaySum = Seconds(0);
    Time jitterSum = Seconds(0);
};

// ============================================================
// GLOBAL OBJECTS
// ============================================================

static std::vector<UeRadioTelemetry> g_radioTelemetry(NUM_UE);

static ResourceTelemetry g_resourceTelemetry;

static Ptr<FlowMonitor> g_flowMonitor;
static Ptr<Ipv4FlowClassifier> g_flowClassifier;

static Ipv4InterfaceContainer g_ueIpInterfaces;

static NodeContainer g_ueNodes;
static NodeContainer g_gnbNodes;

static NetDeviceContainer g_ueDevices;
static NetDeviceContainer g_gnbDevices;

static std::map<FlowId, PreviousFlowStats> g_previousFlowStats;

static std::ofstream g_telemetryFile;

static double g_lastTelemetryTime = 0.0;

// ============================================================
// RSRP / RSRQ
//
// 5G-LENA callback:
//
// uint16_t rnti
// uint16_t cellId
// double rsrp
// double rsrq
// bool servingCell
// uint8_t componentCarrierId
// ============================================================

void
ReportUeMeasurements(uint32_t ueId,
                     uint16_t rnti,
                     uint16_t cellId,
                     double rsrp,
                     double rsrq,
                     bool servingCell,
                     uint8_t componentCarrierId)
{
    if (ueId >= g_radioTelemetry.size())
    {
        return;
    }

    if (!servingCell)
    {
        return;
    }

    g_radioTelemetry[ueId].rsrpDbm = rsrp;
    g_radioTelemetry[ueId].rsrqDb = rsrq;

    g_radioTelemetry[ueId].cellId = cellId;

    g_radioTelemetry[ueId].hasRsrp = true;
    g_radioTelemetry[ueId].hasRsrq = true;
}

// ============================================================
// SINR
//
// IMPORTANT:
// 5G-LENA DlDataSinr is LINEAR SINR.
// We convert it to dB.
// ============================================================

void
ReportDlSinr(uint32_t ueId,
             uint16_t cellId,
             uint16_t rnti,
             double sinrLinear,
             uint16_t bwpId)
{
    if (ueId >= g_radioTelemetry.size())
    {
        return;
    }

    if (sinrLinear > 0.0 &&
        std::isfinite(sinrLinear))
    {
        g_radioTelemetry[ueId].sinrDb =
            10.0 * std::log10(sinrLinear);

        g_radioTelemetry[ueId].hasSinr = true;
    }

    g_radioTelemetry[ueId].cellId = cellId;
}

// ============================================================
// CQI / MCS / RI
//
// 5G-LENA callback:
// rnti, CQI, MCS, RI
// ============================================================

void
ReportCqiMcsRi(uint32_t ueId,
              uint16_t rnti,
              uint8_t cqi,
              uint8_t mcs,
              uint8_t ri)
{
    if (ueId >= g_radioTelemetry.size())
    {
        return;
    }

    g_radioTelemetry[ueId].cqi = cqi;
    g_radioTelemetry[ueId].mcs = mcs;
    g_radioTelemetry[ueId].ri = ri;

    g_radioTelemetry[ueId].hasCqi = true;
}

// ============================================================
// SLOT DATA STATISTICS
//
// SlotDataStats:
//
// SfnSf
// scheduled UE
// used RE
// used symbols
// available RB
// available symbols
// BWP ID
// Cell ID
// ============================================================

void
ReportSlotStats(const SfnSf& sfnSf,
                uint32_t scheduledUe,
                uint32_t usedReg,
                uint32_t usedSym,
                uint32_t availableRb,
                uint32_t availableSym,
                uint16_t bwpId,
                uint16_t cellId)
{
    g_resourceTelemetry.slotSamples++;

    g_resourceTelemetry.scheduledUeSum +=
        static_cast<double>(scheduledUe);

    g_resourceTelemetry.lastAvailableRb =
        availableRb;

    g_resourceTelemetry.lastAvailableSymbols =
        availableSym;

    g_resourceTelemetry.cellId = cellId;
    g_resourceTelemetry.bwpId = bwpId;

    // Symbol utilization:
    //
    // used symbols / all available symbols.
    //
    // usedSym is the number of symbols containing
    // scheduled data resources.

    if (availableSym > 0)
    {
        double symbolUtilization =
            100.0 *
            static_cast<double>(usedSym) /
            static_cast<double>(availableSym);

        symbolUtilization =
            std::clamp(
                symbolUtilization,
                0.0,
                100.0);

        g_resourceTelemetry.symbolUtilizationSum +=
            symbolUtilization;
    }
}

// ============================================================
// RB DATA STATISTICS
//
// RBDataStats:
//
// SfnSf
// symbol
// rbMap
// BWP ID
// cell ID
//
// rbMap contains the active RB indexes.
// ============================================================

void
ReportRbStats(const SfnSf& sfnSf,
              uint8_t symbol,
              const std::vector<int>& rbMap,
              uint16_t bwpId,
              uint16_t cellId)
{
    if (rbMap.empty())
    {
        return;
    }

    g_resourceTelemetry.rbSamples++;

    g_resourceTelemetry.rbActiveCount +=
        static_cast<uint64_t>(rbMap.size());

    g_resourceTelemetry.cellId = cellId;
    g_resourceTelemetry.bwpId = bwpId;
}

// ============================================================
// FIND UE FROM IP
// ============================================================

int
FindUeFromDestination(Ipv4Address destination)
{
    for (uint32_t i = 0;
         i < g_ueIpInterfaces.GetN();
         ++i)
    {
        if (g_ueIpInterfaces.GetAddress(i) ==
            destination)
        {
            return static_cast<int>(i);
        }
    }

    return -1;
}

// ============================================================
// STOP FILE CHECK
// ============================================================

bool
StopFileExists()
{
    std::ifstream file("STOP_SIMULATION");

    return file.good();
}

// ============================================================
// STOP POLLER
//
// Allows:
//
// touch STOP_SIMULATION
//
// or Ctrl+C
// ============================================================

void
CheckStopRequest()
{
    if (g_stopSignal != 0 ||
        StopFileExists())
    {
        std::cout
            << "\n[NDT] Stop request detected.\n"
            << "[NDT] Stopping simulation gracefully...\n";

        Simulator::Stop();

        return;
    }

    Simulator::Schedule(
        MilliSeconds(100),
        &CheckStopRequest);
}

// ============================================================
// CONTINUOUS TELEMETRY
// ============================================================

void
CollectTelemetry()
{
    const double now =
        Simulator::Now().GetSeconds();

    if (g_flowMonitor == nullptr ||
        g_flowClassifier == nullptr)
    {
        return;
    }

    g_flowMonitor->CheckForLostPackets();

    FlowMonitor::FlowStatsContainer flowStats =
        g_flowMonitor->GetFlowStats();

    // ========================================================
    // Per-UE interval statistics
    // ========================================================

    std::vector<uint64_t> txPackets(NUM_UE, 0);
    std::vector<uint64_t> rxPackets(NUM_UE, 0);

    std::vector<uint64_t> txBytes(NUM_UE, 0);
    std::vector<uint64_t> rxBytes(NUM_UE, 0);

    std::vector<double> delaySumMs(NUM_UE, 0.0);
    std::vector<double> jitterSumMs(NUM_UE, 0.0);

    // ========================================================
    // FlowMonitor delta
    // ========================================================

    for (const auto& flow : flowStats)
    {
        FlowId flowId = flow.first;

        const FlowMonitor::FlowStats& current =
            flow.second;

        Ipv4FlowClassifier::FiveTuple tuple =
            g_flowClassifier->FindFlow(flowId);

        int ueId =
            FindUeFromDestination(
                tuple.destinationAddress);

        if (ueId < 0 ||
            ueId >= static_cast<int>(NUM_UE))
        {
            continue;
        }

        PreviousFlowStats& previous =
            g_previousFlowStats[flowId];

        uint64_t deltaTxPackets =
            current.txPackets -
            previous.txPackets;

        uint64_t deltaRxPackets =
            current.rxPackets -
            previous.rxPackets;

        uint64_t deltaTxBytes =
            current.txBytes -
            previous.txBytes;

        uint64_t deltaRxBytes =
            current.rxBytes -
            previous.rxBytes;

        Time deltaDelay =
            current.delaySum -
            previous.delaySum;

        Time deltaJitter =
            current.jitterSum -
            previous.jitterSum;

        txPackets[ueId] +=
            deltaTxPackets;

        rxPackets[ueId] +=
            deltaRxPackets;

        txBytes[ueId] +=
            deltaTxBytes;

        rxBytes[ueId] +=
            deltaRxBytes;

        delaySumMs[ueId] +=
            deltaDelay.GetSeconds() *
            1000.0;

        jitterSumMs[ueId] +=
            deltaJitter.GetSeconds() *
            1000.0;

        previous.txPackets =
            current.txPackets;

        previous.rxPackets =
            current.rxPackets;

        previous.txBytes =
            current.txBytes;

        previous.rxBytes =
            current.rxBytes;

        previous.delaySum =
            current.delaySum;

        previous.jitterSum =
            current.jitterSum;
    }

    // ========================================================
    // Resource KPIs
    // ========================================================

    double prbUtilizationPercent =
        0.0;

    if (g_resourceTelemetry.rbSamples > 0 &&
        g_resourceTelemetry.lastAvailableRb > 0)
    {
        double totalAvailableRbSamples =
            static_cast<double>(
                g_resourceTelemetry.rbSamples) *
            static_cast<double>(
                g_resourceTelemetry.lastAvailableRb);

        prbUtilizationPercent =
            100.0 *
            static_cast<double>(
                g_resourceTelemetry.rbActiveCount) /
            totalAvailableRbSamples;

        prbUtilizationPercent =
            std::clamp(
                prbUtilizationPercent,
                0.0,
                100.0);
    }

    double symbolUtilizationPercent =
        0.0;

    if (g_resourceTelemetry.slotSamples > 0)
    {
        symbolUtilizationPercent =
            g_resourceTelemetry.symbolUtilizationSum /
            static_cast<double>(
                g_resourceTelemetry.slotSamples);

        symbolUtilizationPercent =
            std::clamp(
                symbolUtilizationPercent,
                0.0,
                100.0);
    }

    double scheduledUes =
        0.0;

    if (g_resourceTelemetry.slotSamples > 0)
    {
        scheduledUes =
            g_resourceTelemetry.scheduledUeSum /
            static_cast<double>(
                g_resourceTelemetry.slotSamples);
    }

    // ========================================================
    // One row per UE
    // ========================================================

    for (uint32_t ueId = 0;
         ueId < NUM_UE;
         ++ueId)
    {
        // ----------------------------------------------------
        // Mobility
        // ----------------------------------------------------

        Ptr<MobilityModel> mobility =
            g_ueNodes.Get(ueId)
                ->GetObject<MobilityModel>();

        Vector position =
            mobility->GetPosition();

        double speed = 0.0;

        Ptr<ConstantVelocityMobilityModel>
            constantVelocity =
            DynamicCast<
                ConstantVelocityMobilityModel>(
                    mobility);

        if (constantVelocity)
        {
            speed =
                constantVelocity
                    ->GetVelocity()
                    .GetLength();
        }

        // ----------------------------------------------------
        // Throughput
        // ----------------------------------------------------

        double throughputMbps = 0.0;

        if (TELEMETRY_INTERVAL > 0.0)
        {
            throughputMbps =
                static_cast<double>(
                    rxBytes[ueId]) *
                8.0 /
                TELEMETRY_INTERVAL /
                1e6;
        }

        // ----------------------------------------------------
        // Offered load
        //
        // Actual traffic injected during this interval.
        // ----------------------------------------------------

        double offeredLoadMbps = 0.0;

        if (TELEMETRY_INTERVAL > 0.0)
        {
            offeredLoadMbps =
                static_cast<double>(
                    txBytes[ueId]) *
                8.0 /
                TELEMETRY_INTERVAL /
                1e6;
        }

        // ----------------------------------------------------
        // Latency
        // ----------------------------------------------------

        double latencyMs = 0.0;

        if (rxPackets[ueId] > 0)
        {
            latencyMs =
                delaySumMs[ueId] /
                static_cast<double>(
                    rxPackets[ueId]);
        }

        // ----------------------------------------------------
        // Jitter
        // ----------------------------------------------------

        double jitterMs = 0.0;

        if (rxPackets[ueId] > 0)
        {
            jitterMs =
                jitterSumMs[ueId] /
                static_cast<double>(
                    rxPackets[ueId]);
        }

        // ----------------------------------------------------
        // Packet loss
        // ----------------------------------------------------

        double packetLossPercent = 0.0;

        if (txPackets[ueId] > 0)
        {
            uint64_t lostPackets = 0;

            if (txPackets[ueId] >=
                rxPackets[ueId])
            {
                lostPackets =
                    txPackets[ueId] -
                    rxPackets[ueId];
            }

            packetLossPercent =
                100.0 *
                static_cast<double>(
                    lostPackets) /
                static_cast<double>(
                    txPackets[ueId]);

            packetLossPercent =
                std::clamp(
                    packetLossPercent,
                    0.0,
                    100.0);
        }

        // ----------------------------------------------------
        // Packet delivery ratio
        // ----------------------------------------------------

        double packetDeliveryRatioPercent =
            0.0;

        if (txPackets[ueId] > 0)
        {
            packetDeliveryRatioPercent =
                100.0 *
                static_cast<double>(
                    rxPackets[ueId]) /
                static_cast<double>(
                    txPackets[ueId]);

            packetDeliveryRatioPercent =
                std::clamp(
                    packetDeliveryRatioPercent,
                    0.0,
                    100.0);
        }

        // ----------------------------------------------------
        // Cell ID
        // ----------------------------------------------------

        uint16_t cellId =
            g_radioTelemetry[ueId].cellId;

        if (cellId == 0)
        {
            Ptr<NrUeNetDevice> ue =
                DynamicCast<NrUeNetDevice>(
                    g_ueDevices.Get(ueId));

            if (ue)
            {
                Ptr<NrUePhy> uePhy =
                    NrHelper::GetUePhy(
                        g_ueDevices.Get(ueId),
                        0);

                if (uePhy)
                {
                    cellId =
                        uePhy->GetCellId();
                }
            }
        }

        // ----------------------------------------------------
        // UE TX power
        // ----------------------------------------------------

        double ueTxPowerDbm =
            INVALID_METRIC;

        Ptr<NrUePhy> uePhy =
            NrHelper::GetUePhy(
                g_ueDevices.Get(ueId),
                0);

        if (uePhy)
        {
            ueTxPowerDbm =
                uePhy->GetTxPower();
        }

        // ----------------------------------------------------
        // Write CSV
        // ----------------------------------------------------

        g_telemetryFile
            << std::fixed
            << std::setprecision(6)

            // Timestamp
            << now << ","

            // Identity / topology
            << ueId << ","
            << cellId << ","
            << NUM_UE << ","
            << NUM_GNB << ","

            // Mobility
            << position.x << ","
            << position.y << ","
            << position.z << ","
            << speed << ","

            // Traffic
            << throughputMbps << ","
            << offeredLoadMbps << ","

            // Latency
            << latencyMs << ","
            << jitterMs << ","

            // Reliability
            << packetLossPercent << ","
            << packetDeliveryRatioPercent << ","

            // Packets
            << txPackets[ueId] << ","
            << rxPackets[ueId] << ","

            // Bytes
            << txBytes[ueId] << ","
            << rxBytes[ueId] << ","

            // Radio
            << g_radioTelemetry[ueId].rsrpDbm << ","
            << g_radioTelemetry[ueId].rsrqDb << ","
            << g_radioTelemetry[ueId].sinrDb << ","

            // RAN / PHY
            << static_cast<uint32_t>(
                   g_radioTelemetry[ueId].cqi)
            << ","

            << static_cast<uint32_t>(
                   g_radioTelemetry[ueId].mcs)
            << ","

            << static_cast<uint32_t>(
                   g_radioTelemetry[ueId].ri)
            << ","

            // Resources
            << prbUtilizationPercent << ","
            << symbolUtilizationPercent << ","
            << scheduledUes << ","

            // UE transmit power
            << ueTxPowerDbm

            << "\n";

        // ====================================================
        // Console output
        // ====================================================

        std::cout
            << "[TELEMETRY]"
            << " timestamp="
            << now

            << " ue_id="
            << ueId

            << " cell_id="
            << cellId

            << " ue_count="
            << NUM_UE

            << " gnb_count="
            << NUM_GNB

            << " ue_x="
            << position.x

            << " ue_y="
            << position.y

            << " ue_z="
            << position.z

            << " ue_speed_mps="
            << speed

            << " throughput_mbps="
            << throughputMbps

            << " offered_load_mbps="
            << offeredLoadMbps

            << " latency_ms="
            << latencyMs

            << " jitter_ms="
            << jitterMs

            << " packet_loss_percent="
            << packetLossPercent

            << " packet_delivery_ratio_percent="
            << packetDeliveryRatioPercent

            << " tx_packets="
            << txPackets[ueId]

            << " rx_packets="
            << rxPackets[ueId]

            << " tx_bytes="
            << txBytes[ueId]

            << " rx_bytes="
            << rxBytes[ueId]

            << " rsrp_dbm="
            << g_radioTelemetry[ueId].rsrpDbm

            << " rsrq_db="
            << g_radioTelemetry[ueId].rsrqDb

            << " sinr_db="
            << g_radioTelemetry[ueId].sinrDb

            << " cqi="
            << static_cast<uint32_t>(
                   g_radioTelemetry[ueId].cqi)

            << " mcs="
            << static_cast<uint32_t>(
                   g_radioTelemetry[ueId].mcs)

            << " ri="
            << static_cast<uint32_t>(
                   g_radioTelemetry[ueId].ri)

            << " prb_utilization_percent="
            << prbUtilizationPercent

            << " symbol_utilization_percent="
            << symbolUtilizationPercent

            << " scheduled_ues="
            << scheduledUes

            << " ue_tx_power_dbm="
            << ueTxPowerDbm

            << std::endl;
    }

    // ========================================================
    // IMPORTANT:
    // Flush immediately so the Digital Twin can read the file
    // while the simulation is still running.
    // ========================================================

    g_telemetryFile.flush();

    // ========================================================
    // Reset interval resource statistics
    // ========================================================

    g_resourceTelemetry.rbActiveCount = 0;
    g_resourceTelemetry.rbSamples = 0;

    g_resourceTelemetry.slotSamples = 0;

    g_resourceTelemetry.scheduledUeSum = 0.0;

    g_resourceTelemetry.symbolUtilizationSum = 0.0;

    g_lastTelemetryTime = now;

    // ========================================================
    // Schedule next telemetry sample
    // ========================================================

    if (now + TELEMETRY_INTERVAL <=
        Simulator::GetMaximumSimulationTime().GetSeconds())
    {
        Simulator::Schedule(
            Seconds(TELEMETRY_INTERVAL),
            &CollectTelemetry);
    }
}

// ============================================================
// MAIN
// ============================================================

int
main(int argc, char* argv[])
{
    // ========================================================
    // Signal handlers
    // ========================================================

    std::signal(
        SIGINT,
        SignalHandler);

    std::signal(
        SIGTERM,
        SignalHandler);

    // ========================================================
    // Command line
    // ========================================================

    CommandLine cmd(__FILE__);

    double simulationTime =
        DEFAULT_SIMULATION_TIME;

    std::string telemetryFileName =
        "ndt_telemetry.csv";

    cmd.AddValue(
        "simulationTime",
        "Simulation duration in seconds",
        simulationTime);

    cmd.AddValue(
        "telemetryFile",
        "Telemetry CSV filename",
        telemetryFileName);

    cmd.Parse(argc, argv);

    // ========================================================
    // Header
    // ========================================================

    std::cout
        << "========================================\n"
        << " NDT O-RAN Network Simulation\n"
        << " NS-3 + 5G-LENA\n"
        << "========================================\n";

    std::cout
        << "[O-RAN] Logical O-CU initialized\n"
        << "[O-RAN] Logical O-DU initialized\n"
        << "[O-RAN] Logical O-RU initialized\n";

    // ========================================================
    // Nodes
    // ========================================================

    g_gnbNodes.Create(NUM_GNB);
    g_ueNodes.Create(NUM_UE);

    // ========================================================
    // gNB mobility
    // ========================================================

    MobilityHelper gnbMobility;

    gnbMobility.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");

    gnbMobility.Install(
        g_gnbNodes);

    Ptr<MobilityModel> gnbMobilityModel =
        g_gnbNodes.Get(0)
            ->GetObject<MobilityModel>();

    gnbMobilityModel->SetPosition(
        Vector(0.0, 0.0, 10.0));

    // ========================================================
    // UE mobility
    // ========================================================

    MobilityHelper ueMobility;

    ueMobility.SetMobilityModel(
        "ns3::ConstantVelocityMobilityModel");

    ueMobility.Install(
        g_ueNodes);

    // UE 0
    Ptr<ConstantVelocityMobilityModel> ue0 =
        g_ueNodes.Get(0)
            ->GetObject<
                ConstantVelocityMobilityModel>();

    ue0->SetPosition(
        Vector(20.0, 0.0, 1.5));

    ue0->SetVelocity(
        Vector(0.0, 0.0, 0.0));

    // UE 1
    Ptr<ConstantVelocityMobilityModel> ue1 =
        g_ueNodes.Get(1)
            ->GetObject<
                ConstantVelocityMobilityModel>();

    ue1->SetPosition(
        Vector(40.0, 0.0, 1.5));

    ue1->SetVelocity(
        Vector(0.0, 0.0, 0.0));

    // ========================================================
    // NR helpers
    // ========================================================

    std::cout
        << "[NR] Creating NR helpers...\n";

    Ptr<NrPointToPointEpcHelper> nrEpcHelper =
        CreateObject<NrPointToPointEpcHelper>();

    Ptr<IdealBeamformingHelper>
        beamformingHelper =
            CreateObject<
                IdealBeamformingHelper>();

    Ptr<NrHelper> nrHelper =
        CreateObject<NrHelper>();

    nrHelper->SetEpcHelper(
        nrEpcHelper);

    nrHelper->SetBeamformingHelper(
        beamformingHelper);

    beamformingHelper->SetAttribute(
        "BeamformingMethod",
        TypeIdValue(
            DirectPathBeamforming::GetTypeId()));

    // ========================================================
    // NR spectrum
    // ========================================================

    CcBwpCreator ccBwpCreator;

    CcBwpCreator::SimpleOperationBandConf
        bandConf(
            CENTRAL_FREQUENCY,
            BANDWIDTH,
            1);

    OperationBandInfo band =
        ccBwpCreator
            .CreateOperationBandContiguousCc(
                bandConf);

    Ptr<NrChannelHelper> channelHelper =
        CreateObject<NrChannelHelper>();

    channelHelper->ConfigureFactories(
        "UMa",
        "Default",
        "ThreeGpp");

    channelHelper->SetChannelConditionModelAttribute(
        "UpdatePeriod",
        TimeValue(MilliSeconds(0)));

    channelHelper->SetPathlossAttribute(
        "ShadowingEnabled",
        BooleanValue(false));

    channelHelper->AssignChannelsToBands(
        {band});

    BandwidthPartInfoPtrVector allBwps =
        CcBwpCreator::GetAllBwps(
            {band});

    // ========================================================
    // ANTENNA CONFIGURATION
    //
    // IMPORTANT:
    // Do NOT use IsotropicAntennaModel here.
    //
    // We use the official NrHelper AntennaParams API.
    // ========================================================

    NrHelper::AntennaParams ueAntenna;

    ueAntenna.antennaElem =
        "ns3::ThreeGppAntennaModel";

    ueAntenna.nAntCols = 2;
    ueAntenna.nAntRows = 2;

    ueAntenna.nHorizPorts = 2;
    ueAntenna.nVertPorts = 1;

    ueAntenna.isDualPolarized = false;

    ueAntenna.bearingAngle = 0.0;
    ueAntenna.polSlantAngle = 0.0;

    NrHelper::AntennaParams gnbAntenna;

    gnbAntenna.antennaElem =
        "ns3::ThreeGppAntennaModel";

    gnbAntenna.nAntCols = 4;
    gnbAntenna.nAntRows = 2;

    gnbAntenna.nHorizPorts = 2;
    gnbAntenna.nVertPorts = 1;

    gnbAntenna.isDualPolarized = false;

    gnbAntenna.bearingAngle = 0.0;
    gnbAntenna.polSlantAngle = 0.0;

    nrHelper->SetupUeAntennas(
        ueAntenna);

    nrHelper->SetupGnbAntennas(
        gnbAntenna);

    // ========================================================
    // PHY
    // ========================================================

    nrHelper->SetGnbPhyAttribute(
        "Numerology",
        UintegerValue(1));

    nrHelper->SetGnbPhyAttribute(
        "TxPower",
        DoubleValue(30.0));

    nrHelper->SetUePhyAttribute(
        "TxPower",
        DoubleValue(23.0));

    // ========================================================
    // Install gNB
    // ========================================================

    std::cout
        << "[NR] Installing gNB device...\n";

    g_gnbDevices =
        nrHelper->InstallGnbDevice(
            g_gnbNodes,
            allBwps);

    // ========================================================
    // Install UE
    // ========================================================

    std::cout
        << "[NR] Installing UE devices...\n";

    g_ueDevices =
        nrHelper->InstallUeDevice(
            g_ueNodes,
            allBwps);

    // ========================================================
    // RNG streams
    // ========================================================

    nrHelper->AssignStreams(
        {.gnbDevs = g_gnbDevices,
         .ueDevs = g_ueDevices});

    // ========================================================
    // Update device configurations
    // ========================================================

    nrHelper->UpdateDeviceConfigs(
        g_gnbDevices);

    nrHelper->UpdateDeviceConfigs(
        g_ueDevices);

    // ========================================================
    // Remote host / EPC
    // ========================================================

    auto [remoteHost,
          remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost(
            "100Gb/s",
            2500,
            Seconds(0.0));

    // ========================================================
    // Internet
    // ========================================================

    InternetStackHelper internet;

    internet.Install(
        g_ueNodes);

    // ========================================================
    // UE IPv4
    // ========================================================

    g_ueIpInterfaces =
        nrEpcHelper->AssignUeIpv4Address(
            g_ueDevices);

    // ========================================================
    // UE default routes
    // ========================================================

    Ipv4StaticRoutingHelper
        routingHelper;

    for (uint32_t i = 0;
         i < NUM_UE;
         ++i)
    {
        Ptr<Ipv4> ipv4 =
            g_ueNodes.Get(i)
                ->GetObject<Ipv4>();

        Ptr<Ipv4StaticRouting>
            ueRouting =
                routingHelper.GetStaticRouting(
                    ipv4);

        ueRouting->SetDefaultRoute(
            nrEpcHelper
                ->GetUeDefaultGatewayAddress(),
            1);
    }

    // ========================================================
    // Attach UEs
    // ========================================================

    std::cout
        << "[NR] Attaching UEs to gNB...\n";

    nrHelper->AttachToClosestGnb(
        g_ueDevices,
        g_gnbDevices);

    // ========================================================
    // RADIO TRACE CONNECTIONS
    // ========================================================

    for (uint32_t i = 0;
         i < NUM_UE;
         ++i)
    {
        Ptr<NrUePhy> uePhy =
            NrHelper::GetUePhy(
                g_ueDevices.Get(i),
                0);

        if (!uePhy)
        {
            NS_FATAL_ERROR(
                "Unable to retrieve UE PHY");
        }

        // ----------------------------------------------------
        // RSRP / RSRQ
        // ----------------------------------------------------

        bool connectedRsrpRsrq =
            uePhy->TraceConnectWithoutContext(
                "ReportUeMeasurements",
                MakeBoundCallback(
                    &ReportUeMeasurements,
                    i));

        if (!connectedRsrpRsrq)
        {
            NS_LOG_WARN(
                "Could not connect "
                "ReportUeMeasurements for UE "
                << i);
        }

        // ----------------------------------------------------
        // SINR
        // ----------------------------------------------------

        bool connectedSinr =
            uePhy->TraceConnectWithoutContext(
                "DlDataSinr",
                MakeBoundCallback(
                    &ReportDlSinr,
                    i));

        if (!connectedSinr)
        {
            NS_LOG_WARN(
                "Could not connect DlDataSinr "
                "for UE "
                << i);
        }

        // ----------------------------------------------------
        // CQI / MCS / RI
        // ----------------------------------------------------

        bool connectedCqi =
            uePhy->TraceConnectWithoutContext(
                "CqiFeedbackTrace",
                MakeBoundCallback(
                    &ReportCqiMcsRi,
                    i));

        if (!connectedCqi)
        {
            NS_LOG_WARN(
                "Could not connect "
                "CqiFeedbackTrace for UE "
                << i);
        }
    }

    // ========================================================
    // gNB resource traces
    // ========================================================

    Ptr<NrGnbPhy> gnbPhy =
        NrHelper::GetGnbPhy(
            g_gnbDevices.Get(0),
            0);

    if (!gnbPhy)
    {
        NS_FATAL_ERROR(
            "Unable to retrieve gNB PHY");
    }

    // --------------------------------------------------------
    // Slot statistics
    // --------------------------------------------------------

    bool connectedSlot =
        gnbPhy->TraceConnectWithoutContext(
            "SlotDataStats",
            MakeCallback(
                &ReportSlotStats));

    if (!connectedSlot)
    {
        NS_LOG_WARN(
            "Could not connect SlotDataStats");
    }

    // --------------------------------------------------------
    // RB statistics
    // --------------------------------------------------------

    bool connectedRb =
        gnbPhy->TraceConnectWithoutContext(
            "RBDataStats",
            MakeCallback(
                &ReportRbStats));

    if (!connectedRb)
    {
        NS_LOG_WARN(
            "Could not connect RBDataStats");
    }

    // ========================================================
    // UDP TRAFFIC
    // ========================================================

    std::cout
        << "[TRAFFIC] Installing UDP traffic...\n";

    ApplicationContainer
        serverApps;

    ApplicationContainer
        clientApps;

    const uint16_t basePort = 5000;

    for (uint32_t i = 0;
         i < NUM_UE;
         ++i)
    {
        uint16_t port =
            basePort + i;

        // ----------------------------------------------------
        // UE UDP server
        // ----------------------------------------------------

        UdpServerHelper server(
            port);

        serverApps.Add(
            server.Install(
                g_ueNodes.Get(i)));

        // ----------------------------------------------------
        // Remote host UDP client
        // ----------------------------------------------------

        UdpClientHelper client(
            g_ueIpInterfaces.GetAddress(i),
            port);

        client.SetAttribute(
            "Interval",
            TimeValue(
                MilliSeconds(1)));

        client.SetAttribute(
            "PacketSize",
            UintegerValue(
                PACKET_SIZE));

        client.SetAttribute(
            "MaxPackets",
            UintegerValue(
                0xFFFFFFFF));

        clientApps.Add(
            client.Install(
                remoteHost));
    }

    // ========================================================
    // Applications
    // ========================================================

    serverApps.Start(
        Seconds(0.5));

    clientApps.Start(
        Seconds(0.5));

    serverApps.Stop(
        Seconds(simulationTime));

    clientApps.Stop(
        Seconds(simulationTime));

    // ========================================================
    // FLOW MONITOR
    // ========================================================

    FlowMonitorHelper
        flowMonitorHelper;

    g_flowMonitor =
        flowMonitorHelper.InstallAll();

    g_flowClassifier =
        DynamicCast<Ipv4FlowClassifier>(
            flowMonitorHelper.GetClassifier());

    if (!g_flowClassifier)
    {
        NS_FATAL_ERROR(
            "Unable to create IPv4 "
            "flow classifier");
    }

    // ========================================================
    // TELEMETRY FILE
    // ========================================================

    g_telemetryFile.open(
        telemetryFileName,
        std::ios::out |
        std::ios::trunc);

    if (!g_telemetryFile.is_open())
    {
        NS_FATAL_ERROR(
            "Unable to open telemetry file: "
            << telemetryFileName);
    }

    // ========================================================
    // CSV HEADER
    //
    // These are the requested DT KPIs.
    // ========================================================

    g_telemetryFile
        << "timestamp,"
        << "ue_id,"
        << "cell_id,"
        << "ue_count,"
        << "gnb_count,"
        << "ue_x,"
        << "ue_y,"
        << "ue_z,"
        << "ue_speed_mps,"
        << "throughput_mbps,"
        << "offered_load_mbps,"
        << "latency_ms,"
        << "jitter_ms,"
        << "packet_loss_percent,"
        << "packet_delivery_ratio_percent,"
        << "tx_packets,"
        << "rx_packets,"
        << "tx_bytes,"
        << "rx_bytes,"
        << "rsrp_dbm,"
        << "rsrq_db,"
        << "sinr_db,"
        << "cqi,"
        << "mcs,"
        << "ri,"
        << "prb_utilization_percent,"
        << "symbol_utilization_percent,"
        << "scheduled_ues,"
        << "ue_tx_power_dbm"
        << "\n";

    g_telemetryFile.flush();

    // ========================================================
    // STOP FILE
    //
    // Delete an old stop file before starting.
    // ========================================================

    std::remove(
        "STOP_SIMULATION");

    // ========================================================
    // TELEMETRY START
    // ========================================================

    std::cout
        << "[NDT] Starting telemetry collection...\n";

    std::cout
        << "[NDT] Telemetry file: "
        << telemetryFileName
        << "\n";

    // First sample at t = 1 second.
    Simulator::Schedule(
        Seconds(1.0),
        &CollectTelemetry);

    // ========================================================
    // STOP POLLER
    // ========================================================

    Simulator::Schedule(
        MilliSeconds(100),
        &CheckStopRequest);

    // ========================================================
    // SIMULATION STOP TIME
    // ========================================================

    std::cout
        << "[NDT] Starting "
        << simulationTime
        << " second simulation...\n";

    std::cout
        << "[NDT] To stop safely:\n"
        << "      Ctrl+C\n"
        << "      OR\n"
        << "      touch STOP_SIMULATION\n";

    Simulator::Stop(
        Seconds(simulationTime));

    // ========================================================
    // RUN
    // ========================================================

    Simulator::Run();

    // ========================================================
    // FINAL FLOW STATISTICS
    // ========================================================

    g_flowMonitor->CheckForLostPackets();

    FlowMonitor::FlowStatsContainer
        finalStats =
            g_flowMonitor->GetFlowStats();

    std::cout
        << "\n========================================\n"
        << " Final NDT Network Statistics\n"
        << "========================================\n";

    for (const auto& flow :
         finalStats)
    {
        FlowId flowId =
            flow.first;

        const FlowMonitor::FlowStats&
            stats =
                flow.second;

        Ipv4FlowClassifier::FiveTuple
            tuple =
                g_flowClassifier
                    ->FindFlow(flowId);

        int ueId =
            FindUeFromDestination(
                tuple.destinationAddress);

        if (ueId < 0)
        {
            continue;
        }

        // ----------------------------------------------------
        // Actual simulation duration
        // ----------------------------------------------------

        double actualDuration =
            Simulator::Now()
                .GetSeconds();

        if (actualDuration <= 0.0)
        {
            actualDuration = 1.0;
        }

        // ----------------------------------------------------
        // Throughput
        // ----------------------------------------------------

        double throughputMbps =
            static_cast<double>(
                stats.rxBytes) *
            8.0 /
            actualDuration /
            1e6;

        // ----------------------------------------------------
        // Offered load
        // ----------------------------------------------------

        double offeredLoadMbps =
            static_cast<double>(
                stats.txBytes) *
            8.0 /
            actualDuration /
            1e6;

        // ----------------------------------------------------
        // Latency
        // ----------------------------------------------------

        double latencyMs = 0.0;

        if (stats.rxPackets > 0)
        {
            latencyMs =
                stats.delaySum.GetSeconds() /
                static_cast<double>(
                    stats.rxPackets) *
                1000.0;
        }

        // ----------------------------------------------------
        // Jitter
        // ----------------------------------------------------

        double jitterMs = 0.0;

        if (stats.rxPackets > 0)
        {
            jitterMs =
                stats.jitterSum.GetSeconds() /
                static_cast<double>(
                    stats.rxPackets) *
                1000.0;
        }

        // ----------------------------------------------------
        // Packet loss
        // ----------------------------------------------------

        double packetLossPercent = 0.0;

        if (stats.txPackets > 0)
        {
            uint64_t lostPackets = 0;

            if (stats.txPackets >=
                stats.rxPackets)
            {
                lostPackets =
                    stats.txPackets -
                    stats.rxPackets;
            }

            packetLossPercent =
                100.0 *
                static_cast<double>(
                    lostPackets) /
                static_cast<double>(
                    stats.txPackets);
        }

        // ----------------------------------------------------
        // PDR
        // ----------------------------------------------------

        double pdrPercent = 0.0;

        if (stats.txPackets > 0)
        {
            pdrPercent =
                100.0 *
                static_cast<double>(
                    stats.rxPackets) /
                static_cast<double>(
                    stats.txPackets);
        }

        // ----------------------------------------------------
        // Output
        // ----------------------------------------------------

        std::cout
            << "UE "
            << ueId

            << " | Throughput="
            << throughputMbps
            << " Mbps"

            << " | OfferedLoad="
            << offeredLoadMbps
            << " Mbps"

            << " | Latency="
            << latencyMs
            << " ms"

            << " | Jitter="
            << jitterMs
            << " ms"

            << " | PacketLoss="
            << packetLossPercent
            << "%"

            << " | PDR="
            << pdrPercent
            << "%"

            << " | TX="
            << stats.txPackets

            << " | RX="
            << stats.rxPackets

            << std::endl;
    }

    // ========================================================
    // FINAL FLUSH
    // ========================================================

    g_telemetryFile.flush();

    g_telemetryFile.close();

    // ========================================================
    // CLEANUP
    // ========================================================

    Simulator::Destroy();

    std::cout
        << "\n========================================\n"
        << " Simulation completed successfully\n"
        << " Telemetry saved to:\n"
        << " "
        << telemetryFileName
        << "\n"
        << "========================================\n";

    return 0;
}
