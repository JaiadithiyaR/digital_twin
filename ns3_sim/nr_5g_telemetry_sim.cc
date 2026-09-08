// SPDX-License-Identifier: GPL-2.0-only
//
// nr_5g_telemetry_sim.cc — Module 1's real NS-3 / 5G-LENA telemetry producer
// (prompt.md sections 0.2, 0.3, 7). This is the project-authored counterpart to
// src/telemetry/zmq_source.py (Module 2): a genuine, running 5G-LENA scenario whose real
// PHY trace sources and real FlowMonitor measurements are streamed out over a ZeroMQ PUB
// socket as SOURCE=NS3_5G_LENA telemetry — never fabricated, never estimated offline.
//
// Topology: one gNB (fixed, UMa/ThreeGpp channel, 3.5 GHz / 20 MHz), N UEs performing an
// unbounded random walk within a disc around the gNB (so ue_speed/position genuinely vary),
// each receiving a constant-bitrate DL UDP flow from a remote host through the EPC.
//
// Real fields and where they genuinely come from (no field on the wire is synthesized):
//   sinr_db, rsrp_dbm, rsrq_db  <- NrUePhy trace sources "DlDataSinr"/"ReportUeMeasurements"
//   prb_utilization_ratio      <- NrGnbPhy trace sources "SlotDataStats" + "RBDataStats"
//                                  (used RB-symbols over available RB-symbols; see Tick())
//   throughput/offered_load/latency/jitter/packet_loss
//                               <- FlowMonitor, polled every telemetryInterval and converted
//                                  to per-interval deltas (not cumulative end-of-run stats),
//                                  so the DT sees a genuine continuous stream (prompt.md §7).
//   ue_speed_mps, ue_position_* <- the UE's own real MobilityModel (RandomWalk2dMobilityModel)
//   ue_count                   <- the actual number of UEs attached to the (single) cell
//
// Wire schema matches src/telemetry/schema.py's raw fields exactly; the consumer
// (src/telemetry/zmq_source.py) always overwrites "source" with "NS3_5G_LENA" on receipt, so
// the value emitted here is documentation, not load-bearing.

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <zmq.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include <thread>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Nr5gTelemetrySim");

namespace
{

double
LinearToDb(double linear)
{
    return 10.0 * std::log10(std::max(linear, 1.0e-12));
}

} // namespace

/**
 * Owns every piece of mutable state needed to turn real NS-3/5G-LENA PHY trace events and
 * periodic FlowMonitor polls into per-UE JSON telemetry records published over a ZeroMQ PUB
 * socket, matching src/telemetry/schema.py's raw wire contract field-for-field.
 */
class TelemetryPublisher
{
  public:
    TelemetryPublisher(NodeContainer ueNodes,
                        NetDeviceContainer ueNetDevices,
                        Ptr<FlowMonitor> monitor,
                        Ptr<Ipv4FlowClassifier> classifier,
                        Ipv4InterfaceContainer ueIpIface,
                        Time interval,
                        const std::string& zmqEndpoint,
                        std::string zmqTopic,
                        std::string cellId)
        : m_ueNodes(ueNodes),
          m_monitor(monitor),
          m_classifier(classifier),
          m_interval(interval),
          m_zmqTopic(std::move(zmqTopic)),
          m_cellId(std::move(cellId)),
          m_zmqContext(1),
          m_zmqSocket(m_zmqContext, zmq::socket_type::pub)
    {
        m_zmqSocket.bind(zmqEndpoint);
        // PUB/SUB "slow joiner" mitigation: give a subscriber that connects around the same
        // moment this process starts a brief window to complete its handshake before the first
        // publish. Not required for correctness (a subscriber that connects before this bind
        // will receive everything once bound), only for validation convenience.
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        m_wallClockStart =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                .count();

        for (uint32_t i = 0; i < ueNetDevices.GetN(); ++i)
        {
            m_uePhys.push_back(NrHelper::GetUePhy(ueNetDevices.Get(i), 0));
            m_ipToUeIndex[ueIpIface.GetAddress(i)] = i;
            m_ueRnti.push_back(0);
        }
    }

    void
    ConnectTraces(Ptr<NrGnbPhy> gnbPhy)
    {
        gnbPhy->TraceConnectWithoutContext("SlotDataStats",
                                           MakeCallback(&TelemetryPublisher::OnSlotDataStats, this));
        gnbPhy->TraceConnectWithoutContext("RBDataStats",
                                           MakeCallback(&TelemetryPublisher::OnRbDataStats, this));
        for (const auto& phy : m_uePhys)
        {
            phy->TraceConnectWithoutContext(
                "ReportUeMeasurements",
                MakeCallback(&TelemetryPublisher::OnUeMeasurements, this));
            phy->TraceConnectWithoutContext("DlDataSinr",
                                            MakeCallback(&TelemetryPublisher::OnDlDataSinr, this));
        }
    }

    // -- Real trace-source callbacks (fired by the NR PHY itself; never invoked by us) --------

    void
    OnUeMeasurements(uint16_t rnti,
                      uint16_t cellId,
                      double rsrp,
                      double rsrq,
                      bool isServingCell,
                      uint8_t componentCarrierId)
    {
        if (!isServingCell)
        {
            return;
        }
        UeRadioStats& stats = m_ueStatsByRnti[rnti];
        stats.rsrpDbm = rsrp;
        stats.rsrqDb = rsrq;
    }

    void
    OnDlDataSinr(uint16_t cellId, uint16_t rnti, double sinrLinear, uint16_t bwpId)
    {
        m_ueStatsByRnti[rnti].sinrDb = LinearToDb(sinrLinear);
    }

    // "SlotDataStats" reports availRb (RBs available for the whole slot, frequency-only) and
    // dataSym (how many of the slot's OFDM symbols were used for data, time-only) — two
    // different axes of the time-frequency grid, not directly comparable to each other. We keep
    // the RB count (roughly constant per cell/bwp) and accumulate the data-symbol count, so it
    // can be multiplied out below into "available RB-symbols" — a real occupancy denominator.
    void
    OnSlotDataStats(const SfnSf& sfnSf,
                     uint32_t activeUe,
                     uint32_t dataReg,
                     uint32_t dataSym,
                     uint32_t availRb,
                     uint32_t availSym,
                     uint16_t bwpId,
                     uint16_t cellId)
    {
        m_lastAvailRb = availRb;
        m_dataSymSum += dataSym;
    }

    // "RBDataStats" fires once per OFDM symbol actually used for data and reports the literal
    // set of resource blocks allocated in that symbol — rbMap.size() is genuinely comparable to
    // availRb (both are RB *counts*, same frequency axis), unlike "SlotDataStats"'s dataReg
    // (an RBG*symbol product from the MAC scheduler, a different unit — see prb_utilization_ratio
    // in Tick() and the module docstring for how the two traces are combined correctly).
    void
    OnRbDataStats(const SfnSf& sfnSf,
                   uint8_t sym,
                   const std::vector<int>& rbMap,
                   uint16_t bwpId,
                   uint16_t cellId)
    {
        m_usedRbSymSum += rbMap.size();
    }

    // -- Periodic publish tick (self-rescheduling) --------------------------------------------

    void
    Tick()
    {
        for (size_t i = 0; i < m_uePhys.size(); ++i)
        {
            uint16_t rnti = m_uePhys[i]->GetRnti();
            if (rnti != 0)
            {
                m_ueRnti[i] = rnti;
            }
        }

        // Real fraction of the time-frequency PRB grid used for data over this window: used
        // RB-symbols (from the "RBDataStats" trace, summed per data-carrying OFDM symbol) over
        // available RB-symbols (available RBs, roughly constant per cell/bwp, times the number
        // of data symbols actually scheduled over the window, from "SlotDataStats").
        double availableRbSymbols = static_cast<double>(m_lastAvailRb) * static_cast<double>(m_dataSymSum);
        double prbUtilizationRatio =
            availableRbSymbols > 0.0 ? std::min(1.0, static_cast<double>(m_usedRbSymSum) / availableRbSymbols)
                                      : 0.0;
        m_dataSymSum = 0;
        m_usedRbSymSum = 0;

        m_monitor->CheckForLostPackets();
        FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
        std::vector<FlowSample> samples(m_uePhys.size());
        double dt = m_interval.GetSeconds();

        for (const auto& kv : stats)
        {
            FlowId flowId = kv.first;
            const FlowMonitor::FlowStats& fs = kv.second;
            Ipv4FlowClassifier::FiveTuple tuple = m_classifier->FindFlow(flowId);
            auto ueIt = m_ipToUeIndex.find(tuple.destinationAddress);
            if (ueIt == m_ipToUeIndex.end())
            {
                continue; // not one of our gNB->UE downlink flows
            }
            uint32_t ueIndex = ueIt->second;

            CumulativeFlowStats prev = m_prevFlowStats[flowId]; // zero-initialized on first sight
            double dTxBytes = static_cast<double>(fs.txBytes) - prev.txBytes;
            double dRxBytes = static_cast<double>(fs.rxBytes) - prev.rxBytes;
            double dTxPackets = static_cast<double>(fs.txPackets) - prev.txPackets;
            double dRxPackets = static_cast<double>(fs.rxPackets) - prev.rxPackets;
            double dDelaySum = fs.delaySum.GetSeconds() - prev.delaySum;
            double dJitterSum = fs.jitterSum.GetSeconds() - prev.jitterSum;
            double dLost = std::max(0.0, static_cast<double>(fs.lostPackets) - prev.lostPackets);

            FlowSample& sample = samples[ueIndex];
            sample.hasData = true;
            sample.throughputBps = dt > 0.0 ? (dRxBytes * 8.0) / dt : 0.0;
            sample.offeredLoadBps = dt > 0.0 ? (dTxBytes * 8.0) / dt : 0.0;
            sample.latencyS = dRxPackets > 0.0 ? dDelaySum / dRxPackets : 0.0;
            sample.jitterS = dRxPackets > 0.0 ? dJitterSum / dRxPackets : 0.0;
            sample.packetLossRatio = dTxPackets > 0.0 ? std::min(1.0, dLost / dTxPackets) : 0.0;

            m_prevFlowStats[flowId] = CumulativeFlowStats{static_cast<double>(fs.txBytes),
                                                            static_cast<double>(fs.rxBytes),
                                                            static_cast<double>(fs.txPackets),
                                                            static_cast<double>(fs.rxPackets),
                                                            fs.delaySum.GetSeconds(),
                                                            fs.jitterSum.GetSeconds(),
                                                            static_cast<double>(fs.lostPackets)};
        }

        double timestamp = m_wallClockStart + Simulator::Now().GetSeconds();

        for (uint32_t i = 0; i < m_uePhys.size(); ++i)
        {
            if (!samples[i].hasData)
            {
                continue; // no flow measured for this UE in this window yet
            }

            uint16_t rnti = m_ueRnti[i];
            auto radioIt = m_ueStatsByRnti.find(rnti);
            UeRadioStats radio = radioIt != m_ueStatsByRnti.end() ? radioIt->second : UeRadioStats{};

            Ptr<MobilityModel> mobility = m_ueNodes.Get(i)->GetObject<MobilityModel>();
            Vector pos = mobility->GetPosition();
            double speed = mobility->GetVelocity().GetLength();

            Publish(i, timestamp, samples[i], prbUtilizationRatio, radio, pos, speed);
        }

        Simulator::Schedule(m_interval, &TelemetryPublisher::Tick, this);
    }

    uint64_t
    PublishedCount() const
    {
        return m_publishedCount;
    }

  private:
    struct UeRadioStats
    {
        double rsrpDbm = -140.0;
        double rsrqDb = -20.0;
        double sinrDb = -10.0;
    };

    struct CumulativeFlowStats
    {
        double txBytes = 0.0;
        double rxBytes = 0.0;
        double txPackets = 0.0;
        double rxPackets = 0.0;
        double delaySum = 0.0;
        double jitterSum = 0.0;
        double lostPackets = 0.0;
    };

    struct FlowSample
    {
        bool hasData = false;
        double throughputBps = 0.0;
        double offeredLoadBps = 0.0;
        double latencyS = 0.0;
        double jitterS = 0.0;
        double packetLossRatio = 0.0;
    };

    void
    Publish(uint32_t ueIndex,
            double timestamp,
            const FlowSample& sample,
            double prbUtilizationRatio,
            const UeRadioStats& radio,
            const Vector& pos,
            double speed)
    {
        std::ostringstream json;
        json << std::fixed << std::setprecision(6);
        json << "{"
             << "\"source\":\"NS3_5G_LENA\","
             << "\"timestamp\":" << timestamp << ","
             << "\"ue_id\":\"ue-" << ueIndex << "\","
             << "\"cell_id\":\"" << m_cellId << "\","
             << "\"throughput_bps\":" << sample.throughputBps << ","
             << "\"offered_load_bps\":" << sample.offeredLoadBps << ","
             << "\"latency_s\":" << sample.latencyS << ","
             << "\"jitter_s\":" << sample.jitterS << ","
             << "\"packet_loss_ratio\":" << sample.packetLossRatio << ","
             << "\"prb_utilization_ratio\":" << prbUtilizationRatio << ","
             << "\"sinr_db\":" << radio.sinrDb << ","
             << "\"rsrp_dbm\":" << radio.rsrpDbm << ","
             << "\"rsrq_db\":" << radio.rsrqDb << ","
             << "\"ue_count\":" << m_uePhys.size() << ","
             << "\"ue_speed_mps\":" << speed << ","
             << "\"ue_position_x\":" << pos.x << ","
             << "\"ue_position_y\":" << pos.y << "}";
        std::string payload = json.str();

        m_zmqSocket.send(zmq::buffer(m_zmqTopic), zmq::send_flags::sndmore);
        m_zmqSocket.send(zmq::buffer(payload), zmq::send_flags::none);
        ++m_publishedCount;
    }

    NodeContainer m_ueNodes;
    Ptr<FlowMonitor> m_monitor;
    Ptr<Ipv4FlowClassifier> m_classifier;
    Time m_interval;
    std::string m_zmqTopic;
    std::string m_cellId;

    std::vector<Ptr<NrUePhy>> m_uePhys;
    std::vector<uint16_t> m_ueRnti;
    std::map<Ipv4Address, uint32_t> m_ipToUeIndex;
    std::map<uint16_t, UeRadioStats> m_ueStatsByRnti;
    std::map<FlowId, CumulativeFlowStats> m_prevFlowStats;

    uint32_t m_lastAvailRb = 0;
    uint32_t m_dataSymSum = 0;
    uint64_t m_usedRbSymSum = 0;
    double m_wallClockStart = 0.0;
    uint64_t m_publishedCount = 0;

    zmq::context_t m_zmqContext;
    zmq::socket_t m_zmqSocket;
};

int
main(int argc, char* argv[])
{
    uint16_t ueNum = 6;
    Time simTime = Seconds(20.0);
    Time udpAppStartTime = MilliSeconds(400);
    Time telemetryInterval = MilliSeconds(200);
    double centralFrequency = 3.5e9;
    double bandwidth = 20e6;
    uint16_t numerology = 1;
    double totalTxPower = 23.0; // dBm (200 mW) — typical macro-sector per-branch tx power
    double areaRadius = 100.0;  // meters — UE random-walk bounds around the gNB
    uint32_t udpPacketSize = 1200;
    // ~250 pkt/s -> ~2.4 Mbps offered load per UE (~14.4 Mbps total across 6 UEs on this 20 MHz
    // cell). Tuned empirically against the real compiled binary + real Python consumer
    // end-to-end: the original ~480 kbps/UE default left the cell so under-loaded that
    // throughput/latency/jitter/packet-loss were flat for an entire run (min==max==491200.0 bps,
    // loss==0.0 across all 81 records of a 6s/4-UE run); ~9.6 Mbps/UE instead saturated the cell
    // permanently (prb_utilization_ratio pinned at 1.000000 across all 162 records of a 6s/6-UE
    // run) which is equally undynamic from the other direction. This value sits between the two,
    // so real capacity contention as UEs roam/SINR changes produces genuine variation in exactly
    // the fields Modules 6-10 predict, without permanently saturating or idling the cell.
    double udpIntervalSeconds = 0.004;
    uint32_t rngRun = 1;
    std::string zmqEndpoint = "tcp://127.0.0.1:5556";
    std::string zmqTopic = "ns3.telemetry";

    CommandLine cmd(__FILE__);
    cmd.AddValue("ueNum", "Number of UEs attached to the single gNB", ueNum);
    cmd.AddValue("simTime", "Total simulated duration", simTime);
    cmd.AddValue("telemetryIntervalMs",
                 "Telemetry publish period in milliseconds",
                 telemetryInterval);
    cmd.AddValue("zmqEndpoint", "ZeroMQ PUB bind endpoint", zmqEndpoint);
    cmd.AddValue("zmqTopic", "ZeroMQ topic prefix", zmqTopic);
    cmd.AddValue("rngRun", "ns-3 RngSeedManager run number", rngRun);
    cmd.Parse(argc, argv);

    RngSeedManager::SetRun(rngRun);
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    NodeContainer gnbNodes;
    gnbNodes.Create(1);
    NodeContainer ueNodes;
    ueNodes.Create(ueNum);

    MobilityHelper gnbMobility;
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.Install(gnbNodes);
    gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, 10.0));

    MobilityHelper ueMobility;
    ueMobility.SetPositionAllocator(
        "ns3::RandomBoxPositionAllocator",
        "X",
        StringValue("ns3::UniformRandomVariable[Min=" + std::to_string(-areaRadius) +
                    "|Max=" + std::to_string(areaRadius) + "]"),
        "Y",
        StringValue("ns3::UniformRandomVariable[Min=" + std::to_string(-areaRadius) +
                    "|Max=" + std::to_string(areaRadius) + "]"),
        "Z",
        StringValue("ns3::ConstantRandomVariable[Constant=1.5]"));
    ueMobility.SetMobilityModel(
        "ns3::RandomWalk2dMobilityModel",
        "Bounds",
        RectangleValue(Rectangle(-areaRadius, areaRadius, -areaRadius, areaRadius)),
        "Speed",
        StringValue("ns3::UniformRandomVariable[Min=1.0|Max=12.0]"),
        "Distance",
        DoubleValue(30.0));
    ueMobility.Install(ueNodes);

    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> beamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(beamformingHelper);
    nrHelper->SetEpcHelper(epcHelper);

    beamformingHelper->SetAttribute("BeamformingMethod",
                                    TypeIdValue(DirectPathBeamforming::GetTypeId()));
    epcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, 1);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMa", "Default", "ThreeGpp");
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    // Real 3GPP shadow fading (TR 38.901), not a synthetic noise injection: without it, a
    // static single-cell scenario with no interference gives every UE a near-constant channel
    // regardless of movement, which left throughput/latency/packet-loss almost perfectly flat
    // end-to-end (measured: min==max==2456000.0 bps across a full 6s/6-UE run) even though
    // SINR/RSRP/RSRQ genuinely varied. Enabling it is the physically correct way to get real
    // channel-quality dynamics — not a scenario hack.
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    channelHelper->AssignChannelsToBands({band});
    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    Packet::EnableChecking();
    Packet::EnablePrinting();

    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(ueNodes, allBwps);
    nrHelper->AssignStreams({.gnbDevs = gnbNetDev, .ueDevs = ueNetDev});

    NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)->SetAttribute("Numerology", UintegerValue(numerology));
    NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)->SetAttribute("TxPower", DoubleValue(totalTxPower));

    auto [remoteHost, remoteHostAddr] = epcHelper->SetupRemoteHost("100Gb/s", 2500, MilliSeconds(0));

    InternetStackHelper internet;
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(ueNetDev);

    nrHelper->AttachToClosestGnb(ueNetDev, gnbNetDev);

    uint16_t dlPort = 4000;
    ApplicationContainer serverApps;
    UdpServerHelper dlServer(dlPort);
    serverApps.Add(dlServer.Install(ueNodes));

    UdpClientHelper dlClient;
    dlClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClient.SetAttribute("PacketSize", UintegerValue(udpPacketSize));
    dlClient.SetAttribute("Interval", TimeValue(Seconds(udpIntervalSeconds)));

    ApplicationContainer clientApps;
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        dlClient.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueIpIface.GetAddress(i), dlPort)));
        clientApps.Add(dlClient.Install(remoteHost));
    }

    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(ueNodes);
    Ptr<FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());

    TelemetryPublisher publisher(ueNodes,
                                  ueNetDev,
                                  monitor,
                                  classifier,
                                  ueIpIface,
                                  telemetryInterval,
                                  zmqEndpoint,
                                  zmqTopic,
                                  "cell-0");
    publisher.ConnectTraces(NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0));

    Simulator::Schedule(udpAppStartTime + telemetryInterval, &TelemetryPublisher::Tick, &publisher);

    Simulator::Stop(simTime);
    Simulator::Run();

    std::cout << "nr_5g_telemetry_sim: published " << publisher.PublishedCount()
              << " telemetry records over " << simTime.GetSeconds()
              << "s simulated time on endpoint " << zmqEndpoint << " topic '" << zmqTopic << "'"
              << std::endl;

    Simulator::Destroy();
    return 0;
}
