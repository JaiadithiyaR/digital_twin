<!-- source: https://docs.o-ran-sc.org/projects/o-ran-sc-nonrtric-plt-a1policymanagementservice/en/latest/overview.html -->
# O-RAN A1 Interface and Policy Management

The O-RAN Alliance defines the A1 interface to connect the Non-RealTime RAN Intelligent
Controller (Non-RT-RIC), which sits in the Service Management & Orchestration (SMO) layer, with
the Near-RealTime RAN Intelligent Controller (near-RT-RIC), which sits in the RAN. The A1
interface lets the Non-RT-RIC provide Policy Guidance to the near-RT-RIC to steer its operation.

## Purpose of A1 Policies

A1 Policy operations are orchestration and automation functions for non-real-time intelligent
management of RAN functions. Their objectives include:

- supporting non-real-time radio resource management,
- optimizing higher-layer procedures,
- optimizing RAN policies, and
- furnishing guidance, parameters, policies, and AI/ML models to support the operation of
  near-RT-RIC functions.

## A1 Policy Management Service Data Model

The A1 Policy Management Service tracks three core data categories:

1. **Policy Instances** — all A1 policy instances in the network. Each policy is targeted to a
   specific near-RT-RIC instance and is owned by a "service" (e.g. an rApp or the NONRTRIC
   Dashboard).
2. **Near-RT-RIC Inventory** — records of all near-RT-RICs in the network.
3. **Policy Types** — the policy types each near-RT-RIC supports.

## Service Behavior

The service provides a unified REST API for managing A1 policies across all near-RT-RICs, and
maintains a synchronized view of policy instances per rApp and per near-RT-RIC, and of policy
types per near-RT-RIC. It also provides a lookup service to find which near-RT-RIC controls
resources in the RAN as defined via O1 (e.g. which near-RT-RIC should be accessed to control a
given CU or DU, which in turn controls a given cell), and monitors all near-RT-RICs to maintain
data consistency (e.g. recovering state after a near-RT-RIC restart).

## Relevance to a Self-Adaptive Digital Twin

The A1 interface's role — a slower-cadence, non-real-time controller providing policy guidance
to a faster-cadence, near-real-time controller, with an explicit typed contract (policy types,
policy instances) between them — is architecturally analogous to how this project's PPO RL
Decision Agent (Module 13) selects an adaptation *strategy* that a specific adaptation *agent*
(Modules 14-16) then executes: a slower, higher-level decision governs a faster, more granular
execution layer, with a well-defined contract (the three-action strategy space) between them.
