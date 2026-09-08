<!-- source: https://docs.o-ran-sc.org/en/latest/architecture/architecture.html -->
# O-RAN Architecture Overview

O-RAN SC architecture follows the O-RAN Alliance defined architecture. The architecture divides
into two sides:

- **Radio side**: Near-RT RIC, O-CU-CP, O-CU-UP, O-DU, and O-RU.
- **Management side**: the Service Management and Orchestration (SMO) Framework, which contains
  a Non-RT RIC function.

## Component Definitions

- **Near-RT RIC**: a logical function that enables near-real-time control and optimization of
  O-RAN elements and resources via fine-grained data collection and actions over the E2
  interface.
- **Non-RT RIC**: a logical function that enables non-real-time control and optimization of RAN
  elements and resources, including AI/ML workflows such as model training and updates.
- **O-CU-CP**: a logical node hosting the RRC and the control-plane part of the PDCP protocol.
- **O-CU-UP**: a logical node hosting the user-plane part of the PDCP protocol and the SDAP
  protocol.
- **O-DU**: a logical node hosting RLC/MAC/High-PHY layers based on a lower-layer functional
  split.
- **O-RU**: a logical node hosting the Low-PHY layer and RF processing based on a lower-layer
  functional split.
- **xApp**: an independent software plug-in to the Near-RT RIC platform that provides functional
  extensibility.

## Interfaces

- **O1**: the interface between management entities in the Service Management and Orchestration
  Framework and O-RAN managed elements, by which FCAPS management, software management, and file
  management are achieved.
- **O1***: an interface variant supporting the Infrastructure Management Framework for virtual
  network functions.
- **E2**: connects the Near-RT RIC to RAN nodes for near-real-time control and optimization
  (fine-grained data collection and actions).

Note: this page did not describe the A1, O2, or Open Fronthaul interfaces in detail — see
`a1_interface.md` in this same corpus category for A1, sourced separately.
