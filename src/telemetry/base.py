"""Common interface for telemetry sources (prompt.md §49 — real vs mock separation).

Every source is responsible for authoritatively stamping its own `SOURCE_LABEL` onto every
record it yields, unconditionally overwriting anything a payload claims. The transport a record
arrived through is the source of truth about its provenance, not a self-reported field inside
untrusted payload content (prompt.md §0.3, §44) — this is what makes it structurally impossible
for mock data to masquerade as NS-3 output, or vice versa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar


class TelemetrySource(ABC):
    SOURCE_LABEL: ClassVar[str]

    @abstractmethod
    def records(self) -> Iterator[dict[str, Any]]:
        """Yield raw telemetry messages (wire-format dicts), each stamped with `source`."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources (sockets, files). Default no-op."""
