"""Common interface for drift sources (Module 11, prompt.md §0.19, §15).

Symmetric to `src/telemetry/base.py:TelemetrySource` by design: every source is responsible for
authoritatively stamping its own `SOURCE_LABEL` onto every raw event it yields, unconditionally
overwriting anything a payload claims. This is what makes it structurally impossible for a mock
drift event to masquerade as a real external detector's output, or vice versa — the transport a
record arrived through is the source of truth about its provenance, never a self-reported field
inside untrusted payload content (prompt.md §0.3/§44, extended here from Module 2's precedent).

Only `mock_drift_source.py` implements this ABC in this project — a real external/partner
detector's own adapter (prompt.md §0.19 rule 7: "do not implement the actual drift-detection
algorithm") would implement it too, whatever its transport, without requiring
`drift_detector.py`'s validation/normalization logic to change at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar


class DriftSource(ABC):
    SOURCE_LABEL: ClassVar[str]

    @abstractmethod
    def events(self) -> Iterator[dict[str, Any]]:
        """Yield raw drift-event messages (wire-format dicts), each stamped with `source`."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources (sockets, files, timers). Default no-op."""
