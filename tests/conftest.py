from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from swarmguard.data.schemas import Event, events_to_frame

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def ev(eid, minute, actor, content="", action="message", mentions=(), resources=(), target=None,
       target_type=None, actor_type="agent", tool=None) -> Event:
    return Event(eid, T0 + timedelta(minutes=minute), actor, actor_type, action, target, target_type, content,
                 "test", {"raw_id": eid, "mentions": list(mentions), "resources": list(resources), "tool": tool})


@pytest.fixture
def frame():
    def _f(events) -> pd.DataFrame:
        return events_to_frame(events)
    return _f
