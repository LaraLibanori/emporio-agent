from datetime import datetime

import pytest

from emporio.config import STORE_TZ, Settings
from emporio.tools import ToolContext
from emporio.tools import orders as orders_mod


def make_ctx(tmp_path, when="2026-03-25T10:00", session="test"):
    dt = datetime.fromisoformat(when).replace(tzinfo=STORE_TZ)
    return ToolContext(session_id=session, settings=Settings(state_dir=tmp_path), clock=lambda: dt)


@pytest.fixture
def ctx(tmp_path):
    orders_mod._failed_attempts.clear()
    return make_ctx(tmp_path)


@pytest.fixture
def ctx_at(tmp_path):
    orders_mod._failed_attempts.clear()
    return lambda when, session="test": make_ctx(tmp_path, when, session)
