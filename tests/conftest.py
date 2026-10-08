import logging

import pytest

try:
    from aa_utilities.wrappers import RSpace
except Exception:
    RSpace = None


@pytest.fixture
def rspace():
    if RSpace is None:
        pytest.skip('RSpace unavailable')
    return RSpace()


@pytest.fixture
def package_log(caplog, monkeypatch):
    """Lets `caplog` see `aa_utilities` records; the package root logger does not propagate by default."""
    monkeypatch.setattr(logging.getLogger('aa_utilities'), 'propagate', True)
    caplog.set_level(logging.DEBUG, logger='aa_utilities')
    return caplog
