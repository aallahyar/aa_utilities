import logging
import os

import pytest

try:
    from aa_utilities.wrappers import RSpace
except (ImportError, OSError):  # a broken import of our own code must fail, not skip
    RSpace = None


def _skip_unless_ci(reason):
    """Skips locally, but fails on CI, where R and its packages are installed and must not be silently missing."""
    if os.environ.get('CI'):
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture
def skip_unless_ci():
    return _skip_unless_ci


@pytest.fixture
def rspace(skip_unless_ci):
    if RSpace is None:
        skip_unless_ci('RSpace unavailable')
    return RSpace()


@pytest.fixture
def package_log(caplog, monkeypatch):
    """Lets `caplog` see `aa_utilities` records; the package root logger does not propagate by default."""
    monkeypatch.setattr(logging.getLogger('aa_utilities'), 'propagate', True)
    caplog.set_level(logging.DEBUG, logger='aa_utilities')
    return caplog
