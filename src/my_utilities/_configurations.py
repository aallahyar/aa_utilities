
from types import SimpleNamespace
import logging

class Configs:

    debug = False

    repository = SimpleNamespace(
        url='https://github.com/aallahyar/my_utilities/'
    )

    # NOTSET=0
    # DEBUG=10
    # INFO=20
    # WARN=30
    # ERROR=40
    # CRITICAL=50
    log = SimpleNamespace(
        level=logging.DEBUG,
    )


configs = Configs()
