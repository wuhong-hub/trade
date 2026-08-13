from .ma_trend import MATrendStrategy
from .momentum import MomentumStrategy
from .value import ValueStrategy
from .volume_price import VolumePriceStrategy

ALL_STRATEGIES = [
    MomentumStrategy(),
    MATrendStrategy(),
    VolumePriceStrategy(),
    ValueStrategy(),
]

STRATEGIES_BY_NAME = {s.meta.name: s for s in ALL_STRATEGIES}
