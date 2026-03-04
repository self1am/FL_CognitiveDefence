from .cognitive_server import CognitiveAggregationStrategy
from .cognitive_defence_posg_server import POSGAggregationStrategy
from .no_defence_server import NoDefenceAggregationStrategy
from .krum_server import KrumAggregationStrategy
from .trimmed_mean_server import TrimmedMeanAggregationStrategy
from .vert_server import VERTAggregationStrategy

__all__ = [
    'CognitiveAggregationStrategy',
    'POSGAggregationStrategy',
    'NoDefenceAggregationStrategy',
    'KrumAggregationStrategy',
    'TrimmedMeanAggregationStrategy',
    'VERTAggregationStrategy'
]
