from dataclasses import dataclass, field
from simple_parsing import Serializable
from typing import List

__all__ = []
__all__.extend([
    'HyperParams'
])

@dataclass
class HyperParams(Serializable):
    """ Parameters representing a complete training and exploration run """
    colname: str
    per_colname: str
    cols_to_drop: List[str] = field(default_factory=list)
    per_cols: List[str] = field(default_factory=list)

