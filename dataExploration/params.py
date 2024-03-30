from dataclasses import dataclass, field
from simple_parsing import Serializable
from typing import List, Union

__all__ = []
__all__.extend([
    'HyperParams'
])

@dataclass
class HyperParams(Serializable):
    """ Parameters representing a complete training and exploration run """
    colname: str
    y_col: str
    per_colname: str
    keep: Union[float, bool, str]
    sample_size: int
    compare_col: str
    cols_to_drop: List[str] = field(default_factory=list)
    per_cols: List[str] = field(default_factory=list)
    compare_vals: List[float] = field(default_factory=list)
    compare_labels: List[str] = field(default_factory=list)

