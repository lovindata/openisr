from dataclasses import dataclass
from typing import Literal


@dataclass
class ProcessAIVal:
    scale: Literal[2, 3, 4]
