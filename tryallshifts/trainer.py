from tryallshifts.model import *
from tryallshifts.dataset import D4RLTrajectoryLoader


class TASTrainer:
    def __init__(self, config) -> None:
        self.config = config
