"""Early stopping based on a monitored validation metric."""


class EarlyStopping:
    """Stop training when the monitored value stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        "min" (lower is better, e.g. val loss) or "max".

    Attributes
    ----------
    best_score : float or None
        Best monitored value seen so far.
    best_epoch : int
        1-based epoch index of the best value.
    counter : int
        Consecutive non-improving epochs.
    should_stop : bool
        True once patience has been exceeded.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self._epoch = 0

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def step(self, current_value: float) -> bool:
        """Record a new monitored value. Returns True if it is a new best."""
        self._epoch += 1

        if self.best_score is None or self._is_improvement(current_value):
            self.best_score = current_value
            self.best_epoch = self._epoch
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
