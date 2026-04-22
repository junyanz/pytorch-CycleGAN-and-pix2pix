import time
from typing import Optional
from loguru import logger

def format_hh_mm_ss(total_seconds: int) -> str:
    """
    Formats `total_seconds` to 00:00:00 format

    Parameters
    ----------
    total_seconds: int

    Returns
    -------
    `total_seconds` formatted as 00:00:00

    """
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class ProgressLogger:
    """
    Log progress every `log_every` * `total` iterations
    Meant to be like tqdm except logger friendly
    """

    def __init__(self, desc: str, total: int, log_every: float = 0.1, start_iter: int = 0):
        """

        Parameters
        ----------
        total: total things to iterate over
        log_every: fraction of total to output log
        """
        self._total = total
        self._start = time.time()
        self._completed = start_iter
        self._desc = desc

        self._log_every = (
            max(1, int(log_every * total)) if isinstance(log_every, float) else log_every
        )

    def log_progress(self, other: Optional[str] = None):
        """
        Log progress
        """
        self._completed += 1
        now = time.time()
        elapsed = now - self._start

        rate = self._completed / elapsed if elapsed > 0 else float("inf")
        remaining = self._total - self._completed
        eta = remaining / rate if rate > 0 else float("inf")

        if self._completed % self._log_every == 0:
            other = '' if other is None else other
            rate_log = f'{1/rate:.2f} s/it' if rate < 1 else f'{rate:.2f} it/s'
            logger.info(
                f"{self._desc}: {self._completed}/{self._total} [{format_hh_mm_ss(int(elapsed))}<{format_hh_mm_ss(int(eta))}], {rate_log}, {other}"
            )