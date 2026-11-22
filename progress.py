"""Level, score and timer, with the rules of the 2D original (M10, chapter 12.3).

    level          starts at 1, increments only after the exit is reached with
                   all four apples collected
    score          grows by the apple values and persists across levels
    collected      resets to 0 at the start of every level
    timer          wall clock since the run started; persists across levels
    restart (R)    level back to 1, score back to 0, collected back to 0, clock
                   restarted, exactly what reset_game() does in the source

The state is deliberately separate from the maze: a new level rebuilds the
maze, the puddles, the birds and the apples, but the score and the clock
survive that rebuild.
"""
from apples import APPLE_TOTAL


class Progress:
    """Score, level and clock for one run."""

    def __init__(self, now, level=1):
        self.level = int(level)
        self.score = 0
        self.collected = 0
        self.start_time = float(now)
        self.levels_cleared = 0

    # -- apples ---------------------------------------------------------
    def collect(self, taken):
        """Register the apples returned by apples.collect()."""
        for apple in taken:
            self.collected += 1
            self.score += apple.value

    def level_ready(self):
        return self.collected >= APPLE_TOTAL

    def missing(self):
        return max(0, APPLE_TOTAL - self.collected)

    # -- level ----------------------------------------------------------
    def advance(self):
        """Called after the exit was reached with every apple collected."""
        self.level += 1
        self.collected = 0
        self.levels_cleared += 1
        return self.level

    def reset(self, now):
        self.level = 1
        self.score = 0
        self.collected = 0
        self.levels_cleared = 0
        self.start_time = float(now)

    # -- clock ----------------------------------------------------------
    def elapsed(self, now):
        return max(0.0, float(now) - self.start_time)

    def clock(self, now):
        return format_clock(self.elapsed(now))


def format_clock(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"
