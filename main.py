"""BTC trend-and-dip bot for make.fun's btc-race-24h-20261123-0511.

Trades CBBTC/USDC on 1-minute bars, long only, one position at a time,
starting from 1,000 USDC of paper money. The strategy is deliberately
simple and robust so it survives an unseen race window: it rides a
confirmed uptrend, banks the swings of the chop that follows, and sits in
cash when there is no trend at all.

  * Entry (when flat): a momentum breakout -- the trailing RET_BARS-bar
    return above ENTRY_BPS with the close above the slow EMA -- which
    catches a rally early; or a dip buy -- the close DIP_BPS below the
    rolling HIGH_BARS-bar high while the trailing TREND_BARS-bar return
    stays positive, so dips are only bought inside an uptrend.
  * Exit (when long): two modes. Rally mode, before the first exit, lets
    the big move run: a wide trailing stop (TRAIL_LOOSE_BPS), an exit on
    momentum exhaustion (a big gain from entry plus a negative 30-bar
    return), and a hard stop-loss (SL_BPS below entry). Chop mode, after
    the first exit, banks the range instead: a tighter trailing stop
    (TRAIL_TIGHT_BPS) and a take-profit (TP_BPS) on every entry, plus the
    same hard stop. A cooldown after each exit keeps the bot from
    re-entering the same chop again and again.
  * Costs: a full-size round trip pays roughly 26-34 bps in fees, slippage
    and price impact, so every signal here needs a move larger than that;
    the breakout and dip thresholds are set well above it.
  * No randomness, no wall clock: every decision is a pure function of the
    bars seen so far, so two runs on the same data give identical replies.

Speaks agent-fun/bars-replay/v1 through the kit's Python shim (vendor/).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

SYMBOL = "CBBTC"
BPS = 10_000

# --- Strategy parameters (tuned on the dev window, checked for robustness:
# every single-parameter move in the neighbourhood keeps dev PnL above 0) ---
RET_BARS = 45           # lookback for the momentum-breakout return, in bars
ENTRY_BPS = 30          # trailing RET_BARS-bar return needed to enter, in bps
SLOW = 48               # slow EMA span, in bars
TRAIL_LOOSE_BPS = 400   # rally-mode trailing stop below the peak close, in bps
TRAIL_TIGHT_BPS = 70    # chop-mode trailing stop below the peak close, in bps
EXHAUST_GAIN_BPS = 150  # gain from entry that arms the exhaustion exit, in bps
EXHAUST_BPS = -20       # 30-bar return at/below which a rally is exhausted, in bps
TP_BPS = 100            # chop-mode take-profit above entry, in bps
SL_BPS = 60             # hard stop-loss below entry, in bps
TREND_BARS = 240        # lookback for the uptrend regime filter, in bars
TREND_MIN_BPS = 50      # trailing TREND_BARS-bar return needed to buy dips, in bps
DIP_BPS = 50            # dip below the rolling high that arms a dip buy, in bps
HIGH_BARS = 60          # lookback for the rolling high, in bars
COOLDOWN_BARS = 30      # bars to wait after an exit before a fresh entry
MAX_SLIPPAGE_BPS = 200  # cap on the move from the latest close a fill may take
MAX_IMPACT_BPS = 50     # skip an entry whose own price impact exceeds this, in bps


class _Ema:
    """One exponential moving average over integer closes (float64, deterministic)."""

    def __init__(self, span):
        self.span = span
        self.alpha = 2.0 / (span + 1)
        self._warm = []
        self.value = None

    def update(self, close):
        if self.value is None:
            self._warm.append(close)
            if len(self._warm) >= self.span:
                self.value = sum(self._warm) / len(self._warm)
                self._warm = []
            return self.value
        self.value += self.alpha * (close - self.value)
        return self.value


class BtcRaceBot:
    """State for one run. Pure function of the bars: no clock, no randomness."""

    def __init__(self):
        self.ema = _Ema(SLOW)
        self.closes = []          # closes seen, oldest first
        self.peak = None          # highest close since the current entry
        self.entry = None         # close at which the current position opened
        self.exited_once = False  # False until the first exit: rally vs chop mode
        self.cooldown_until = -1  # bar index before which no fresh entry
        self.bar_no = -1          # bars of SYMBOL seen so far
        self._last_t = None

    def _ret_bps(self, bars_back):
        """Return over the last bars_back bars, in bps (int, truncated)."""
        if len(self.closes) <= bars_back:
            return None
        old = self.closes[-bars_back - 1]
        new = self.closes[-1]
        return (new - old) * BPS // old

    def on_batch(self, state, batch):
        if state.warming_up:
            return []
        bar = None
        for b in batch.get("bars", []):
            if b.get("symbol") == SYMBOL:
                bar = b
                break
        if bar is None:
            return []
        close = int(bar["c"])
        if self._last_t is not None and int(bar["t"]) == self._last_t:
            return []
        self._last_t = int(bar["t"])
        self.bar_no += 1
        self.closes.append(close)
        ema = self.ema.update(close)

        holding = state.positions.get(SYMBOL)
        qty = holding.qty if holding is not None else 0

        if qty > 0:
            if self.peak is None or close > self.peak:
                self.peak = close
            gain = (close - self.entry) * BPS // self.entry if self.entry else 0
            if self.exited_once:
                # Chop mode: bank the range with a tight trail and a take-profit.
                trail = TRAIL_TIGHT_BPS
                tp_hit = self.entry is not None and close * BPS >= self.entry * (BPS + TP_BPS)
                exhaust_hit = False
            else:
                # Rally mode: let the big move run; exit on a wide trail or on
                # momentum exhaustion after a big gain.
                trail = TRAIL_LOOSE_BPS
                tp_hit = False
                ret30 = self._ret_bps(30)
                exhaust_hit = (
                    gain >= EXHAUST_GAIN_BPS
                    and ret30 is not None
                    and ret30 <= EXHAUST_BPS
                )
            trail_hit = self.peak is not None and close * BPS <= self.peak * (BPS - trail)
            sl_hit = self.entry is not None and close * BPS <= self.entry * (BPS - SL_BPS)
            if trail_hit or exhaust_hit or tp_hit or sl_hit:
                self.cooldown_until = self.bar_no + COOLDOWN_BARS
                self.peak = None
                self.entry = None
                self.exited_once = True
                return [
                    {
                        "side": "sell",
                        "symbol": SYMBOL,
                        "qty": qty,
                        "max_slippage_bps": MAX_SLIPPAGE_BPS,
                    }
                ]
            return []

        # Flat: look for a fresh entry.
        if self.bar_no < self.cooldown_until:
            return []
        ret = self._ret_bps(RET_BARS)
        breakout = (
            ret is not None
            and ret >= ENTRY_BPS
            and ema is not None
            and close >= ema
        )
        dip = False
        if len(self.closes) > TREND_BARS:
            trend = self._ret_bps(TREND_BARS)
            high = max(self.closes[-HIGH_BARS:])
            dip = (
                trend is not None
                and trend >= TREND_MIN_BPS
                and close * BPS <= high * (BPS - DIP_BPS)
            )
        if not (breakout or dip):
            return []
        room = state.weight_room(SYMBOL)
        amount = min(state.cash, room)
        if amount < 1_000_000:
            return []
        # Never buy into a thin pool: if the trade's own price impact at the
        # latest close's depth would eat more than MAX_IMPACT_BPS, sit out
        # this bar and wait for depth (or a better price) instead.
        impact = state.impact_bps(SYMBOL, amount)
        if impact is not None and impact > MAX_IMPACT_BPS:
            return []
        self.entry = close
        self.peak = close
        return [
            {
                "side": "buy",
                "symbol": SYMBOL,
                "notional_quote_units": amount,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }
        ]


_BOT = BtcRaceBot()


def on_batch(state, batch):
    return _BOT.on_batch(state, batch)


if __name__ == "__main__":
    makefun_bars.run(on_batch)
