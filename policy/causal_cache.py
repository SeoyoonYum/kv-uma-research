"""causal_cache.py — Phase 2a: causal KV-eviction policy on the real mlx-lm cache.

Subclasses mlx-lm's `LRUPromptCache` (the engine's sequence-level prompt cache, whose
eviction is plain LRU) and replaces only the eviction VICTIM SELECTION with the Exp3
drop_gain score:

    drop_gain(seq) = keep_ms − P_reuse · recompute_ms          (evict argmax)
      keep_ms      = entry KV bytes / bandwidth   (memory freed, expressed as read-time, ms)
      recompute_ms = prefill_ms(len(tokens))      (Exp1 cost curve; superlinear in N)
      P_reuse      = exp(−idle / K)               (recency estimate — no foresight)

Identical formula to the simulated causal policy (Exp3 / src/exp3_policygap.py); the only
input is recency (P_reuse). The UNMODIFIED `LRUPromptCache` is the baseline (literally the
current-engine recency eviction). See policy/README.md for the recon + plan.

mlx-lm 0.31.3.
"""
import math

from mlx_lm.models.cache import LRUPromptCache


class CausalPromptCache(LRUPromptCache):
    def __init__(self, prefill_ms, K=8.0, bw_bytes_per_ms=92e6,
                 max_size=1 << 30, max_bytes=1 << 63):
        super().__init__(max_size=max_size, max_bytes=max_bytes)
        self._prefill_ms = prefill_ms              # callable: N tokens -> ms (Exp1 curve)
        self._K = float(K)
        self._bw = float(bw_bytes_per_ms)
        self._clock = 0
        self._last_used = {}                       # key -> logical access time
        self._lru = self._CausalOrder(self)        # swap LRU victim selection for drop_gain

    @staticmethod
    def _key(model, tokens):
        return (id(model), tuple(tokens))

    def _touch(self, model, tokens):
        self._clock += 1
        self._last_used[self._key(model, tokens)] = self._clock

    def drop_gain(self, model, tokens):
        entry = self._trie.get(model, tokens)
        if entry is None:
            return -math.inf
        keep_ms = entry.nbytes / self._bw
        recompute_ms = self._prefill_ms(len(tokens))
        idle = self._clock - self._last_used.get(self._key(model, tokens), self._clock)
        p_reuse = math.exp(-idle / self._K)
        return keep_ms - p_reuse * recompute_ms

    def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
        # Each insert (turn) is the access event that refreshes this sequence's recency.
        self._touch(model, tokens)
        super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)

    class _CausalOrder:
        """Drop-in for LRUPromptCache.CacheOrder: same push/remove/pop/len interface, but
        pop() returns the max-drop_gain victim instead of the least-recently-used one.
        (Ignores the LRU type-priority; Phase 2a uses a single cache_type.)"""

        def __init__(self, parent):
            self._p = parent
            self._items = []                       # resident (model, tokens)

        def __len__(self):
            return len(self._items)

        def push(self, model, tokens, cache_type="assistant"):
            self._items.append((model, tokens))

        def remove(self, model, tokens):
            try:
                self._items.remove((model, tokens))
            except ValueError:
                pass

        def pop(self):
            victim = max(self._items, key=lambda mt: self._p.drop_gain(*mt))
            self._items.remove(victim)
            return victim
