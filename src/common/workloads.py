"""Token / prompt builders and (TODO) multi-turn chat trace loaders.

Exp1/Exp2 use shape-only dummy tokens (cost depends on shape(N), not values).
Exp3 needs real multi-turn traces (ShareGPT / LMSys-Chat) -> the loaders below.
"""
from .measure import make_tokens


def dummy_tokens(n):
    """(1, N) int32 dummy ids for shape-driven cost measurement (Exp1/Exp2)."""
    return make_tokens(n)


def build_chat_prompt(tokenizer, turns):
    """TODO(Exp3): apply the chat template to [{role, content}, ...] and return token ids
    via tokenizer.apply_chat_template. Used by trace replay."""
    raise NotImplementedError("Exp3: chat-prompt builder not implemented yet")


def load_multiturn_trace(path):
    """TODO(Exp3): load a multi-turn chat trace (ShareGPT / LMSys-Chat) from data/traces/
    and yield conversations (each = list of per-turn token sequences with growing context)."""
    raise NotImplementedError("Exp3: multi-turn trace loader not implemented yet")
