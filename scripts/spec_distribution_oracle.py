"""Exact finite reference laws, not an emulator or proof of C++ execution."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from fractions import Fraction

Row = tuple[Fraction, ...]
History = tuple[int, ...]
Model = Callable[[History], Row]
Law = dict[History, Fraction]


def normalize(weights: Sequence[int | Fraction]) -> Row:
    if not weights or any(not isinstance(x, (int, Fraction)) or x < 0 for x in weights):
        raise ValueError("weights must be nonnegative exact numbers")
    mass = sum(weights, Fraction())
    if mass == 0:
        raise ValueError("zero mass")
    return tuple(Fraction(x) / mass for x in weights)


def validate(p: Row, q: Row) -> None:
    if len(p) != len(q) or normalize(p) != p or normalize(q) != q:
        raise ValueError("rows must have equal vocabulary and unit mass")


def conditional_kernel(p: Row, q: Row, proposal: int) -> Row:
    """Output law conditional on one proposal; acceptance uses independent U."""
    validate(p, q)
    if not 0 <= proposal < len(q) or q[proposal] == 0:
        raise ValueError("proposal must be in q support")
    accept = min(Fraction(1), p[proposal] / q[proposal])
    residual = tuple(max(Fraction(), a - b) for a, b in zip(p, q))
    mass = sum(residual, Fraction())
    result = [Fraction() for _ in p]
    result[proposal] = accept
    if accept < 1:
        assert mass > 0
        for token, weight in enumerate(residual):
            result[token] += (1 - accept) * weight / mass
    return tuple(result)


def one_token_law(p: Row, q: Row, *, actual_q: Row | None = None,
                  eligible: Callable[[int], bool] = lambda _: True) -> Row:
    """Also models proposal-dependent fallback and incorrectly recorded q."""
    validate(p, q)
    actual_q = q if actual_q is None else actual_q
    validate(p, actual_q)
    if any(a and not b for a, b in zip(actual_q, q)):
        raise ValueError("recorded q omits actual proposal support")
    result = [Fraction() for _ in p]
    for proposal, weight in enumerate(actual_q):
        if weight == 0:
            continue
        row = conditional_kernel(p, q, proposal) if eligible(proposal) else p
        for token, probability in enumerate(row):
            result[token] += weight * probability
    return tuple(result)


def block_law(target: Model, draft: Model, history: History, width: int,
              continue_at: Callable[[History], bool] = lambda _: True) -> Law:
    """Lazy proposal enumeration; continuation is decided before the next draw."""
    if not 0 <= width <= 4:
        raise ValueError("oracle width must be in [0, 4]")
    result: defaultdict[History, Fraction] = defaultdict(Fraction)

    def visit(prefix: History, remaining: int, probability: Fraction) -> None:
        p = target(history + prefix)
        if not remaining or not continue_at(history + prefix):
            for token, weight in enumerate(p):
                result[prefix + (token,)] += probability * weight
            return
        q = draft(history + prefix)
        validate(p, q)
        for token, (pt, qt) in enumerate(zip(p, q)):
            accepted = min(pt, qt)
            if accepted:
                visit(prefix + (token,), remaining - 1, probability * accepted)
            # Summing rejected proposals gives residual mass exactly, without division.
            if pt > qt:
                result[prefix + (token,)] += probability * (pt - qt)

    visit((), width, Fraction(1))
    return {tokens: mass for tokens, mass in result.items() if mass}


def stream_law(target: Model, draft: Model, horizon: int, width: int,
               eog: int | None = None,
               continue_at: Callable[[History], bool] = lambda _: True) -> Law:
    if not 0 <= horizon <= 6:
        raise ValueError("oracle horizon must be in [0, 6]")
    result: defaultdict[History, Fraction] = defaultdict(Fraction)

    def visit(history: History, probability: Fraction) -> None:
        if len(history) == horizon or (history and history[-1] == eog):
            result[history] += probability
            return
        for block, mass in block_law(target, draft, history, width, continue_at).items():
            kept = block[:horizon - len(history)]
            if eog is not None and eog in kept:
                kept = kept[:kept.index(eog) + 1]
            visit(history + kept, probability * mass)

    visit((), Fraction(1))
    return dict(result)


def sequential_law(target: Model, horizon: int, eog: int | None = None) -> Law:
    """Independent autoregressive enumeration, without the block recurrence."""
    if not 0 <= horizon <= 6:
        raise ValueError("oracle horizon must be in [0, 6]")
    pending: Law = {(): Fraction(1)}
    for _ in range(horizon):
        next_law: defaultdict[History, Fraction] = defaultdict(Fraction)
        for history, mass in pending.items():
            if history and history[-1] == eog:
                next_law[history] += mass
            else:
                for token, probability in enumerate(target(history)):
                    if probability:
                        next_law[history + (token,)] += mass * probability
        pending = dict(next_law)
    return pending
