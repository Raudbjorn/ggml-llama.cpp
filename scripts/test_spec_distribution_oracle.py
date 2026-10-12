"""Exact finite checks and counterexamples for the distribution proof."""

from fractions import Fraction as F
import itertools
import struct
import unittest

from scripts.spec_distribution_oracle import (
    block_law, conditional_kernel, normalize, one_token_law,
    sequential_law, stream_law,
)


def compositions(total, length):
    if length == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, length - 1):
                yield (first,) + rest


class DistributionOracleTests(unittest.TestCase):
    def test_all_small_rational_simplexes(self):
        checked = 0
        for vocab in range(2, 5):
            for denominator in range(1, 6):
                rows = [normalize(row) for row in compositions(denominator, vocab)]
                for p, q in itertools.product(rows, repeat=2):
                    self.assertEqual(one_token_law(p, q), p)
                    checked += 1
        print(f"exact single-token pairs checked: {checked}")

    def test_conditional_rejection_and_bonus(self):
        p, q = normalize((3, 1)), normalize((1, 1))
        self.assertEqual(conditional_kernel(p, q, 1), normalize((1, 1)))
        law = block_law(lambda _: p, lambda _: q, (), 1)
        self.assertEqual(law, {(0, 0): F(3, 8), (0, 1): F(1, 8),
                               (1, 0): F(3, 16), (1, 1): F(1, 16), (0,): F(1, 4)})

    def test_history_dependent_joint_law_and_stopping(self):
        def p(history):
            return normalize((2, 1, 0) if not history or history[-1] == 0 else (1, 2, 3))

        def q(history):
            return normalize((1, 3, 2) if len(history) % 2 else (4, 0, 1))

        for width, eog in itertools.product(range(5), (None, 0, 2)):
            with self.subTest(width=width, eog=eog):
                self.assertEqual(stream_law(p, q, 4, width, eog), sequential_law(p, 4, eog))
                self.assertEqual(stream_law(p, q, 4, width, eog,
                                           lambda h: not h or h[-1] != 1),
                                 sequential_law(p, 4, eog))

    def test_identical_disjoint_tiny_and_nearly_equal(self):
        tiny = F(1, 10**100)
        for p, q in [(normalize((1, 0)), normalize((0, 1))),
                     ((tiny, 1 - tiny), (1 - tiny, tiny)),
                     ((F(1, 2), F(1, 2)), (F(1, 2) - tiny, F(1, 2) + tiny)),
                     (normalize((2, 3, 5)), normalize((2, 3, 5)))]:
            self.assertEqual(one_token_law(p, q), p)

    def test_minimum_length_selection_bias_counterexample(self):
        # n_min=2, p_min=.6: next q is (.9,.1) after A, (.5,.5) after B.
        # Clearing the short B proposal retrospectively conditions the first draw.
        p, q = normalize((1, 1)), normalize((3, 1))
        biased = one_token_law(p, q, eligible=lambda proposal: proposal == 0)
        self.assertEqual(biased, (F(5, 8), F(3, 8)))
        self.assertNotEqual(biased, p)
        self.assertEqual(one_token_law(p, q, eligible=lambda _: False), p)

    def test_invalid_suffix_selection_bias_counterexample(self):
        p, q = normalize((3, 1)), normalize((1, 1))
        self.assertEqual(one_token_law(p, q, eligible=lambda x: x == 0), (F(7, 8), F(1, 8)))

    def test_recorded_float_normalization_differs_from_actual_weights(self):
        def f32(value):
            return struct.unpack("f", struct.pack("f", value))[0]

        # Exact float inputs; emulate only the final f32 normalization store.
        weights = tuple(F.from_float(f32(value)) for value in (1, .1, .01))
        actual = normalize(weights)
        recorded = normalize(tuple(F.from_float(f32(float(x))) for x in actual))
        self.assertNotEqual(recorded, actual)
        target = normalize((1, 1, 1))
        self.assertNotEqual(one_token_law(target, recorded, actual_q=actual), target)
        self.assertEqual(one_token_law(target, recorded), target)

    def test_finite_grid_is_not_continuous_uniform(self):
        p, q = normalize((1, 1)), normalize((1, 3))
        # Four equiprobable uniform grid values, independent for accept/correction.
        grid = tuple(F(i, 4) for i in range(4))
        law = [F(), F()]
        for proposed, mass in enumerate(q):
            for u in grid:
                token = proposed if u * q[proposed] < p[proposed] else 0
                law[token] += mass / len(grid)
        self.assertEqual(tuple(law), (F(7, 16), F(9, 16)))
        self.assertNotEqual(tuple(law), p)

    def test_zero_boundary_does_not_select_zero_mass(self):
        weights, u = (F(0), F(1)), F(0)
        cumulative = F()
        old = new = None
        for token, weight in enumerate(weights):
            cumulative += weight
            if old is None and cumulative >= u:
                old = token
            if new is None and weight > 0 and cumulative > u:
                new = token
        self.assertEqual((old, new), (0, 1))

    def test_invalid_inputs_and_bounds(self):
        for values in ((), (0, 0), (-1, 2), (.5, .5)):
            with self.assertRaises(ValueError):
                normalize(values)
        with self.assertRaises(ValueError):
            conditional_kernel(normalize((1, 1)), normalize((1, 0)), 1)
        with self.assertRaises(ValueError):
            one_token_law((F(1), F(1)), normalize((1, 1)))
        with self.assertRaises(ValueError):
            block_law(lambda _: normalize((1,)), lambda _: normalize((1,)), (), 5)
        with self.assertRaises(ValueError):
            stream_law(lambda _: normalize((1,)), lambda _: normalize((1,)), 7, 1)


if __name__ == "__main__":
    unittest.main()
