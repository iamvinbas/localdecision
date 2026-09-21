import numpy as np
import pytest

from localdecision import readout as R


def biased_views(true_logits, position_prior, orders):
    """What a model with a multiplicative position preference would show for each view."""
    return [
        (order, np.array([true_logits[c] + position_prior[pos] for pos, c in enumerate(order)]))
        for order in orders
    ]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7])
def test_cyclic_views_cancel_position_bias_exactly(n):
    rng = np.random.default_rng(n)
    true_logits = rng.normal(size=n)
    prior = rng.normal(scale=2.0, size=n)  # strong, arbitrary position preference
    pooled = R.pool_views(
        n, biased_views(true_logits, prior, R.view_orders(n, "cyclic", max_views=n))
    )
    np.testing.assert_allclose(pooled.logp, R.log_softmax(true_logits), atol=1e-12)


def test_single_view_keeps_the_bias():
    true_logits = np.zeros(3)
    prior = np.array([2.0, 0.0, 0.0])
    pooled = R.pool_views(3, biased_views(true_logits, prior, R.view_orders(3, "none")))
    assert np.argmax(pooled.logp) == 0 and np.exp(pooled.logp[0]) > 0.7


def test_swap_is_exact_for_two_options():
    true_logits = np.array([0.3, -0.4])
    prior = np.array([1.7, 0.0])
    pooled = R.pool_views(2, biased_views(true_logits, prior, R.view_orders(2, "swap")))
    np.testing.assert_allclose(pooled.logp, R.log_softmax(true_logits), atol=1e-12)


def test_agreement_counts_views_that_match_the_pooled_answer():
    views = [
        ([0, 1], np.array([2.0, 0.0])),
        ([1, 0], np.array([1.0, 0.0])),
    ]  # 2nd view picks candidate 1
    pooled = R.pool_views(2, views)
    assert pooled.view_argmax == (0, 1)
    assert pooled.agreement == 0.5


def test_cyclic_orders_are_capped_and_evenly_spaced():
    orders = R.cyclic_orders(20, max_views=4)
    assert len(orders) == 4
    assert [o[0] for o in orders] == [0, 5, 10, 15]
    assert all(sorted(o) == list(range(20)) for o in orders)


def test_chunks_are_balanced_and_cover_everything():
    chunks = R.chunk_indices(77, 20)
    assert sorted(len(c) for c in chunks) == [19, 19, 19, 20]
    assert sum(chunks, []) == list(range(77))


def test_luce_chain_reproduces_the_full_softmax():
    rng = np.random.default_rng(0)
    n = 100
    logits = rng.normal(size=n)
    chunks = R.chunk_indices(n, 20)
    chunk_logp = [R.log_softmax(logits[c]) for c in chunks]
    finalists = R.select_finalists(chunks, chunk_logp, budget=26)
    final_logp = R.log_softmax(logits[finalists])
    merged = R.luce_chain(n, chunks, chunk_logp, finalists, final_logp)
    np.testing.assert_allclose(merged, R.log_softmax(logits), atol=1e-12)


def test_finalists_fit_the_budget():
    chunks = R.chunk_indices(255, 20)
    finalists = R.select_finalists(chunks, [np.zeros(len(c)) for c in chunks], budget=26)
    assert len(finalists) <= 26
    assert len(set(finalists)) == len(finalists)


def test_confidence_statistics():
    assert R.choice_confidence(np.array([1.0, 0.0, 0.0])) == 1.0
    assert R.choice_confidence(np.array([1 / 3, 1 / 3, 1 / 3])) == pytest.approx(0.0, abs=1e-12)
    assert R.choice_confidence(np.array([0.88, 0.12, 0.0])) == pytest.approx(0.82)
    assert R.margin(np.array([0.6, 0.3, 0.1])) == pytest.approx(0.3)
    assert R.entropy_confidence(np.array([0.5, 0.5])) == pytest.approx(0.0, abs=1e-12)
    assert R.expected_level(np.array([0.0, 0.95, 0.05])) == pytest.approx(1.05)


def test_temperature_keeps_argmax_and_flattens():
    logp = R.log_softmax(np.array([2.0, 0.5, 0.0]))
    hot = R.apply_temperature(logp, 3.0)
    assert np.argmax(hot) == np.argmax(logp)
    assert np.exp(hot).max() < np.exp(logp).max()


def test_conformal_set_is_ordered_and_never_empty():
    p = np.array([0.1, 0.6, 0.3])
    assert R.conformal_set(p, qhat=0.75) == [1, 2]
    assert R.conformal_set(p, qhat=0.0) == [1]
