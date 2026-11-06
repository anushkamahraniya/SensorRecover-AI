"""Checks that the model and the evaluation behave as claimed.

Run with:  python -m pytest tests -q
"""

import numpy as np
import pytest

from models.hopfield import (
    BinaryHopfield,
    BinaryHopfieldImputer,
    ContinuousHopfield,
    ThermometerEncoder,
    build_windows,
)
from utils.evaluation import interpolate_impute, mean_impute, score
from utils.preprocessing import (
    UCI_CSV_PATH,
    inject_noise,
    load_uci_raw,
    load_uploaded_csv,
    outage_missing_mask,
    prepare_uci,
    random_missing_mask,
    select_features,
    split_and_scale,
)


@pytest.fixture
def correlated():
    """Synthetic test fixture only. It is never used for reported results."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(400, 2))
    mix = rng.normal(size=(2, 6))
    return base @ mix + 0.05 * rng.normal(size=(400, 6))


# ---- classical network ---------------------------------------------------- #
def test_weight_matrix_is_symmetric_with_zero_diagonal():
    patterns = np.random.default_rng(1).choice([-1.0, 1.0], size=(5, 60))
    for rule in ("hebbian", "pseudoinverse"):
        w = BinaryHopfield(rule).fit(patterns).weights
        assert np.allclose(w, w.T)
        assert np.allclose(np.diag(w), 0)


def test_binary_network_recovers_a_damaged_pattern():
    rng = np.random.default_rng(2)
    patterns = rng.choice([-1.0, 1.0], size=(5, 100))
    net = BinaryHopfield("hebbian").fit(patterns)
    damaged = patterns[:1].copy()
    damaged[0, :15] *= -1
    recovered, _ = net.retrieve(damaged, np.ones_like(damaged, dtype=bool))
    assert np.array_equal(recovered, patterns[:1])


def test_binary_energy_never_rises():
    rng = np.random.default_rng(3)
    net = BinaryHopfield().fit(rng.choice([-1.0, 1.0], size=(10, 80)))
    start = rng.choice([-1.0, 1.0], size=(20, 80))
    _, trace = net.retrieve(start, np.ones_like(start, dtype=bool))
    assert np.all(np.diff(trace.energy) <= 1e-9)


def test_binary_retrieval_keeps_clamped_neurons():
    rng = np.random.default_rng(4)
    net = BinaryHopfield().fit(rng.choice([-1.0, 1.0], size=(10, 80)))
    start = rng.choice([-1.0, 1.0], size=(5, 80))
    free = rng.random(start.shape) < 0.5
    final, _ = net.retrieve(start, free)
    assert np.array_equal(final[~free], start[~free])


def test_thermometer_code_preserves_order_and_distance():
    z = np.linspace(-3, 3, 500)[:, None]
    encoder = ThermometerEncoder(levels=8).fit(z)
    bits = encoder.encode(np.array([[-2.0], [-1.9], [2.0]]))
    assert np.abs(bits[0] - bits[1]).sum() < np.abs(bits[0] - bits[2]).sum()
    decoded = encoder.decode(encoder.encode(z))
    assert np.abs(decoded - z).max() < 0.6


# ---- continuous network --------------------------------------------------- #
def test_stored_pattern_is_recalled_from_a_partial_cue():
    rng = np.random.default_rng(5)
    memories = rng.normal(size=(30, 8)) * 3
    net = ContinuousHopfield(beta=8.0).fit(memories)
    unknown = np.zeros((1, 8), dtype=bool)
    unknown[0, :3] = True
    completed, _ = net.reconstruct(memories[4:5], unknown)
    assert np.allclose(completed, memories[4:5], atol=1e-3)


@pytest.mark.parametrize("gamma", [0.05, 0.25, 1.0])
def test_continuous_energy_never_rises(correlated, gamma):
    net = ContinuousHopfield(beta=4.0, free_weight=gamma, tol=0).fit(correlated[:300])
    z = correlated[300:]
    mask = random_missing_mask(z, 0.4, np.random.default_rng(6))
    _, trace = net.reconstruct(z, mask)
    assert np.all(np.diff(trace.energy) <= 1e-9)


@pytest.mark.parametrize("context", [0, 1])
def test_known_readings_are_never_changed(correlated, context):
    net = ContinuousHopfield(beta=4.0, context=context).fit(correlated[:300])
    z = correlated[300:]
    mask = random_missing_mask(z, 0.3, np.random.default_rng(7))
    completed, _ = net.reconstruct(z, mask)
    assert np.array_equal(completed[~mask], z[~mask])
    assert not np.isnan(completed).any()


def test_model_cannot_see_hidden_values(correlated):
    """Changing a hidden value must not change the reconstruction."""
    net = ContinuousHopfield(beta=4.0, context=1).fit(correlated[:300])
    z = correlated[300:].copy()
    mask = random_missing_mask(z, 0.3, np.random.default_rng(8))
    first, _ = net.reconstruct(z, mask)
    z[mask] = 999.0
    second, _ = net.reconstruct(z, mask)
    assert np.allclose(first[mask], second[mask])


def test_memory_holds_only_complete_patterns(correlated):
    data = correlated[:100].copy()
    data[10, 2] = np.nan
    assert ContinuousHopfield(context=0).fit(data).memories.shape[0] == 99
    assert ContinuousHopfield(context=1).fit(data).memories.shape[0] == 98 - 3


def test_windows_line_up_in_time():
    z = np.arange(10.0).reshape(5, 2)
    w = build_windows(z, 1)
    assert np.array_equal(w[2], [2, 3, 4, 5, 6, 7])
    assert np.isnan(w[0, :2]).all() and np.isnan(w[-1, -2:]).all()


def test_binary_imputer_end_to_end(correlated):
    model = BinaryHopfieldImputer(levels=8, n_prototypes=16).fit(correlated[:300])
    z = correlated[300:]
    mask = random_missing_mask(z, 0.3, np.random.default_rng(9))
    completed, _ = model.reconstruct(z, mask)
    assert np.array_equal(completed[~mask], z[~mask])
    assert score(z, completed, mask)["RMSE"] < score(z, mean_impute(z, mask), mask)["RMSE"]


# ---- corruption and scoring ----------------------------------------------- #
def test_artificial_damage_only_touches_observed_cells():
    rng = np.random.default_rng(10)
    z = rng.normal(size=(300, 5))
    z[rng.random(z.shape) < 0.2] = np.nan
    for make_mask in (random_missing_mask, outage_missing_mask):
        mask = make_mask(z, 0.3, np.random.default_rng(11))
        assert not np.isnan(z[mask]).any()
        assert mask.sum() == pytest.approx(0.3 * (~np.isnan(z)).sum(), rel=0.1)
    _, corrupted = inject_noise(z, 0.2, np.random.default_rng(12))
    assert not np.isnan(z[corrupted]).any()


def test_masks_respect_selected_columns():
    z = np.random.default_rng(13).normal(size=(200, 4))
    mask = random_missing_mask(z, 0.5, np.random.default_rng(14), columns=[1])
    assert mask[:, [0, 2, 3]].sum() == 0 and mask[:, 1].sum() > 0


def test_score_matches_hand_calculation():
    truth = np.array([[1.0], [2.0], [3.0], [100.0]])
    guess = np.array([[1.0], [2.0], [5.0], [0.0]])
    mask = np.array([[True], [True], [True], [False]])
    result = score(truth, guess, mask)
    assert result["MAE"] == pytest.approx(2 / 3)
    assert result["MSE"] == pytest.approx(4 / 3)
    assert result["R2"] == pytest.approx(1 - 4 / 2)
    assert result["cells"] == 3


def test_score_ignores_natural_gaps():
    truth = np.array([[1.0], [np.nan]])
    assert score(truth, np.array([[1.0], [9.0]]), np.array([[True], [True]]))["cells"] == 1


def test_interpolation_uses_time_neighbours():
    z = np.array([[0.0], [9.0], [2.0]])
    filled = interpolate_impute(z, np.array([[False], [True], [False]]))
    assert filled[1, 0] == pytest.approx(1.0)


# ---- real dataset --------------------------------------------------------- #
def test_uci_file_loads_as_published():
    raw = load_uci_raw()
    table = prepare_uci(raw)
    assert table.shape == (9357, 13)
    assert (table == -200).sum().sum() == 0
    assert table["CO(GT)"].isna().sum() == 1683
    kept, dropped = select_features(table)
    assert "NMHC(GT)" in dropped and kept.shape[1] == 12


def test_split_is_chronological_and_scaler_uses_training_data_only():
    kept, _ = select_features(prepare_uci(load_uci_raw()))
    split = split_and_scale(kept)
    assert split.train.index.max() < split.val.index.min() < split.test.index.min()
    assert np.allclose(split.mean, split.train.mean().to_numpy())


# ---- uploads -------------------------------------------------------------- #
def test_upload_rejects_unusable_files():
    with pytest.raises(ValueError):
        load_uploaded_csv(b"")
    with pytest.raises(ValueError):
        load_uploaded_csv(b"name,city\nA,B\nC,D\n")
    with pytest.raises(ValueError):
        load_uploaded_csv(b"a,b\n1,2\n3,4\n")


def test_upload_reads_semicolon_decimal_comma_and_sentinel():
    rows = "".join(f"2024-01-{i // 24 + 1:02d} {i % 24:02d}:00;{i},5;-200\n" for i in range(150))
    table = load_uploaded_csv(("time;x;y\n" + rows).encode(), sentinel=-200)
    assert table["x"].iloc[0] == 0.5
    assert table["y"].isna().all()


def test_upload_keeps_every_row_when_date_and_time_are_separate():
    """The UCI file itself: a date-only column must not delete rows as duplicates."""
    table = load_uploaded_csv(UCI_CSV_PATH.read_bytes(), sentinel=-200)
    assert len(table) == 9357
    assert table.shape[1] == 13
    assert (table == -200).sum().sum() == 0
