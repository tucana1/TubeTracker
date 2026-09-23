"""Task scheduling: queue order, calibration spread, mixture caps."""

from tubetracker.annotation_tasks import (
    calibration_batch,
    mixture_batch,
    next_unfinished,
    progress_by_stratum,
)


def _t(uid, owner="o1", stratum="clean", completed=False):
    return {"uuid": uid, "owner_uuid": owner, "stratum": stratum,
            "completed": completed}


def test_next_unfinished_stable_and_none():
    ts = [_t("b"), _t("a", completed=True), _t("c")]
    assert next_unfinished(ts)["uuid"] == "b"
    assert next_unfinished([_t("a", completed=True)]) is None


def test_calibration_spreads_strata():
    ts = [_t(f"c{i}", stratum="crossing") for i in range(6)]
    ts += [_t(f"f{i}", stratum="faint") for i in range(6)]
    ts += [_t(f"x{i}", stratum="clean") for i in range(6)]
    b = calibration_batch(ts, n=12, seed=1)
    assert len(b) == 12
    strata = {t["stratum"] for t in b}
    assert strata == {"crossing", "faint", "clean"}


def test_mixture_caps_and_progress():
    ts = [_t(f"h{i}", owner="hot", stratum="crossing") for i in range(10)]
    ts += [_t(f"a{i}", owner=f"o{i}", stratum="clean") for i in range(10)]
    scores = {f"h{i}": 10.0 - i for i in range(10)}
    b = mixture_batch(ts, scores, n=10, per_owner_cap=2, seed=1)
    assert len(b) == 10
    from collections import Counter
    assert max(Counter(t["owner_uuid"] for t in b).values()) <= 2
    prog = progress_by_stratum(ts)
    assert prog["crossing"] == (0, 10)
