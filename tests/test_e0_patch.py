"""Batch-level, CPU-only unit tests for the E0 patches.

These tests do not need a GPU, a victim model, or network access. They check
the flag plumbing and the branch logic of HoneypotTeacher; they cannot tell us
anything about Agreement. Run with:

    python tests/test_e0_patch.py

They are deliberately runnable from the repo root with the stdlib only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeypot.honeypot_teacher import HoneypotTeacher


def fake_base_query(question, domain="medical", max_new_tokens=128, model="x"):
    # Deterministic, long-enough "real" answer so lengths are comparable to HKG.
    return f"REAL_ANSWER for: {question} " + ("details " * 30)


def make_teacher(**kw):
    return HoneypotTeacher(
        domain="medical",
        base_query=fake_base_query,
        hkg_size=50,
        hkg_path="honeypot_hkg/medical_hkg_1500_repaired.json",
        **kw,
    )


def test_default_matches_original():
    """Default flags (mode='full', legacy detector, no E0 flags) behave as shipped."""
    t = make_teacher(mode="full", detection_threshold=0.6)
    q = "Which of the following describes a long medical question about diagnosis"
    # ans_entity source makes suspicion >= 0.6 at step 0 -> substitution.
    r = t.query(q, model="x", max_new_tokens=64, source="ans_entity", step=0, budget=500)
    assert r.route == "hkg_breadcrumb", r.route
    assert r.hkg_node is not None
    assert len(r.breadcrumbs) == 2


def test_mask_source_delays_detection():
    q = "Which of the following describes a long medical question about diagnosis"
    t_masked = make_teacher(mode="full", detection_threshold=0.6, mask_source=True)
    r = t_masked.query(q, model="x", max_new_tokens=64, source="ans_entity", step=0, budget=500)
    assert r.route == "real", f"masked source should not trigger at step 0, got {r.route}"
    assert r.suspicion < 0.6, r.suspicion


def test_junk_control_no_breadcrumbs():
    q = "Which of the following describes a long medical question about diagnosis"
    t = make_teacher(mode="full", detection_threshold=0.6, junk_control=True)
    r = t.query(q, model="x", max_new_tokens=64, source="ans_entity", step=0, budget=500)
    assert r.route == "junk_breadcrumb", r.route
    assert r.breadcrumbs == [], r.breadcrumbs
    assert r.hkg_node is None
    assert "REAL_ANSWER" not in r.answer
    assert "extraction pipeline" not in r.answer


def test_oracle_detector_fires_step0_without_tag():
    q = "Which of the following describes a long medical question about diagnosis"
    t = make_teacher(mode="full", detector_mode="oracle", mask_source=True)
    r = t.query(q, model="x", max_new_tokens=64, source=None, step=0, budget=500)
    assert r.route == "hkg_breadcrumb", r.route


def test_lure_only_keeps_real_answer():
    q = "Which of the following describes a long medical question about diagnosis"
    t = make_teacher(mode="full", detection_threshold=0.6, lure_only=True)
    r = t.query(q, model="x", max_new_tokens=64, source="ans_entity", step=0, budget=500)
    assert r.route == "lure_breadcrumb", r.route
    assert "REAL_ANSWER" in r.answer
    assert len(r.breadcrumbs) == 2


def test_lure_always_without_detection():
    q = "Which of the following describes a long medical question about diagnosis"
    t = make_teacher(mode="full", detector_mode="oracle_none", detection_threshold=2.0, lure_only=True, lure_always=True)
    r = t.query(q, model="x", max_new_tokens=64, source=None, step=0, budget=500)
    assert r.route == "lure_breadcrumb", r.route
    assert "REAL_ANSWER" in r.answer


def test_summary_counts_junk():
    q = "Which of the following describes a long medical question about diagnosis"
    t = make_teacher(mode="full", detection_threshold=0.6, junk_control=True)
    t.query(q, model="x", max_new_tokens=64, source="ans_entity", step=0, budget=500)
    s = t.summary()
    assert s["junk"] == 1
    assert s["junk_fraction"] > 0


def test_log_records_source_masked(tmp_path=None):
    import tempfile
    q = "Which of the following describes a long medical question about diagnosis"
    log = os.path.join(tempfile.mkdtemp(), "log.jsonl")
    t = make_teacher(mode="full", detection_threshold=0.6, mask_source=True, log_path=log)
    t.query(q, model="x", max_new_tokens=64, source="ans_entity", step=0, budget=500)
    with open(log) as f:
        rec = json.loads(f.readline())
    assert rec["source"] == "ans_entity"          # true tag preserved for analysis
    assert rec["source_masked"] is True           # detector saw None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
