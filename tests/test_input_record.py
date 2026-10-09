"""Every run records its inputs; and a point at (0, 0) is a point.

`input_parameters.json` is written into the output directory at the start of
a run. From the CLI it is in ``--config`` form, so ``pyuvimage fit --config
<out>/input_parameters.json`` repeats the run; from a direct `run()` call it
holds that call's keyword arguments. `fit_parameters.json` stays what it was:
what the run *resolved* those inputs to.
"""
import json
import logging

import pytest

from pyuvimage import api, cli, mock
from pyuvimage.api import normalise_point_sources
from pyuvimage.config import INPUT_RECORD


@pytest.fixture
def captured(monkeypatch):
    seen = []
    monkeypatch.setattr(api, "run", lambda *a, **k: seen.append((a, k)))
    return seen


# --- the CLI record ----------------------------------------------------------

def test_the_cli_hands_run_a_config_form_record(captured):
    cli.main(["fit", "d.npz", "--fov", "5", "--reg", "gibbs",
              "--point=0,0", "--image-centre=-2,1", "--no-streaming"])
    rec = captured[-1][1]["input_parameters"]
    assert rec["fov"] == 5.0 and rec["reg"] == "gibbs"
    assert rec["point"] == [[0.0, 0.0]]
    assert rec["image-centre"] == [-2.0, 1.0]
    assert rec["streaming"] is False and "no-streaming" not in rec
    assert rec["dataset"].endswith("d.npz") and rec["dataset"].startswith("/")
    assert "config" not in rec


def test_the_record_repeats_the_run(tmp_path, captured):
    """Flags -> record -> --config must give the same `run` call."""
    argv = ["fit", str(tmp_path / "d.npz"), "--fov", "5", "--reg", "gibbs",
            "--point=0,0", "--point=-1.5,0.25", "--image-centre=-2,1",
            "--lambda", "3e4", "--no-streaming", "--uncertainty", "statistical",
            "--mesh", "40", "--chunk-k", "2048"]
    cli.main(argv)
    a1, k1 = captured[-1]
    path = tmp_path / "rec.json"
    path.write_text(json.dumps(k1["input_parameters"]))

    cli.main(["fit", "--config", str(path)])
    a2, k2 = captured[-1]
    assert a2 == a1
    assert {k: v for k, v in k2.items() if k != "input_parameters"} == \
        {k: v for k, v in k1.items() if k != "input_parameters"}


def test_the_record_reflects_the_config_file_and_the_flags(tmp_path, captured):
    """Flags win over the file before the record is made."""
    cfg = tmp_path / "params.json"
    cfg.write_text(json.dumps({"dataset": "d.npz", "fov": 5, "reg": "gibbs"}))
    cli.main(["fit", "--config", str(cfg), "--fov", "12"])
    rec = captured[-1][1]["input_parameters"]
    assert rec["fov"] == 12.0 and rec["reg"] == "gibbs"


# --- run() writes it -----------------------------------------------------------

def _stop_after_the_record(monkeypatch):
    def stop(*a, **k):
        raise RuntimeError("stop")
    monkeypatch.setattr(api, "resolve_streaming", stop)


def test_run_writes_the_record_before_it_starts(tmp_path, monkeypatch):
    """A run that dies still says what it was asked to do."""
    _stop_after_the_record(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        api.run("d.npz", fov=3.0, out=str(tmp_path), point_sources=(0, 0))
    rec = json.loads((tmp_path / INPUT_RECORD).read_text())
    assert rec["fov"] == 3.0
    assert rec["point_sources"] == [0, 0]
    assert "pyuvimage.run()" in rec["_comment"]


def test_a_cli_record_is_written_as_given(tmp_path, monkeypatch):
    _stop_after_the_record(monkeypatch)
    with pytest.raises(RuntimeError):
        api.run("d.npz", fov=3.0, out=str(tmp_path),
                input_parameters={"fov": 3.0, "point": [[0.0, 0.0]]})
    rec = json.loads((tmp_path / INPUT_RECORD).read_text())
    assert rec["point"] == [[0.0, 0.0]] and "--config" in rec["_comment"]
    assert "_pyuvimage_version" in rec and "_written" in rec


def test_no_record_without_write(tmp_path, monkeypatch):
    _stop_after_the_record(monkeypatch)
    with pytest.raises(RuntimeError):
        api.run("d.npz", fov=3.0, out=str(tmp_path / "out"), write=False)
    assert not (tmp_path / "out").exists()


def test_an_in_memory_dataset_is_named_not_serialised(tmp_path, monkeypatch):
    _stop_after_the_record(monkeypatch)
    uvd, _, _, _ = mock.make_demo_dataset(n_vis=40)
    with pytest.raises(RuntimeError):
        api.run(uvd, fov=3.0, out=str(tmp_path))
    rec = json.loads((tmp_path / INPUT_RECORD).read_text())
    assert rec["dataset"].startswith("<in-memory")


# --- point_sources -------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (None, False),
    (False, False),
    ([], False),
    (True, True),
    ("auto", True),
    ((0, 0), [(0.0, 0.0)]),            # the trap: read as "auto-detect"
    ([0, 0], [(0.0, 0.0)]),            # used to crash
    ("0,0", [(0.0, 0.0)]),
    ([(0, 0)], [(0.0, 0.0)]),
    ([[1, 2], (-0.5, 0.25)], [(1.0, 2.0), (-0.5, 0.25)]),
])
def test_point_sources_normalise(value, expected):
    assert normalise_point_sources(value) == expected


def test_a_malformed_position_is_refused():
    with pytest.raises(ValueError, match="not an \\(x, y\\) position"):
        normalise_point_sources([(1, 2, 3)])


def test_the_default_is_none():
    import inspect

    assert inspect.signature(api.run).parameters["point_sources"].default is None


def test_a_point_at_the_phase_centre_is_fitted_as_a_position(caplog):
    """`run(point_sources=(0, 0))` must fit a point *at* 0,0 -- not run
    auto-detection and drop the position, which it used to."""
    uvd, _, _, _ = mock.make_demo_dataset(point_flux_jy=0.004)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        api.run(uvd, fov=3.0, point_sources=(0, 0), uncertainty_map=False,
                write=False)
    assert "fitting analytic point sources (user positions)" in caplog.text
    assert "fitting analytic point sources (auto-detect" not in caplog.text
