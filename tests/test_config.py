"""`pyuvimage fit --config params.json` -- parameters from a file.

The contract, and what each test here pins: a key is the flag without its
dashes (or the destination it sets), the value is what the flag would take,
the command line still wins, and the file may supply the dataset and --fov so
that `pyuvimage fit --config params.json` is a complete command.
"""
import json
from pathlib import Path

import pytest

from pyuvimage import api, cli, config


REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "docs" / "fit-config-template.json"


@pytest.fixture
def captured(monkeypatch):
    """Every `run(...)` the CLI makes, without running a fit."""
    seen = []
    monkeypatch.setattr(api, "run", lambda *a, **k: seen.append((a, k)))
    return seen


def write(tmp_path, mapping) -> str:
    path = tmp_path / "params.json"
    path.write_text(json.dumps(mapping))
    return str(path)


# --- the basic substitution ------------------------------------------------

def test_a_file_can_replace_the_whole_command_line(tmp_path, captured):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 5.0, "reg": "gibbs"})
    cli.main(["fit", "--config", path])
    (dataset,), kwargs = captured[-1]
    assert dataset == "d.npz"
    assert kwargs["fov"] == 5.0
    assert kwargs["reg"] == "gibbs"


def test_the_command_line_wins(tmp_path, captured):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 5.0, "reg": "gibbs"})
    cli.main(["fit", "other.npz", "--config", path, "--fov", "12"])
    (dataset,), kwargs = captured[-1]
    assert dataset == "other.npz"
    assert kwargs["fov"] == 12.0
    assert kwargs["reg"] == "gibbs"  # not overridden, so the file's value


def test_the_equals_form_is_read_too(tmp_path, captured):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 5.0})
    cli.main(["fit", f"--config={path}"])
    assert captured[-1][0] == ("d.npz",)


def test_without_a_config_nothing_changes(captured):
    cli.main(["fit", "d.npz", "--fov", "3"])
    (dataset,), kwargs = captured[-1]
    assert dataset == "d.npz" and kwargs["fov"] == 3.0


# --- key spellings ---------------------------------------------------------

@pytest.mark.parametrize("key", ["pixel-scale", "pixel_scale", "--pixel-scale"])
def test_dashes_underscores_and_leading_dashes_all_work(
    tmp_path, captured, key
):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1, key: 0.05})
    cli.main(["fit", "--config", path])
    assert captured[-1][1]["pixel_scale"] == 0.05


@pytest.mark.parametrize("key", ["lambda", "coefficient"])
def test_a_flag_and_the_destination_it_sets_are_the_same_key(
    tmp_path, captured, key
):
    """--lambda sets `coefficient`; both names reach it."""
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1, key: 3.0})
    cli.main(["fit", "--config", path])
    assert captured[-1][1]["coefficient"] == 3.0


def test_underscore_keys_are_comments(tmp_path, captured):
    path = write(tmp_path, {
        "_comment": "notes for a human", "_why": {"nested": "is fine"},
        "dataset": "d.npz", "fov": 1,
    })
    cli.main(["fit", "--config", path])
    assert captured[-1][0] == ("d.npz",)


def test_one_parameter_cannot_be_set_twice(tmp_path):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1,
                            "lambda": 3.0, "coefficient": 4.0})
    with pytest.raises(SystemExit, match="twice"):
        cli.main(["fit", "--config", path])


# --- value conversion ------------------------------------------------------

def test_numbers_are_accepted_where_the_flag_takes_a_string(
    tmp_path, captured
):
    """`--pixel-scale` parses its own argument, so 0.05 must not arrive as a
    float the CLI then cannot `float()` -- and `--fov 5` must not arrive as
    the int 5."""
    path = write(tmp_path, {"dataset": "d.npz", "fov": 5, "pixel-scale": 0.05,
                            "mesh": 128, "nu": 2})
    cli.main(["fit", "--config", path])
    kwargs = captured[-1][1]
    assert kwargs["fov"] == 5.0 and isinstance(kwargs["fov"], float)
    assert kwargs["pixel_scale"] == 0.05
    assert kwargs["mesh_shape"] == (128, 128)
    assert kwargs["nu"] == 2.0


def test_a_position_may_be_a_pair_or_a_string(tmp_path, captured):
    """A negative x needs `--image-centre="-2,-2"` on the command line; in a
    file it is just a number."""
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1,
                            "image-centre": [-2.0, -3.0]})
    cli.main(["fit", "--config", path])
    assert captured[-1][1]["image_centre"] == (-2.0, -3.0)

    path = write(tmp_path, {"dataset": "d.npz", "fov": 1,
                            "image-centre": "-2,-3"})
    cli.main(["fit", "--config", path])
    assert captured[-1][1]["image_centre"] == (-2.0, -3.0)


def test_image_centre_auto_still_reaches_the_api(tmp_path, captured):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1,
                            "image-centre": "auto"})
    cli.main(["fit", "--config", path])
    assert captured[-1][1]["image_centre"] == "auto"


def test_points_are_a_list_of_positions(tmp_path, captured):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1,
                            "point": [[1.0, 2.0], "-0.5,0.25"]})
    cli.main(["fit", "--config", path])
    assert captured[-1][1]["point_sources"] == [(1.0, 2.0), (-0.5, 0.25)]


def test_a_single_point_need_not_be_wrapped_in_a_list(tmp_path, captured):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1, "point": [1.0, 2.0]})
    cli.main(["fit", "--config", path])
    assert captured[-1][1]["point_sources"] == [(1.0, 2.0)]


def test_switches_take_true_and_false(tmp_path, captured):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1,
                            "no-uncertainty": True, "no-pb": False,
                            "point-sources": True})
    cli.main(["fit", "--config", path])
    kwargs = captured[-1][1]
    assert kwargs["uncertainty_map"] is False
    assert kwargs["pb_correction"] is True
    assert kwargs["point_sources"] is True


def test_null_is_the_unset_value(tmp_path, captured):
    """`"adapt-power": null` must mean "not given", not "None"; the CLI drops
    the keyword entirely so `run`'s own default applies."""
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1,
                            "adapt-power": None, "mesh": None, "point": None})
    cli.main(["fit", "--config", path])
    kwargs = captured[-1][1]
    assert "adapt_power" not in kwargs
    assert kwargs["mesh_shape"] is None
    assert kwargs["point_sources"] is False


# --- the tri-state pair ----------------------------------------------------

@pytest.mark.parametrize(
    "mapping,expected",
    [
        ({"streaming": True}, True),
        ({"streaming": False}, False),
        ({"streaming": "auto"}, "auto"),
        ({"no-streaming": True}, False),
        ({"no-streaming": False}, "auto"),  # "the flag was not given"
        ({}, "auto"),
    ],
)
def test_streaming_reads_as_it_looks(tmp_path, captured, mapping, expected):
    """--streaming and --no-streaming share one destination. Under its own
    name the value is literal; `"no-streaming": false` must not clobber the
    default with the other action's `None`."""
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1, **mapping})
    cli.main(["fit", "--config", path])
    assert captured[-1][1]["streaming"] == expected


# --- refusals --------------------------------------------------------------

def test_an_unknown_key_refuses_and_says_what_is_accepted(tmp_path):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1, "regularisation": 2})
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["fit", "--config", path])
    message = str(excinfo.value)
    assert "regularisation" in message and "pixel-scale" in message


def test_a_value_outside_the_choices_refuses(tmp_path):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1, "mode": "spectral"})
    with pytest.raises(SystemExit, match="mfs"):
        cli.main(["fit", "--config", path])


def test_a_switch_given_a_value_refuses(tmp_path):
    path = write(tmp_path, {"dataset": "d.npz", "fov": 1, "no-pb": "yes"})
    with pytest.raises(SystemExit, match="switch"):
        cli.main(["fit", "--config", path])


def test_a_number_where_a_number_is_needed(tmp_path):
    path = write(tmp_path, {"dataset": "d.npz", "fov": "wide"})
    with pytest.raises(SystemExit):
        cli.main(["fit", "--config", path])


def test_a_missing_file_refuses(tmp_path):
    with pytest.raises(SystemExit, match="no such file"):
        cli.main(["fit", "--config", str(tmp_path / "absent.json")])


def test_broken_json_refuses(tmp_path):
    path = tmp_path / "params.json"
    path.write_text("{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        cli.main(["fit", "--config", str(path)])


def test_a_json_list_is_not_a_parameter_file(tmp_path):
    path = tmp_path / "params.json"
    path.write_text("[1, 2]")
    with pytest.raises(SystemExit, match="JSON object"):
        cli.main(["fit", "--config", str(path)])


def test_config_with_no_file_refuses(tmp_path):
    with pytest.raises(SystemExit, match="needs a file"):
        cli.main(["fit", "d.npz", "--fov", "1", "--config"])


def test_a_dataset_is_still_required(tmp_path):
    path = write(tmp_path, {"fov": 1})
    with pytest.raises(SystemExit):
        cli.main(["fit", "--config", path])


def test_fov_is_still_required(tmp_path):
    path = write(tmp_path, {"dataset": "d.npz"})
    with pytest.raises(SystemExit):
        cli.main(["fit", "--config", path])


# --- the template ----------------------------------------------------------

def test_the_template_parses_and_reproduces_every_default(captured):
    """docs/fit-config-template.json is the documented starting point, so it
    must be valid and must change nothing: running it must give the same
    `run(...)` call as the bare command line."""
    cli.main(["fit", "--config", str(TEMPLATE)])
    from_file = captured[-1][1]

    cli.main(["fit", "my_data.npz", "--fov", "5"])
    from_flags = captured[-1][1]

    assert captured[-2][0] == captured[-1][0] == ("my_data.npz",)
    assert from_file == from_flags


def test_every_template_key_is_a_real_flag():
    """A key the parser does not know would refuse at run time; this says so
    at test time, naming it."""
    mapping = json.loads(TEMPLATE.read_text())
    keys = {k for k in mapping if not k.startswith("_")}
    parser = _fit_parser()
    known = set(config._lookup(parser))
    unknown = {k for k in keys if config._normalise(k) not in known}
    assert not unknown, f"not flags: {sorted(unknown)}"


def _fit_parser():
    """The `fit` subparser `cli.main` builds."""
    import argparse

    holder = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        holder["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        cli.main(["fit", "d.npz", "--fov", "1"])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    subparsers = [
        a for a in holder["parser"]._actions
        if isinstance(a, argparse._SubParsersAction)
    ][0]
    return subparsers.choices["fit"]
