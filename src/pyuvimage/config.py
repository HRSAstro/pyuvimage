"""Parameters from a JSON file, as an alternative to command-line flags.

``pyuvimage fit --config params.json`` reads every ``fit`` option from the
file; anything also given on the command line wins. Keys are the flag names
without the leading dashes (``"pixel-scale"`` or ``"pixel_scale"``), or the
destination the flag sets (``"lambda"`` and ``"coefficient"`` are the same
parameter). Values are what the flag would take: strings and numbers as on
the command line, ``true``/``false`` for switches, ``[x, y]`` or ``"x,y"`` for
positions, a list of positions for ``"point"``. ``docs/fit-config-template.json``
lists every key with its default.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: the flag this module hangs off; scanned for before argparse runs
CONFIG_FLAG = "--config"

#: "this key changes nothing" -- a `"no-x": false` that must not overwrite the
#: default living on the paired flag's action
UNSET = object()


def config_path_from(argv: list[str]) -> str | None:
    """The value of ``--config`` in ``argv``, or ``None``.

    Found before parsing so the file's values can become the parser's
    defaults -- which is what lets an explicit flag override them, and lets a
    file supply the otherwise-required ``dataset`` and ``--fov``.
    """
    for i, item in enumerate(argv):
        if item == CONFIG_FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{CONFIG_FLAG} needs a file")
            return argv[i + 1]
        if item.startswith(CONFIG_FLAG + "="):
            return item[len(CONFIG_FLAG) + 1:]
    return None


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the JSON file; a top-level object is required."""
    try:
        with open(path) as fh:
            config = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"{CONFIG_FLAG}: no such file {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{CONFIG_FLAG}: {path} is not valid JSON ({exc})")
    if not isinstance(config, dict):
        raise SystemExit(f"{CONFIG_FLAG}: {path} must contain a JSON object")
    return config


def _normalise(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def _lookup(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Every accepted key -> its action: flag names and destinations alike."""
    table: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if action.dest == "config":
            continue
        for option in action.option_strings:
            if option.startswith("--"):
                table[_normalise(option)] = action
        if not action.option_strings:  # positional
            table[_normalise(action.dest)] = action
        table.setdefault(_normalise(action.dest), action)
    return table


def _pair_text(value: Any) -> Any:
    """``[x, y]`` -> ``"x,y"`` so the flag's own parser reads it."""
    if (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in value)
    ):
        return f"{value[0]},{value[1]}"
    return value


def _convert(action: argparse.Action, key: str, value: Any) -> Any:
    """Coerce a JSON value to what the flag would have produced.

    A switch's value says whether the flag is given, so ``"no-pb": true`` is
    ``--no-pb``. Two flags that share a destination (``--streaming`` /
    ``--no-streaming``) are the one exception: under the destination's own
    name the value is taken literally, so ``"streaming": false`` means what it
    looks like rather than "the --streaming flag was absent".
    """
    if isinstance(action, argparse._StoreTrueAction):
        if not isinstance(value, bool):
            raise SystemExit(
                f"{CONFIG_FLAG}: {key!r} is a switch; give true or false"
            )
        return value
    if isinstance(action, argparse._StoreConstAction):
        if _normalise(key) != action.dest:  # an alias, e.g. "no-streaming"
            if not isinstance(value, bool):
                raise SystemExit(
                    f"{CONFIG_FLAG}: {key!r} is a switch; give true or false"
                )
            # false is "the flag was not given" -- it must not overwrite the
            # destination's own default, which lives on the other action
            return action.const if value else UNSET
        if isinstance(value, bool) or value == action.default:
            return value
        raise SystemExit(
            f"{CONFIG_FLAG}: {key!r} takes true, false or {action.default!r}"
        )
    if isinstance(action, argparse._AppendAction):
        if value is None:
            return None
        # a bare [x, y] is one position, not two; [[x, y], ...] is a list
        if not isinstance(value, list) or _pair_text(value) is not value:
            value = [value]
        return [_pair_text(v) for v in value]
    if value is None:
        return None
    value = _pair_text(value)
    if action.type is not None and isinstance(value, str):
        try:
            value = action.type(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{CONFIG_FLAG}: {key!r}: {exc}")
    elif action.type is float and isinstance(value, (int, float)):
        value = float(value)
    elif action.type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"{CONFIG_FLAG}: {key!r} must be an integer")
    elif action.type is None and not isinstance(value, str):
        if isinstance(value, bool):
            raise SystemExit(f"{CONFIG_FLAG}: {key!r} takes a value, not a switch")
        value = str(value)  # "fov": 5 -> what `--fov 5` would have parsed
    if action.choices is not None and value not in action.choices:
        raise SystemExit(
            f"{CONFIG_FLAG}: {key!r} must be one of "
            f"{', '.join(map(str, action.choices))}, not {value!r}"
        )
    return value


def _legacy_keys(config: dict[str, Any], table) -> dict[str, Any]:
    """Translate keys older files carry into the options that replaced them.

    ``no-uncertainty`` (a boolean, until Oct 2026) became ``uncertainty``
    with a mode: false meant the map was made, which is what "systematic"
    produces, and true meant none. Files written by earlier runs
    (input_parameters.json) then repeat exactly.
    """
    out = dict(config)
    for old in ("no-uncertainty", "no_uncertainty"):
        if old in out and _normalise("uncertainty") in table and "uncertainty" not in out:
            out["uncertainty"] = "none" if bool(out.pop(old)) else "systematic"
    return out


def apply_config(
    parser: argparse.ArgumentParser, config: dict[str, Any], source: str,
) -> list[str]:
    """Make the file's values the parser's defaults; return the keys applied.

    Unknown keys refuse, naming what is accepted. A required flag or
    positional the file supplies stops being required, so
    ``pyuvimage fit --config params.json`` alone is a complete command.
    """
    table = _lookup(parser)
    config = _legacy_keys(config, table)
    values: dict[str, Any] = {}
    unknown = []
    for key, raw in config.items():
        if key.startswith("_"):  # "_comment" and friends
            continue
        action = table.get(_normalise(key))
        if action is None:
            unknown.append(key)
            continue
        converted = _convert(action, key, raw)
        if converted is UNSET:
            continue
        if action.dest in values:
            raise SystemExit(
                f"{CONFIG_FLAG}: {source} sets {action.dest!r} twice "
                f"(the last key seen was {key!r})"
            )
        values[action.dest] = converted
        for other in parser._actions:  # --fov given here need not be a flag
            if other.dest == action.dest and other.required:
                other.required = False
    if unknown:
        accepted = sorted({
            option.lstrip("-")
            for action in parser._actions
            if not isinstance(action, argparse._HelpAction)
            and action.dest != "config"
            for option in (action.option_strings or [action.dest])
            if option.startswith("--") or not action.option_strings
        })
        raise SystemExit(
            f"{CONFIG_FLAG}: unknown key(s) in {source}: "
            f"{', '.join(map(repr, unknown))}. Accepted: {', '.join(accepted)}"
        )
    parser.set_defaults(**values)
    return sorted(values)


# --------------------------------------------------------------------------
# The other direction: what a run was given, written so it can be given again.
# --------------------------------------------------------------------------

#: written into every output directory by `api.run`
INPUT_RECORD = "input_parameters.json"


def _pair_value(text: str) -> list[float]:
    x, y = (float(v) for v in str(text).replace("(", "").replace(")", "").split(","))
    return [x, y]


def config_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace,
) -> dict[str, Any]:
    """The inverse of `apply_config`: every option of ``parser`` as it was
    parsed, keyed the way a ``--config`` file is.

    So ``pyuvimage fit --config <out>/input_parameters.json`` repeats a run
    whether its options came from flags, a file, or both -- flags having
    already won over the file by the time ``args`` exists. Keys follow
    ``docs/fit-config-template.json`` (the flag without its dashes), in the
    parser's own order. The dataset is made absolute so the record still
    works from another directory.
    """
    out: dict[str, Any] = {}
    seen_dest: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or action.dest == "config":
            continue
        if action.dest in seen_dest:      # --no-streaming after --streaming
            continue
        seen_dest.add(action.dest)
        longs = [o for o in action.option_strings if o.startswith("--")]
        key = longs[0][2:] if longs else action.dest
        value = getattr(args, action.dest, None)
        if not action.option_strings and isinstance(value, str):
            value = str(Path(value).expanduser().resolve())   # the dataset
        elif isinstance(action, argparse._AppendAction) and value is not None:
            value = [_pair_value(v) for v in value]
        elif key == "image-centre" and isinstance(value, str) and "," in value:
            value = _pair_value(value)
        out[key] = value
    return out


def write_input_record(out_dir, record: dict[str, Any], *, cli: bool) -> Path:
    """Write the inputs of a run to ``out_dir/input_parameters.json``.

    Written at the *start* of a run, so a run that is killed or crashes still
    leaves a record of what it was asked to do. ``cli`` says whether the keys
    are the command line's (re-runnable with ``--config``) or `run()`'s own
    keyword arguments (a Python call).
    """
    import datetime

    from . import __version__

    head: dict[str, Any] = {
        "_written": datetime.datetime.now().isoformat(timespec="seconds"),
        "_pyuvimage_version": __version__,
    }
    if cli:
        head["_comment"] = (
            "The inputs of this run, in --config form. Re-run it with "
            "`pyuvimage fit --config input_parameters.json`; change any value "
            "here or override it on the command line. fit_parameters.json is "
            "what the run *resolved* these to."
        )
    else:
        head["_comment"] = (
            "The keyword arguments of this pyuvimage.run() call. These are "
            "run()'s names, not the command line's, so this file is a record "
            "rather than a --config input."
        )
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    path = path / INPUT_RECORD
    path.write_text(json.dumps({**head, **record}, indent=2, default=str))
    return path
