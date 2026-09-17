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


def apply_config(
    parser: argparse.ArgumentParser, config: dict[str, Any], source: str,
) -> list[str]:
    """Make the file's values the parser's defaults; return the keys applied.

    Unknown keys refuse, naming what is accepted. A required flag or
    positional the file supplies stops being required, so
    ``pyuvimage fit --config params.json`` alone is a complete command.
    """
    table = _lookup(parser)
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
