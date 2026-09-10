"""Letting the scripts take their arguments on stdin as well as the command line.

Everything under `scripts/` is configured by flags, which is comfortable to type
and awkward to generate. A sweep over eight datasets has to assemble a shell
command per run, a configuration worth keeping has nowhere to live but shell
history, and a run that took forty flags cannot be repeated without them.
`--stdin` lets the caller pipe the arguments in instead, in whichever shape is
nearer to hand:

    echo "--dataset LenDB --model xgboost" | uv run python scripts/stream_full.py --stdin

    uv run python scripts/stream_full.py --stdin <<'EOF'
    --dataset LenDB
    --model xgboost     # line breaks and comments are fine
    --epochs 5
    EOF

    jq -c '.runs[0]' sweep.json | uv run python scripts/stream_full.py --stdin

A JSON array is read as argv itself -- `["--dataset", "LenDB"]` -- which is the
shape another program building a run is most likely to already have.

Every shape is turned into argv tokens and handed to the same parser, so types,
choices, defaults and `--help` are exactly what they are on the command line and
there is no second definition of the interface to keep in step. What is typed on
the command line beats the same option read from stdin: the piped side is the
template, the typed side is the override.

    cat run.json | uv run python scripts/stream_full.py --stdin --dataset Traffic
"""

import argparse
import difflib
import json
import shlex
import sys
from typing import Any

# == stdin =====================================================================


def read_stdin(parser: argparse.ArgumentParser) -> str:
    """Slurp stdin, refusing the case where there is plainly nothing coming."""

    if sys.stdin.isatty():
        parser.error("--stdin was given but stdin is a terminal; pipe arguments in")

    return sys.stdin.read()


def json_tokens(values: dict[str, Any], parser: argparse.ArgumentParser) -> list[str]:
    """Turn `{"dataset": "LenDB", "epochs": 5}` into `--dataset LenDB --epochs 5`.

    Going back out to tokens rather than straight into the namespace is what
    keeps the two input paths honest: a JSON value passes through the same
    `type=` and `choices=` the flag has always had, so a config cannot smuggle in
    a string where the script expects an int and fail three hours later.
    """

    actions = {action.dest: action for action in parser._actions}

    positionals: list[str] = []
    optionals: list[str] = []

    for key, value in values.items():
        dest = key.replace("-", "_")
        action = actions.get(dest)

        if action is None or dest == "help":
            close = difflib.get_close_matches(dest, actions, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            parser.error(f"--stdin: unknown option {key!r}.{hint}")
            raise AssertionError  # parser.error exits; this is for the checker

        # a list is several values for one option, except for `nargs=0` flags
        items = value if isinstance(value, list) else [value]

        if not action.option_strings:
            positionals.extend(str(item) for item in items)
            continue

        # the long form, so a stdin error quotes something recognisable
        flag = max(action.option_strings, key=len)

        if action.nargs == 0:
            # store_true holds const True, store_false const False, and either
            # way the flag is only worth emitting when it changes the default
            if isinstance(action, argparse._CountAction):
                optionals.extend([flag] * int(value))
            elif bool(value) == bool(action.const):
                optionals.append(flag)
        elif isinstance(action, argparse._AppendAction):
            for item in items:
                optionals.extend([flag, str(item)])
        else:
            optionals.append(flag)
            optionals.extend(str(item) for item in items)

    # positionals first: argparse will not take a positional that follows an
    # option it does not belong to, and this side owns the whole ordering
    return positionals + optionals


def stdin_tokens(text: str, parser: argparse.ArgumentParser) -> list[str]:
    """Read any of the shapes -- JSON object, JSON array, or flags as typed."""

    if not text.strip():
        return []

    if text.lstrip()[0] not in "{[":
        # comments=True so a piped file can say why it sets what it sets
        return shlex.split(text, comments=True)

    try:
        values = json.loads(text)
    except json.JSONDecodeError as error:
        parser.error(f"--stdin: not valid JSON: {error}")

    if isinstance(values, dict):
        return json_tokens(values, parser)

    if any(isinstance(item, (dict, list)) for item in values):
        parser.error("--stdin: a JSON array is argv, so it holds only scalars")

    return [str(item) for item in values]


# == parsing ===================================================================


def parse_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    """`parser.parse_args()`, plus a `--stdin` that reads arguments from a pipe.

    Drop-in: a script swaps `parser.parse_args()` for `parse_args(parser)` and
    gains the flag, its help entry and its precedence rule at once.
    """

    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read arguments from stdin as well -- flags written as they would "
        "be typed, a JSON object of option names, or a JSON array of argv. "
        "Options given on the command line win over the same option from stdin.",
    )

    argv = list(sys.argv[1:] if argv is None else argv)

    # a first pass that knows only about --stdin, since whether to read the pipe
    # has to be settled before the real parse can be given its tokens
    sniff = argparse.ArgumentParser(add_help=False)
    sniff.add_argument("--stdin", action="store_true")
    piped, _ = sniff.parse_known_args(argv)

    if not piped.stdin:
        return parser.parse_args(argv)

    # stdin first: argparse lets a later occurrence of an option overwrite an
    # earlier one, which is precedence for free
    return parser.parse_args(stdin_tokens(read_stdin(parser), parser) + argv)
