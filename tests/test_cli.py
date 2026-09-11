"""Arguments read from stdin behave like arguments typed on the command line.

`fit2082.cli` funnels every stdin shape back through the script's own parser
rather than into the namespace directly, so what is worth pinning is that the
detour changes nothing: types still convert, `choices` still reject, flags with
a non-obvious dest (`--no-cuda-async-pool` sets `cuda_async_pool`) still land
the right way round, and the command line still wins over the pipe.
"""

import argparse
import io
import json

import pytest

from fit2082.cli import parse_args


@pytest.fixture
def parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--dataset", default="LenDB")
    parser.add_argument("--model", default="hashboost", choices=("hashboost", "xgb"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--random-access", action="store_true")
    parser.add_argument(
        "--no-cuda-async-pool", dest="cuda_async_pool", action="store_false"
    )

    return parser


@pytest.fixture
def pipe(monkeypatch):
    """Put text on stdin, as a pipe rather than a terminal."""

    def feed(text: str) -> None:
        stream = io.StringIO(text)
        monkeypatch.setattr(stream, "isatty", lambda: False)
        monkeypatch.setattr("sys.stdin", stream)

    return feed


# == the two shapes ============================================================


def test_flags_on_stdin(parser, pipe):

    pipe("--dataset Traffic --epochs 3")

    args = parse_args(parser, ["--stdin"])

    assert (args.dataset, args.epochs) == ("Traffic", 3)


def test_flags_on_stdin_span_lines_and_carry_comments(parser, pipe):

    pipe("--dataset Traffic   # the big one\n--epochs 3\n\n--random-access\n")

    args = parse_args(parser, ["--stdin"])

    assert (args.dataset, args.epochs, args.random_access) == ("Traffic", 3, True)


def test_json_on_stdin(parser, pipe):

    pipe(json.dumps({"dataset": "Traffic", "epochs": 3, "lr": 0.5}))

    args = parse_args(parser, ["--stdin"])

    assert (args.dataset, args.epochs, args.lr) == ("Traffic", 3, 0.5)


def test_json_keys_may_be_written_with_dashes(parser, pipe):

    pipe(json.dumps({"random-access": True}))

    assert parse_args(parser, ["--stdin"]).random_access is True


def test_json_positionals(parser, pipe):

    pipe(json.dumps({"datasets": ["Traffic", "LenDB"], "epochs": 2}))

    args = parse_args(parser, ["--stdin"])

    assert (args.datasets, args.epochs) == (["Traffic", "LenDB"], 2)


def test_bare_names_on_stdin_are_positionals(parser, pipe):

    pipe("Traffic\nLenDB\n")

    assert parse_args(parser, ["--stdin"]).datasets == ["Traffic", "LenDB"]


def test_json_array_is_argv(parser, pipe):

    pipe('["--dataset", "Traffic", "LenDB"]')

    args = parse_args(parser, ["--stdin"])

    assert (args.dataset, args.datasets) == ("Traffic", ["LenDB"])


# == store_true / store_false ==================================================


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"random_access": True}, True),
        ({"random_access": False}, False),
        ({}, False),
    ],
)
def test_json_store_true(parser, pipe, values, expected):

    pipe(json.dumps(values))

    assert parse_args(parser, ["--stdin"]).random_access is expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"cuda_async_pool": False}, False),
        ({"cuda_async_pool": True}, True),
        ({}, True),
    ],
)
def test_json_store_false_uses_the_dest_not_the_flag(parser, pipe, values, expected):
    """`--no-cuda-async-pool` is the flag; `cuda_async_pool` is what it sets."""

    pipe(json.dumps(values))

    assert parse_args(parser, ["--stdin"]).cuda_async_pool is expected


# == precedence ================================================================


@pytest.mark.parametrize("shape", ["flags", "json"])
def test_command_line_beats_stdin(parser, pipe, shape):

    pipe(
        '{"dataset": "Traffic", "epochs": 3}'
        if shape == "json"
        else "--dataset Traffic --epochs 3"
    )

    args = parse_args(parser, ["--stdin", "--epochs", "9"])

    # the option that was typed is overridden, the rest of the pipe still stands
    assert (args.dataset, args.epochs) == ("Traffic", 9)


def test_stdin_is_not_read_without_the_flag(parser, pipe):

    pipe("--dataset Traffic")

    assert parse_args(parser, []).dataset == "LenDB"


# == rejections ================================================================


def test_stdin_still_type_checks(parser, pipe):
    """A config cannot smuggle in a value the flag would have rejected."""

    pipe(json.dumps({"epochs": "many"}))

    with pytest.raises(SystemExit):
        parse_args(parser, ["--stdin"])


def test_stdin_still_honours_choices(parser, pipe):

    pipe("--model lightgbm")

    with pytest.raises(SystemExit):
        parse_args(parser, ["--stdin"])


def test_unknown_json_key_is_rejected(parser, pipe, capsys):

    pipe(json.dumps({"datset": "Traffic"}))

    with pytest.raises(SystemExit):
        parse_args(parser, ["--stdin"])

    assert "dataset" in capsys.readouterr().err  # names the near miss


def test_malformed_json_is_rejected(parser, pipe):

    pipe('{"dataset": "Traffic",}')

    with pytest.raises(SystemExit):
        parse_args(parser, ["--stdin"])


def test_nested_json_array_is_rejected(parser, pipe):

    pipe('[["--dataset", "Traffic"]]')

    with pytest.raises(SystemExit):
        parse_args(parser, ["--stdin"])


def test_empty_stdin_leaves_the_defaults(parser, pipe):

    pipe("")

    assert parse_args(parser, ["--stdin"]).dataset == "LenDB"


def test_stdin_flag_on_a_terminal_is_rejected(parser, monkeypatch):

    stream = io.StringIO("")
    monkeypatch.setattr(stream, "isatty", lambda: True)
    monkeypatch.setattr("sys.stdin", stream)

    with pytest.raises(SystemExit):
        parse_args(parser, ["--stdin"])
