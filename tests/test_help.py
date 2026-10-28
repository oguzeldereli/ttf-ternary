"""python -m ttf.train --help formats every option's help (an unescaped % broke it, 26-28 Oct)"""
from ttf import args as A


def test_help_formats():
    assert "--ttf-rank" in A.build_parser().format_help()
