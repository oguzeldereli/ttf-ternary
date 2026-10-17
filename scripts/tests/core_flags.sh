# For scripts/tests/gate.sh (CORE_FLAGS_FN): the reference's command line (the trainer's flags at tag
# pre-consolidation-2026-10-17) translated into the core's options by ttf.args.from_legacy (docs/OPTIONS_MAP.md).
core_flags() {
  python3 -c "import sys, shlex; from ttf.args import from_legacy; print(shlex.join(from_legacy(shlex.split(sys.argv[1]))))" "$2"
}
