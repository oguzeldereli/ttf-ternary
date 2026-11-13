"""ttf.export: TTF models -> .ttf files for the ttf runtime (a GGUF container with the graph IR; see io).

  emit     the module library's IR emitters (one per module class), build_ir, the core-model export CLI
  ir       the IR's Python reference: every op in plain torch and a runner with explicit state and positions
  io       the .ttf container (GGUF): write_ttf / read_ttf
  refdump  per-part teacher-forced reference dumps (q, s, acc, t, out) for the runtime's exactness tests
"""
