"""The command line (ttf/args.py): recipes, derived settings, and the translation of the old flags."""
import glob
import os
import shlex

import pytest

from ttf import args as A

OLD_TTF = ("--preset small --mode kernel --data d --val v --seq_len 512 --batch_size 4 --grad_accum 1 --warmup 5 "
           "--rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 5 --g_ref 3.0 --int8 --dw_mode dense "
           "--tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000 --eval_iters 2 --save_secs 99999 --snap_every 0 "
           "--no_final_save --lookahead 0 --seed 7 --ts --ts_nomom --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 "
           "--ts_tau_anneal --ts_fused --gemm int8mm --lowrank 64 --ts_rank_s 64 --rc_scale --lowrank_mag add:16 "
           "--mag_wd 0.1 --qk_temp --steps 60")


def test_defaults_are_the_recipe():
    a = A.parse([])
    assert (a.mode, a.ttf_rank, a.ttf_theta, a.ttf_tau, a.ttf_step_p) == ("ttf", 512, 16.0, 1000.0, 0.75)
    assert a.adapter_rank == 16 and a.adapter_wd == 0.1 and a.qk_temp and a.rc_scale
    assert a.factored_k == 256 and a.tail == "fp32" and a.dx == "int8"


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(A.RECIPES, "*.toml"))))
def test_recipes_parse(path):
    name = os.path.basename(path)[:-5]
    a = A.parse(["--recipe", name])
    assert a.recipe == name
    assert A.parse(["--recipe", name, "--ttf-theta", "3"]).ttf_theta == 3.0     # the command line wins


def test_recipe_unknown_key(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('ttf-thetaa = 3\n')
    with pytest.raises(SystemExit):
        A.parse(["--recipe", str(p)])


def test_recipe_value_outside_choices(tmp_path):
    p = tmp_path / "bad.toml"     # (a removed value: before 26 Oct a recipe's values skipped argparse's choices check)
    p.write_text('ttf-offload = "host"\n')
    with pytest.raises(SystemExit, match="one of none, async"):
        A.parse(["--recipe", str(p)])


def test_from_legacy_ttf():
    a = A.parse(A.from_legacy(shlex.split(OLD_TTF)))
    assert (a.mode, a.ttf_rank, a.ttf_theta, a.adapter_rank, a.adapter_wd) == ("ttf", 64, 16.0, 16, 0.1)
    assert a.factored_k == 0                                     # off in the old command
    assert a.accum == 1 and a.batch_size == 4 and a.seq_len == 512 and a.final_save is False
    assert a.tail == "fp32"


def test_from_legacy_fast_and_master():
    a = A.parse(A.from_legacy(shlex.split(OLD_TTF + " --ts_micro --grad_accum 2 --ts_factored 64 --ts_fire_fused "
                                                    "--fused_scales --lean_tail --lean_batched")))
    assert a.factored_k == 64 and a.accum == 2 and a.tail == "lean"
    m = A.parse(A.from_legacy(shlex.split(OLD_TTF + " --mode master")))
    assert m.mode == "bitnet"


def test_from_legacy_dx():
    """the old trainer's input gradient: --int8_dx one int8 slice; without it bf16, whose equal here is int8x2"""
    assert A.parse(A.from_legacy(shlex.split(OLD_TTF))).dx == "int8x2"
    assert A.parse(A.from_legacy(shlex.split(OLD_TTF + " --int8_dx"))).dx == "int8"


@pytest.mark.parametrize("extra", ["", " --ts_nomom_x 1", " --lr_spend 1", " --dw_fp32", " --hid_kd 0.1",
                                   " --ts_store_a int8", " --init_hf_method row", " --ts_tau_anneal_pow 0.5",
                                   " --extend_to 9000", " --master_dtype bf16", "-int8", "-ts_tau_anneal", "-tail_fp32", " --grad_accum 2"])
def test_from_legacy_rejects(extra):
    """the old trainer only: no flip rule, unknown flags, and the settings removed on 25 Oct 2026 (a leading "-": the
    flag left out, e.g. the bf16 forward; --grad_accum 2 without --ts_micro: one update from the summed gradients)"""
    if extra.startswith("-") and not extra.startswith(" "):
        cmd = OLD_TTF.replace(" -" + extra + " ", " ")
        assert cmd != OLD_TTF
    else:
        cmd = OLD_TTF.replace(" --ts ", " ") if not extra else OLD_TTF + extra
    with pytest.raises(ValueError):
        A.from_legacy(shlex.split(cmd))


def test_options_map_is_current():
    """docs/OPTIONS_MAP.md is generated from the table (scripts/docs/options_map.py)"""
    p = os.path.join(os.path.dirname(A.RECIPES), "docs", "OPTIONS_MAP.md")
    t = open(p).read()
    for f in A._LEGACY:
        assert f in t, f


def test_from_legacy_restored_options():
    a = A.parse(A.from_legacy(shlex.split(OLD_TTF + " --ts_offload --ts_offload_async --w_cache int8 --ts_store_a fp12s")))
    assert a.ttf_offload == "async" and a.ttf_store == "fp12s"
    assert A.parse(A.from_legacy(shlex.split(OLD_TTF + " --ts_offload"))).ttf_offload == "async"     # (host: now async)
