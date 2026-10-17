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
    assert (a.mode, a.ttf_rank, a.ttf_theta, a.ttf_tau, a.ttf_step_p, a.ttf_leak) == ("ttf", 512, 16.0, 1000.0, 0.75,
                                                                                     "annealed")
    assert a.adapter_rank == 16 and a.adapter_wd == 0.1 and a.qk_temp and a.rc_scale
    assert a.factored_k == 256 and a.micro and a.tail == "fp32" and a.gemm == "int8mm" and a.fwd == "int8"


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


def test_from_legacy_ttf():
    a = A.parse(A.from_legacy(shlex.split(OLD_TTF)))
    assert (a.mode, a.ttf_rank, a.ttf_theta, a.adapter_rank, a.adapter_wd) == ("ttf", 64, 16.0, 16, 0.1)
    assert a.factored_k == 0 and not a.micro and a.fire == "torch" and not a.fused_scales   # off in the old command
    assert a.accum == 1 and a.batch_size == 4 and a.seq_len == 512 and a.final_save is False
    assert a.ttf_update == "fused" and a.ttf_leak == "annealed" and a.tail == "fp32"


def test_from_legacy_fast_and_master():
    a = A.parse(A.from_legacy(shlex.split(OLD_TTF + " --ts_micro --grad_accum 2 --ts_factored 64 --ts_fire_fused "
                                                    "--fused_scales --lean_tail --lean_batched")))
    assert a.factored_k == 64 and a.micro and a.fire == "fused" and a.fused_scales and a.tail == "lean"
    m = A.parse(A.from_legacy(shlex.split(OLD_TTF + " --mode master")))
    assert m.mode == "bitnet"


@pytest.mark.parametrize("extra", ["", " --ts_nomom_x 1", " --lr_spend 1"])
def test_from_legacy_rejects(extra):
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
    a = A.parse(A.from_legacy(shlex.split(OLD_TTF + " --ts_offload --ts_offload_async --w_cache int8 --dw_fp32")))
    assert a.ttf_offload == "async" and a.w_cache == "int8" and a.dw_precision == "fp32"
    assert A.parse(A.from_legacy(shlex.split(OLD_TTF + " --ts_offload_async"))).ttf_offload == "async"
