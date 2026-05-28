#!/usr/bin/env python3
"""Expand the trunk-architecture ablation to contact-rich tasks so
tab:supp_adapter_archs averages over Square/Threading/Stack (not just Can/Lift).

New cells (train + eval, 7 variants + released reference each):
  bct_square (NutAssemblySquare), bc_threading (Threading_D0), bc_stack (Stack_D0)
Variants eval via eval_arch_variant.py (now extended to threading/stack);
released reference via eval_transfer_any_base.py (main per-joint Transformer trunk).

Robust: each (cell,config) is one subprocess (train && eval); failures logged
and skipped. Detached-friendly (run in tmux). Concurrency 4 (training is GPU-heavy).
"""
import os, glob, json, subprocess, time
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'results/paper_v2/ablation_multi'; OUT.mkdir(parents=True, exist_ok=True)
ADAP = ROOT / 'multi_task_runs/ablation_multi_arch'; ADAP.mkdir(parents=True, exist_ok=True)
LOG = Path('/tmp/abl_arch_expand_logs'); LOG.mkdir(exist_ok=True)
PY = 'python'
TRAIN_ARCH = str(ROOT / 'thermal_adapters/train_sft_arch.py')
TRAIN_MAIN = str(ROOT / 'thermal_adapters/train_sft_rl_kl_bct.py')
EVAL_ARCH = str(ROOT / 'scripts/eval_arch_variant.py')
EVAL_MAIN = str(ROOT / 'scripts/eval_transfer_any_base.py')
SFT_STEPS = 12000; EPISODES = 20; SEED = 42; MAX_CONCURRENT = 4

def g(p):
    m = glob.glob(str(p)); return m[0] if m else None

ARCH_VARIANTS = ['bottleneck_lr32','crossattn_h128','film_h128','hybrid_h128',
                 'jointwise_h64','mlp_h256_b2','moe4_h64']

ARCH_CELLS = {
    'bct_square': dict(train_task='NutAssemblySquare', eval_task='square', horizon=400,
        base=g(ROOT/'multi_task_runs/square/bct_square_image/*/models/model_epoch_300.pth'),
        demo='datasets/square/ph/image_v141.hdf5'),
    'bc_threading': dict(train_task='Threading_D0', eval_task='threading', horizon=500,
        base=g(ROOT/'external_policies/robosuite_policies/threading_d0_bc/*/models/model_epoch_950.pth'),
        demo='datasets/threading/threading_proper.hdf5'),
    'bc_stack': dict(train_task='Stack_D0', eval_task='stack', horizon=250,
        base=g(ROOT/'external_policies/robosuite_policies/stack_d0_bc/*/models/model_epoch_950.pth'),
        demo='datasets/stack/stack_d0.hdf5'),
}

def done(out):
    if not Path(out).exists(): return False
    try: return 'results' in json.load(open(out))
    except Exception: return False

def build_jobs():
    jobs = []
    for cell, c in ARCH_CELLS.items():
        if not c['base'] or not os.path.exists(c['demo']):
            print(f'[skip arch] {cell}: base/demo missing (base={c["base"]})', flush=True); continue
        for v in ARCH_VARIANTS:
            adir = ADAP / f'{cell}_{v}'; out = OUT / f'arch_{cell}_{v}.json'
            if done(out): continue
            train = (f'{PY} {TRAIN_ARCH} --task-name {c["train_task"]} --base-ckpt "{c["base"]}" '
                     f'--demo-hdf5 "{c["demo"]}" --variant {v} --sft-steps {SFT_STEPS} --seed {SEED} '
                     f'--output-dir "{adir}"')
            evalc = (f'{PY} {EVAL_ARCH} --ckpt "{c["base"]}" --task {c["eval_task"]} '
                     f'--adapter-ckpt "{adir}/sft_adapter.pt" --variant {v} '
                     f'--episodes {EPISODES} --horizon {c["horizon"]} --out "{out}"')
            jobs.append((f'arch_{cell}_{v}', f'set -e; {train} && {evalc}'))
        # released per-joint Transformer reference (main trainer + eval)
        adir = ADAP / f'{cell}_released'; out = OUT / f'arch_{cell}_released.json'
        if not done(out):
            train = (f'{PY} {TRAIN_MAIN} --task-name {c["train_task"]} --base-ckpt "{c["base"]}" '
                     f'--demo-hdf5 "{c["demo"]}" --adapter-class tambot --hidden 128 --tam-n-layers 3 '
                     f'--tam-n-heads 4 --alpha 0.3 --tam-gamma-range 0.75 --gate-shape smoothstep '
                     f'--sft-steps {SFT_STEPS} --sft-lr 5e-4 --batch-size 256 --skip-rl '
                     f'--output-dir "{adir}" --seed {SEED} --horizon {c["horizon"]}')
            evalc = (f'{PY} {EVAL_MAIN} --ckpt "{c["base"]}" --task {c["eval_task"]} '
                     f'--adapter-ckpt "{adir}/sft_adapter.pt" --episodes {EPISODES} '
                     f'--horizon {c["horizon"]} --seed {SEED} --alpha 0.30 --gamma-range 0.75 --out "{out}"')
            jobs.append((f'arch_{cell}_released', f'set -e; {train} && {evalc}'))
    return jobs

def main():
    jobs = build_jobs()
    print(f'[arch_expand] {len(jobs)} train+eval jobs', flush=True)
    running = []; queue = list(jobs); gpu = 0; t0 = time.time()
    while queue or running:
        still = []
        for name, p, lf, st in running:
            if p.poll() is None: still.append((name, p, lf, st))
            else:
                lf.close(); print(f'[done {time.strftime("%H:%M:%S")}] {name} rc={p.returncode} ({(time.time()-st)/60:.1f}m)', flush=True)
        running = still
        while queue and len(running) < MAX_CONCURRENT:
            name, wrapper = queue.pop(0)
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu % 2); gpu += 1
            env['MUJOCO_GL'] = 'egl'
            lf = open(LOG / f'{name}.log', 'w')
            p = subprocess.Popen(['bash','-c',wrapper], env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append((name, p, lf, time.time()))
            print(f'[launch {time.strftime("%H:%M:%S")}] {name} gpu={env["CUDA_VISIBLE_DEVICES"]} (run={len(running)} q={len(queue)})', flush=True)
        if not running and not queue: break
        time.sleep(20)
    print(f'[arch_expand] DONE in {(time.time()-t0)/3600:.2f}h', flush=True)

if __name__ == '__main__':
    main()
