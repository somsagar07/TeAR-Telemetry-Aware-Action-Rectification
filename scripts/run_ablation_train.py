#!/usr/bin/env python3
"""Multi-cell SEVERITY-CURRICULUM ablation (train + eval), so tab:supp_severity
reports an AVERAGE over policies/tasks instead of a single BC-T Square cell.

For each cell x tier in {1,2,3,4}: train a TAM adapter with --num-sev-tiers tier
(SFT only), then evaluate it. Robust: each (cell,tier) runs as one subprocess
(train && eval); failures are logged and skipped. Detached-friendly (run in tmux).

Outputs: results/paper_v2/ablation_multi/sev_{cell}_t{tier}.json
Adapters: multi_task_runs/ablation_multi_sev/{cell}_t{tier}/sft_adapter.pt
"""
import os, sys, glob, json, subprocess, time
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'results/paper_v2/ablation_multi'; OUT.mkdir(parents=True, exist_ok=True)
ADAP = ROOT / 'multi_task_runs/ablation_multi_sev'; ADAP.mkdir(parents=True, exist_ok=True)
LOG = Path('/tmp/abl_train_logs'); LOG.mkdir(exist_ok=True)
PY = 'python'
TRAIN = str(ROOT / 'thermal_adapters/train_sft_rl_kl_bct.py')
EVAL = str(ROOT / 'scripts/eval_transfer_any_base.py')
SFT_STEPS = 12000; EPISODES = 20; SEED = 42
MAX_CONCURRENT = 4   # training is GPU-heavier; run alongside the eval pipeline

def g(p):  # first glob match
    m = glob.glob(str(p)); return m[0] if m else None

CELLS = {
    'bct_can': dict(train_task='PickPlaceCan', eval_task='can', horizon=400,
        base=g(ROOT/'multi_task_runs/can/bct_can_image/*/models/model_epoch_300.pth'),
        demo='datasets/can/ph/image_v141.hdf5'),
    'bc_threading': dict(train_task='Threading_D0', eval_task='threading', horizon=500,
        base=g(ROOT/'external_policies/robosuite_policies/threading_d0_bc/*/models/model_epoch_950.pth'),
        demo='datasets/threading/threading_proper.hdf5'),
    'bc_stack': dict(train_task='Stack_D0', eval_task='stack', horizon=250,
        base=g(ROOT/'external_policies/robosuite_policies/stack_d0_bc/*/models/model_epoch_950.pth'),
        demo='datasets/stack/stack_d0.hdf5'),
}
TIERS = [1, 2, 3, 4]

def build_jobs():
    jobs = []
    for cell, c in CELLS.items():
        if not c['base'] or not os.path.exists(c['demo']):
            print(f'[skip] {cell}: base/demo missing (base={c["base"]})', flush=True); continue
        for tier in TIERS:
            adir = ADAP / f'{cell}_t{tier}'
            adapter = adir / 'sft_adapter.pt'
            out = OUT / f'sev_{cell}_t{tier}.json'
            if out.exists():
                try:
                    if 'results' in json.load(open(out)): continue
                except Exception: pass
            train = (f'{PY} {TRAIN} --task-name {c["train_task"]} --base-ckpt "{c["base"]}" '
                     f'--demo-hdf5 "{c["demo"]}" --adapter-class tambot --hidden 128 '
                     f'--tam-n-layers 3 --tam-n-heads 4 --alpha 0.3 --tam-gamma-range 0.75 '
                     f'--gate-shape smoothstep --sft-steps {SFT_STEPS} --sft-lr 5e-4 --batch-size 256 '
                     f'--skip-rl --num-sev-tiers {tier} --output-dir "{adir}" --seed {SEED} --horizon {c["horizon"]}')
            evalc = (f'{PY} {EVAL} --ckpt "{c["base"]}" --task {c["eval_task"]} --adapter-ckpt "{adapter}" '
                     f'--episodes {EPISODES} --horizon {c["horizon"]} --seed {SEED} '
                     f'--alpha 0.30 --gamma-range 0.75 --out "{out}"')
            wrapper = f'set -e; {train} && {evalc}'
            jobs.append((f'sev_{cell}_t{tier}', wrapper))
    return jobs

def main():
    jobs = build_jobs()
    print(f'[abl_train] {len(jobs)} (cell,tier) train+eval jobs', flush=True)
    running = []; queue = list(jobs); gpu = 0; t0 = time.time()
    while queue or running:
        still = []
        for name, p, lf, st in running:
            rc = p.poll()
            if rc is None: still.append((name, p, lf, st))
            else:
                lf.close(); print(f'[done {time.strftime("%H:%M:%S")}] {name} rc={rc} ({(time.time()-st)/60:.1f}m)', flush=True)
        running = still
        while queue and len(running) < MAX_CONCURRENT:
            name, wrapper = queue.pop(0)
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu % 2); gpu += 1
            env['MUJOCO_GL'] = 'egl'
            lf = open(LOG / f'{name}.log', 'w')
            p = subprocess.Popen(['bash', '-c', wrapper], env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append((name, p, lf, time.time()))
            print(f'[launch {time.strftime("%H:%M:%S")}] {name} gpu={env["CUDA_VISIBLE_DEVICES"]} (run={len(running)} q={len(queue)})', flush=True)
        if not running and not queue: break
        time.sleep(20)
    print(f'[abl_train] DONE in {(time.time()-t0)/3600:.2f}h', flush=True)

if __name__ == '__main__':
    main()
