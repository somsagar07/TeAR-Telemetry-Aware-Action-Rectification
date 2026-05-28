#!/usr/bin/env python3
"""Per-joint correction-magnitude analysis for the supplementary.
Runs the released TAM adapter with --log-corrections across 10 base x task cells
and the Table-1 condition set, recording the mean per-action-dim |a_TAM - a_base|
(dim j <-> joint j telemetry; dim 6 = gripper) at each condition.

This is a real-rollout analysis (no fabrication): corrections are logged from the
same MuJoCo rollouts used for success eval. Eval-only, no training.
Outputs: results/paper_v2/ablation_multi/corr_<cell>.json
Detached-friendly (run in tmux). Concurrency 10 (rollout-bound).
"""
import os, json, subprocess, time
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'results/paper_v2/ablation_multi'; OUT.mkdir(parents=True, exist_ok=True)
LOG = Path('/tmp/corr_analysis_logs'); LOG.mkdir(exist_ok=True)
PY = 'python'
EVAL = str(ROOT / 'scripts/eval_transfer_any_base.py')
EPISODES = 20; SEED = 42; MAX_CONCURRENT = 10
CONDITIONS = ['cool', 'hot', 'stall', 'brownout', 'T_mod', 'TC_mod', 'TV_mod', 'TCV_mod']

MANIFEST = {c['cell']: c for c in json.load(open('/tmp/n50_manifest.json'))}
CELLS = ['bc_square', 'bct_square', 'bcq_square', 'iris_square', 'hbc_square',
         'bc_can', 'bc_threading', 'hbc_threading', 'bc_stack', 'bcq_stack']

def done(out):
    if not Path(out).exists(): return False
    try:
        r = json.load(open(out)).get('results', {})
        return all('corr' in r.get(c, {}) for c in CONDITIONS)
    except Exception:
        return False

def build_jobs():
    jobs = []
    for cell in CELLS:
        c = MANIFEST[cell]
        out = OUT / f'corr_{cell}.json'
        if done(out): continue
        cmd = [PY, EVAL, '--ckpt', c['ckpt'], '--task', c['task'],
               '--adapter-ckpt', c['adapter_ckpt'],
               '--episodes', str(EPISODES), '--horizon', str(c['horizon']),
               '--seed', str(SEED), '--alpha', '0.30', '--gamma-range', '0.75',
               '--conditions', *CONDITIONS, '--log-corrections', '--out', str(out)]
        jobs.append((f'corr_{cell}', cmd, out))
    return jobs

def main():
    jobs = build_jobs()
    print(f'[corr_analysis] {len(jobs)} eval jobs', flush=True)
    running = []; queue = list(jobs); gpu = 0; t0 = time.time()
    while queue or running:
        still = []
        for name, p, lf, st in running:
            if p.poll() is None: still.append((name, p, lf, st))
            else:
                lf.close(); print(f'[done {time.strftime("%H:%M:%S")}] {name} rc={p.returncode} ({(time.time()-st)/60:.1f}m)', flush=True)
        running = still
        while queue and len(running) < MAX_CONCURRENT:
            name, cmd, out = queue.pop(0)
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu % 2); gpu += 1
            env['MUJOCO_GL'] = 'egl'
            lf = open(LOG / f'{name}.log', 'w')
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append((name, p, lf, time.time()))
            print(f'[launch {time.strftime("%H:%M:%S")}] {name} gpu={env["CUDA_VISIBLE_DEVICES"]} (run={len(running)} q={len(queue)})', flush=True)
        if not running and not queue: break
        time.sleep(10)
    print(f'[corr_analysis] DONE in {(time.time()-t0)/3600:.2f}h', flush=True)

if __name__ == '__main__':
    main()
