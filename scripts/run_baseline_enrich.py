#!/usr/bin/env python3
"""Enrich tab:rq2_coupled and tab:rq2_nonfact with extra method ROWS, averaged over
the same 10 base x task cells as base/TAM. Adds, under both coupled and
non-factorisable physics:
   - Analytic feedforward (privileged closed-form inverse 1/rho_hat, gated)
   - MRAC / L1 online adaptive control (EMA gain estimate, gated)
All methods leave the base frozen and see only nominal telemetry; the env applies
the coupled / non-factorisable physics. Eval-only (no training).

Outputs: results/paper_v2/ablation_multi/{coupled,nonfact}_{cell}_{analytic,mrac}.json
Detached-friendly (run in tmux). Concurrency 12 (eval is rollout-bound).
"""
import os, json, subprocess, time
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'results/paper_v2/ablation_multi'; OUT.mkdir(parents=True, exist_ok=True)
LOG = Path('/tmp/abl_baseline_enrich_logs'); LOG.mkdir(exist_ok=True)
PY = 'python'
EVAL = str(ROOT / 'scripts/eval_transfer_any_base.py')
EPISODES = 20; SEED = 42; MAX_CONCURRENT = 12

MANIFEST = {c['cell']: c for c in json.load(open('/tmp/n50_manifest.json'))}
CELLS = ['bc_square', 'bct_square', 'bcq_square', 'iris_square', 'hbc_square',
         'bc_can', 'bc_threading', 'hbc_threading', 'bc_stack', 'bcq_stack']

# physics flag per regime, method flag per baseline
PHYSICS = {'coupled': ['--coupled-physics'], 'nonfact': ['--non-factorizable-rho']}
METHODS = {'analytic': ['--analytic-feedforward'], 'mrac': ['--mrac-baseline']}

def base_cmd(cell, out, extra):
    c = MANIFEST[cell]
    cmd = [PY, EVAL, '--ckpt', c['ckpt'], '--task', c['task'],
           '--episodes', str(EPISODES), '--horizon', str(c['horizon']),
           '--seed', str(SEED), '--alpha', '0.30', '--gamma-range', '0.75',
           '--out', str(out)]
    return cmd + extra

def done(out):
    if not Path(out).exists(): return False
    try: return 'results' in json.load(open(out))
    except Exception: return False

def build_jobs():
    jobs = []
    for cell in CELLS:
        for phys, pflag in PHYSICS.items():
            for meth, mflag in METHODS.items():
                out = OUT / f'{phys}_{cell}_{meth}.json'
                if done(out): continue
                name = f'{phys}_{cell}_{meth}'
                jobs.append((name, base_cmd(cell, out, pflag + mflag), out))
    return jobs

def main():
    jobs = build_jobs()
    print(f'[baseline_enrich] {len(jobs)} eval jobs', flush=True)
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
    print(f'[baseline_enrich] DONE in {(time.time()-t0)/3600:.2f}h', flush=True)

if __name__ == '__main__':
    main()
