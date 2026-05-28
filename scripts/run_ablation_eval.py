#!/usr/bin/env python3
"""Multi-policy / multi-task EVAL-ONLY ablations for the supplementary, so the tables
report AVERAGES across policies and tasks rather than a single BC-T Square cell.

Phases (all use the RELEASED adapters; no training):
  A. Channel masking      -> averaged tab:supp_channel_mask
  B. Coupled physics      -> averaged tab:rq2_coupled
  C. Non-factorisable rho -> averaged tab:rq2_nonfact

Robust: each job is an independent subprocess; failures are logged and skipped.
Outputs JSON to results/paper_v2/ablation_multi/. Detached-friendly (run in tmux).
"""
import os, sys, json, subprocess, time
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'results/paper_v2/ablation_multi'; OUT.mkdir(parents=True, exist_ok=True)
LOG = Path('/tmp/abl_eval_logs'); LOG.mkdir(exist_ok=True)
PY = 'python'
EVAL = str(ROOT / 'scripts/eval_transfer_any_base.py')
EPISODES = 20; SEED = 42
MAX_CONCURRENT = 24   # 2x H100 + 128 CPU cores; each mujoco eval uses ~1-2 GB GPU + ~2-3 cores

MANIFEST = {c['cell']: c for c in json.load(open('/tmp/n50_manifest.json'))}
# Broad cell set: 5 policies (BC, BC-T, BCQ, IRIS, HBC) x 4 tasks (Square, Can, Threading, Stack)
CELLS = ['bc_square', 'bct_square', 'bcq_square', 'iris_square', 'hbc_square',
         'bc_can', 'bc_threading', 'hbc_threading', 'bc_stack', 'bcq_stack']

# Channel-mask configs: value = channels MASKED to nominal (so the rest are visible)
CHMASK = {'full': '', 'Tonly': 'C,V', 'Conly': 'T,V', 'Vonly': 'T,C',
          'TC': 'V', 'TV': 'C', 'CV': 'T'}

def base_cmd(cell, out, extra):
    c = MANIFEST[cell]
    cmd = [PY, EVAL, '--ckpt', c['ckpt'], '--task', c['task'],
           '--episodes', str(EPISODES), '--horizon', str(c['horizon']),
           '--seed', str(SEED), '--alpha', '0.30', '--gamma-range', '0.75',
           '--out', str(out)]
    return cmd + extra

def build_jobs():
    jobs = []  # (name, cmd, out)
    # A. channel masking (released adapter, vary visible channels)
    for cell in CELLS:
        c = MANIFEST[cell]
        for cfg, masked in CHMASK.items():
            out = OUT / f'chmask_{cell}_{cfg}.json'
            extra = ['--adapter-ckpt', c['adapter_ckpt']]
            if masked: extra += ['--mask-channels', masked]
            jobs.append((f'chmask_{cell}_{cfg}', base_cmd(cell, out, extra), out))
    # B. coupled physics: base (no adapter) and TAM (released adapter)
    for cell in CELLS:
        c = MANIFEST[cell]
        jobs.append((f'coupled_{cell}_base', base_cmd(cell, OUT/f'coupled_{cell}_base.json',
                     ['--coupled-physics']), OUT/f'coupled_{cell}_base.json'))
        jobs.append((f'coupled_{cell}_tam', base_cmd(cell, OUT/f'coupled_{cell}_tam.json',
                     ['--coupled-physics', '--adapter-ckpt', c['adapter_ckpt']]), OUT/f'coupled_{cell}_tam.json'))
    # C. non-factorisable rho: base and TAM
    for cell in CELLS:
        c = MANIFEST[cell]
        jobs.append((f'nonfact_{cell}_base', base_cmd(cell, OUT/f'nonfact_{cell}_base.json',
                     ['--non-factorizable-rho']), OUT/f'nonfact_{cell}_base.json'))
        jobs.append((f'nonfact_{cell}_tam', base_cmd(cell, OUT/f'nonfact_{cell}_tam.json',
                     ['--non-factorizable-rho', '--adapter-ckpt', c['adapter_ckpt']]), OUT/f'nonfact_{cell}_tam.json'))
    return jobs

def done(out):
    if not out.exists(): return False
    try:
        d = json.load(open(out)); return 'results' in d and 'cool' in d['results']
    except Exception: return False

def main():
    jobs = build_jobs()
    pending = [j for j in jobs if not done(j[2])]
    print(f'[abl_eval] {len(jobs)} jobs, {len(pending)} pending', flush=True)
    running = []; queue = list(pending); gpu_rr = 0; t0 = time.time()
    while queue or running:
        # reap
        still = []
        for j in running:
            rc = j[3].poll()
            if rc is None: still.append(j)
            else:
                j[4].close()
                print(f'[done {time.strftime("%H:%M:%S")}] {j[0]} rc={rc} ({(time.time()-j[5])/60:.1f}m)', flush=True)
        running = still
        # launch
        while queue and len(running) < MAX_CONCURRENT:
            name, cmd, out = queue.pop(0)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_rr % 2); gpu_rr += 1
            env['MUJOCO_GL'] = 'egl'
            lf = open(LOG / f'{name}.log', 'w')
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append((name, cmd, out, p, lf, time.time()))
            print(f'[launch {time.strftime("%H:%M:%S")}] {name} gpu={env["CUDA_VISIBLE_DEVICES"]} (run={len(running)} q={len(queue)})', flush=True)
        if not running and not queue: break
        time.sleep(10)
    print(f'[abl_eval] DONE in {(time.time()-t0)/3600:.2f}h', flush=True)

if __name__ == '__main__':
    main()
