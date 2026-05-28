#!/usr/bin/env python3
"""Aggregate the multi-policy ablation JSONs into AVERAGED supplementary tables.
Reads results/paper_v2/ablation_multi/*.json (written by run_ablation_eval.py and
run_ablation_train.py) and prints, for each ablation, the mean +/- std across cells,
plus ready-to-paste LaTeX. Safe to run anytime; only averages what has completed.
"""
import json, glob, os
import numpy as np
from pathlib import Path

D = Path('./results/paper_v2/ablation_multi')
STRESS = ['hot','stall','brownout','T_mod','TC_mod','TV_mod','TCV_mod']
CELLS = ['bc_square','bct_square','bcq_square','iris_square','hbc_square',
         'bc_can','bc_threading','hbc_threading','bc_stack','bcq_stack']

def smean(fp):
    try:
        r = json.load(open(fp)).get('results', {})
        v = [r[c]['sr'] for c in STRESS if c in r and isinstance(r[c], dict) and 'sr' in r[c]]
        return 100*sum(v)/len(v) if v else None
    except Exception: return None

def avg(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None, 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals)>1 else 0.0), len(vals)

def section(title): print("\n"+"="*70+f"\n{title}\n"+"="*70)

# ---------- A. Channel masking ----------
section("CHANNEL MASKING  (avg over cells, stressed mean %)")
CFG = [('full','T+C+V'),('Tonly','T only'),('Conly','C only'),('Vonly','V only'),
       ('TC','T+C'),('TV','T+V'),('CV','C+V')]
full_m = None; chrows=[]
for key,label in CFG:
    vals=[smean(D/f'chmask_{c}_{key}.json') for c in CELLS]
    m,s,n=avg(vals)
    chrows.append((label,key,m,s,n))
    if key=='full': full_m=m
print(f"{'channels':<10} {'mean':>7} {'std':>6} {'n':>3}  {'Δ vs full':>9}")
for label,key,m,s,n in chrows:
    if m is None: print(f"{label:<10} {'--':>7}"); continue
    d = (m-full_m) if full_m is not None else 0
    print(f"{label:<10} {m:>6.1f}% {s:>5.1f} {n:>3}  {d:>+8.1f}")

# ---------- B. Coupled physics ----------
section("COUPLED PHYSICS  (avg over cells, stressed mean %)")
b=[smean(D/f'coupled_{c}_base.json') for c in CELLS]
t=[smean(D/f'coupled_{c}_tam.json') for c in CELLS]
bm,bs,bn=avg(b); tm,ts,tn=avg(t)
deltas=[(tt-bb) for bb,tt in zip(b,t) if bb is not None and tt is not None]
if bm is not None and tm is not None:
    dm,ds,dn=avg(deltas)
    print(f"frozen base : {bm:.1f}% ± {bs:.1f}  (n={bn})")
    print(f"TAM         : {tm:.1f}% ± {ts:.1f}  (n={tn})")
    print(f"Δ (TAM-base): {dm:+.1f} ± {ds:.1f}  (n={dn})")

# ---------- C. Non-factorisable rho ----------
section("NON-FACTORISABLE rho  (avg over cells, stressed mean %)")
b=[smean(D/f'nonfact_{c}_base.json') for c in CELLS]
t=[smean(D/f'nonfact_{c}_tam.json') for c in CELLS]
bm,bs,bn=avg(b); tm,ts,tn=avg(t)
deltas=[(tt-bb) for bb,tt in zip(b,t) if bb is not None and tt is not None]
if bm is not None and tm is not None:
    dm,ds,dn=avg(deltas)
    print(f"frozen base : {bm:.1f}% ± {bs:.1f}  (n={bn})")
    print(f"TAM         : {tm:.1f}% ± {ts:.1f}  (n={tn})")
    print(f"Δ (TAM-base): {dm:+.1f} ± {ds:.1f}  (n={dn})")

# ---------- D. Severity curriculum ----------
section("SEVERITY CURRICULUM  (avg over cells, stressed mean %)")
SEV_CELLS=['bct_can','bc_threading','bc_stack']
full4=None; sevrows=[]
for tier in [1,2,3,4]:
    vals=[smean(D/f'sev_{c}_t{tier}.json') for c in SEV_CELLS]
    m,s,n=avg(vals); sevrows.append((tier,m,s,n))
    if tier==4: full4=m
print(f"{'tiers':<6} {'mean':>7} {'std':>6} {'n':>3}  {'Δ vs 4':>7}")
for tier,m,s,n in sevrows:
    if m is None: print(f"{tier:<6} {'--':>7}"); continue
    d=(m-full4) if full4 is not None else 0
    print(f"{tier:<6} {m:>6.1f}% {s:>5.1f} {n:>3}  {d:>+6.1f}")

# ---------- E. Trunk architecture sweep ----------
section("TRUNK ARCHITECTURE  (avg over cells, stressed mean %)")
ARCH_CELLS = ['bct_can', 'bc_can', 'bc_lift', 'bct_square', 'bc_threading', 'bc_stack']
ARCH_VARIANTS = ['released','bottleneck_lr32','crossattn_h128','film_h128',
                 'hybrid_h128','jointwise_h64','mlp_h256_b2','moe4_h64']
rel_m = None; archrows = []
for v in ARCH_VARIANTS:
    vals = [smean(D/f'arch_{c}_{v}.json') for c in ARCH_CELLS]
    m,s,n = avg(vals); archrows.append((v,m,s,n))
    if v == 'released': rel_m = m
print(f"{'trunk':<16} {'mean':>7} {'std':>6} {'n':>3}  {'Δ vs released':>13}")
for v,m,s,n in archrows:
    if m is None: print(f"{v:<16} {'--':>7}"); continue
    d = (m-rel_m) if rel_m is not None else 0
    print(f"{v:<16} {m:>6.1f}% {s:>5.1f} {n:>3}  {d:>+12.1f}")

# ---------- F. Trunk capacity (released vs XL) ----------
section("TRUNK CAPACITY  (avg over cells, stressed mean %)")
CAP_CELLS = ['bct_can','bc_threading','bc_stack']
for tag in ['released','xl']:
    vals=[smean(D/f'cap_{c}_{tag}.json') for c in CAP_CELLS]
    m,s,n=avg(vals)
    print(f"{tag:<10} {m:>6.1f}% ± {s:>4.1f}  (n={n})" if m is not None else f"{tag:<10} --")

print("\n[done] Re-run after more jobs finish to refresh the averages.")
