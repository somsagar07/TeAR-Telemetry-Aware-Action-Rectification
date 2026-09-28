"""Validate complete paired coverage and summarize fixed-checkpoint experiments.

Only complete final worker JSONs are consumed. --partial allows absent jobs,
never corrupt jobs; inference then uses complete curve blocks only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ['base','assumed_inverse','tam','tam_dr','oracle_capacity']
LABELS = dict(base='Frozen base',assumed_inverse='Assumed-model inverse',tam='TeAR',
              tam_dr='DR-TeAR',oracle_capacity='Oracle-capacity inverse')
HERE = Path(__file__).resolve().parent


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4],'little')


def fail(condition,message):
    if not condition:raise ValueError(message)


def load_results(manifest_path,results_dir,partial=False):
    manifest_path=Path(manifest_path);root=manifest_path.parent
    m=json.loads(manifest_path.read_text());mh=file_hash(manifest_path)
    fail(m['methods']==METHODS,'Unexpected manifest methods/order')
    for name,wanted in m['source_sha256'].items():
        fail(file_hash(root/name)==wanted,f'Source hash mismatch: {name}')
    code_root=root/m.get('code_root', '.')
    hashes={k:file_hash(code_root/f'{k}.py') for k in ('worker','protocol')}
    cells=list(m['cells']);curves=list(m['curves']);seeds=m['eval_seeds'];n=m['episodes_per_seed']
    expected={(cell,curve,seed,'stressed') for cell in cells for curve in curves for seed in seeds}
    expected|={(cell,curves[0],seed,'healthy') for cell in cells for seed in seeds}
    jobs={};flat=[]
    for path in sorted(Path(results_dir).glob('*.json')):
        d=json.loads(path.read_text())
        fail(d.get('manifest_sha256')==mh,f'{path.name}: manifest hash mismatch')
        for key,h in hashes.items():fail(d.get(f'{key}_sha256')==h,f'{path.name}: {key} hash mismatch')
        fail(d.get('pairing_verified') is True,f'{path.name}: pairing not verified')
        fail(d.get('episodes')==n,f'{path.name}: episode count differs from manifest')
        fail(set(d.get('methods',{}))==set(METHODS),f'{path.name}: missing/unexpected methods')
        conditions=set(d['methods']['base'])
        kind='healthy' if conditions=={'healthy'} else 'stressed'
        wanted={'healthy'} if kind=='healthy' else set(m['conditions'])
        fail(conditions==wanted,f'{path.name}: missing/unexpected conditions')
        key=(d['cell'],d['curve'],d['eval_seed'],kind)
        fail(key in expected,f'{path.name}: unexpected job {key}')
        fail(key not in jobs,f'Duplicate job {key}')
        for method in METHODS:
            fail(set(d['methods'][method])==wanted,f'{path.name}: {method} missing/unexpected conditions')
            for cond in sorted(wanted):
                block=d['methods'][method][cond];rows=block['rows']
                fail(len(rows)==n,f'{path.name}: {method} {cond} episode count mismatch')
                fail([r['episode'] for r in rows]==list(range(n)),f'{path.name}: episode IDs/order mismatch')
                fail(all(r['success'] in (0,1) for r in rows),f'{path.name}: nonbinary success')
                fail(abs(block['sr']-np.mean([r['success'] for r in rows]))<1e-10,f'{path.name}: SR inconsistent with episodes')
                ref=d['methods']['base'][cond]['rows']
                for r,b in zip(rows,ref):
                    fields=('steps','mean_correction','max_correction','saturation_fraction','mean_actual_capacity')
                    fail(all(f in r for f in fields),f'{path.name}: missing diagnostic field/denominator')
                    fail(isinstance(r['steps'],int) and r['steps']>0,f'{path.name}: invalid diagnostic step denominator')
                    for field,upper in [('mean_correction',2.),('max_correction',2.),('saturation_fraction',1.),('mean_actual_capacity',1.)]:
                        fail(np.isfinite(r[field]) and 0<=r[field]<=upper,f'{path.name}: invalid diagnostic {field}')
                    task=m['cells'][d['cell']].get('task',d['cell'])
                    expected_seed=stable_seed(task,d['curve'],cond,d['eval_seed'],r['episode'])
                    fail(r['seed']==expected_seed,f'{path.name}: episode seed mismatch')
                    for field in ('episode','seed','initial_hash','observation_hash','telemetry_hash'):
                        fail(r[field]==b[field],f'Pairing mismatch {path.name} {method} {cond}: {field}')
                    if kind=='healthy':
                        fail(r['trajectory_hash']==b['trajectory_hash'] and r['success']==b['success'] and r['steps']==b['steps'],
                             f'Healthy trajectory/outcome mismatch: {path.name} {method}')
                        fail(r['max_correction']==0 and r.get('max_gate',0)==0,f'Healthy correction/gate nonzero: {path.name} {method}')
                    flat.append(dict(cell=d['cell'],curve=d['curve'],eval_seed=d['eval_seed'],condition=cond,method=method,**r))
        jobs[key]=d
    missing=sorted(expected-set(jobs))
    fail(partial or not missing,f'Missing {len(missing)} of {len(expected)} required jobs; rerun with --partial for preliminary output')
    complete_curves=[curve for curve in curves if all((cell,curve,seed,'stressed') in jobs for cell in cells for seed in seeds)]
    return dict(manifest=m,manifest_sha256=mh,jobs=jobs,rows=flat,missing=missing,
                complete=not missing,expected_jobs=len(expected),complete_curves=complete_curves,
                source_hashes=m['source_sha256'],worker_sha256=hashes['worker'],protocol_sha256=hashes['protocol'])


def curve_bootstrap(curve_values,repeats=10000,seed=20260910):
    """Resample one GLOBAL curve index vector per replicate, shared by all cells."""
    x=np.asarray(curve_values);rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(x),size=(repeats,len(x)))
    return x[indices].mean(axis=1)


def interval(x):return np.percentile(x,[2.5,97.5]).tolist()


def action_diagnostics(rows):
    """Aggregate sufficient statistics recorded by the unchanged rollout worker.

    Corrections and boundary saturation concern six arm action axes. The worker
    records max_correction over all seven components, including the gripper.
    """
    steps=np.asarray([r['steps'] for r in rows],dtype=np.float64)
    weights=steps/steps.sum()
    correction=np.asarray([r['mean_correction'] for r in rows])
    saturation=np.asarray([r['saturation_fraction'] for r in rows])
    capacity=np.asarray([r['mean_actual_capacity'] for r in rows])
    return dict(episodes=len(rows),control_steps=int(steps.sum()),arm_action_component_count=int(6*steps.sum()),
        mean_abs_correction_episode_weighted=float(correction.mean()),
        mean_abs_correction_step_weighted=float(np.dot(correction,weights)),
        max_abs_correction_any_action_component=float(max(r['max_correction'] for r in rows)),
        boundary_saturation_fraction_episode_weighted=float(saturation.mean()),
        boundary_saturation_fraction_command_weighted=float(np.dot(saturation,weights)),
        mean_capacity_episode_weighted=float(capacity.mean()),
        mean_capacity_step_weighted=float(np.dot(capacity,weights)),
        min_episode_mean_capacity=float(capacity.min()),max_episode_mean_capacity=float(capacity.max()))


def summarize(data,bootstrap=10000):
    m=data['manifest'];cells=list(m['cells']);curves=data['complete_curves'];methods=m['methods']
    jobs=data['jobs'];conditions=m['conditions'];seeds=m['eval_seeds']
    result=dict(status='COMPLETE' if data['complete'] else 'PRELIMINARY — INCOMPLETE COVERAGE',
                complete=data['complete'],manifest_sha256=data['manifest_sha256'],
                coverage=dict(expected_jobs=data['expected_jobs'],validated_jobs=len(jobs),missing_jobs=data['missing'],
                              curves_used=curves,curves_expected=list(m['curves'])),
                bootstrap=dict(unit='curve; resampled globally across fixed cells',replicates=bootstrap,seed=20260910,
                               complete_curve_count=len(curves),intervals_available=len(curves)>=2,
                               unavailable_reason='At least two complete curves are required to estimate cluster uncertainty.' if len(curves)<2 else None,
                               interpretation='Conditional on fixed tasks, policies, checkpoints and evaluation episodes; not training-seed uncertainty.'),
                source_hashes=data['source_hashes'],worker_sha256=data['worker_sha256'],protocol_sha256=data['protocol_sha256'],
                overall={},per_cell={},negative_cases=[],case_results=[],healthy={})
    healthy=[r for r in data['rows'] if r['condition']=='healthy']
    result['healthy']={'validated_jobs':sum(k[3]=='healthy' for k in jobs),
                       'expected_jobs':len(cells)*len(seeds),'episodes_per_method':sum(r['method']=='base' for r in healthy),
                       'all_available_trajectories_identical':bool(healthy),'per_cell':{}}
    for cell in cells:
        h=[r for r in healthy if r['cell']==cell and r['method']=='base']
        result['healthy']['per_cell'][cell]=dict(episodes=len(h),sr_percent=100*np.mean([r['success'] for r in h]) if h else None)
    if not curves:return result
    # Average equal-sized episode blocks within each fixed curve/cell/condition.
    values=np.empty((len(curves),len(cells),len(conditions),len(methods)))
    for i,curve in enumerate(curves):
        for j,cell in enumerate(cells):
            for k,cond in enumerate(conditions):
                for l,method in enumerate(methods):
                    values[i,j,k,l]=100*np.mean([r['success'] for seed in seeds for r in jobs[(cell,curve,seed,'stressed')]['methods'][method][cond]['rows']])
                case=dict(cell=cell,curve=curve,condition=cond,sr_percent=dict(zip(methods,values[i,j,k].tolist())))
                result['case_results'].append(case)
                for method in ('tam','tam_dr'):
                    for ref in ('base','assumed_inverse'):
                        delta=values[i,j,k,methods.index(method)]-values[i,j,k,methods.index(ref)]
                        if delta<0:result['negative_cases'].append(dict(cell=cell,curve=curve,condition=cond,method=method,reference=ref,delta_pp=float(delta)))
    # No independently resampled cells/conditions: one shared curve draw keeps
    # paired comparisons and cross-policy correlation intact.
    curve_cell=values.mean(axis=2)
    draws=curve_bootstrap(curve_cell,bootstrap) if len(curves)>=2 else None
    def stats(point,boot):
        out={}
        for i,method in enumerate(methods):
            row=dict(sr_percent=float(point[i]),sr_ci95_percent=interval(boot[:,i]) if boot is not None else None)
            for ref,label in [('base','base'),('assumed_inverse','assumed'),('tam','original_tam')]:
                ri=methods.index(ref)
                row[f'delta_{label}_pp']=float(point[i]-point[ri])
                row[f'delta_{label}_ci95_pp']=interval(boot[:,i]-boot[:,ri]) if boot is not None else None
            out[method]=row
        return out
    result['overall']=stats(curve_cell.mean(axis=(0,1)),draws.mean(axis=1) if draws is not None else None)
    for j,cell in enumerate(cells):result['per_cell'][cell]=stats(curve_cell[:,j].mean(axis=0),draws[:,j] if draws is not None else None)
    used_rows=[r for r in data['rows'] if r['curve'] in curves and r['condition'] in conditions]
    for method in methods:
        selected=[r for r in used_rows if r['method']==method]
        result['overall'][method]['action_diagnostics']=action_diagnostics(selected)
        for cell in cells:
            result['per_cell'][cell][method]['action_diagnostics']=action_diagnostics([r for r in selected if r['cell']==cell])
    result['diagnostic_weighting']={
        'episode':'Every evaluated episode receives equal weight; complete curve blocks only, same coverage as success summaries.',
        'step':'Episode means weighted by actual executed policy-control steps; early success changes these denominators.',
        'command':'Boundary saturation weighted by six arm action components per policy-control step, before degradation; not physics-control torques.',
        'capacity':'Actual grouped capacity averaged across six arm axes; constant within each episode. Both episode- and step-weighted summaries are shown.',
        'limits':'No pre-clipping action magnitudes or count of commands changed by clamp are recorded. Boundary saturation at |command|>=.99999 cannot establish true clipping frequency; maximum correction includes the gripper.'}
    result['stressed_episodes_per_method']=len(curves)*len(cells)*len(conditions)*len(seeds)*m['episodes_per_seed']
    result['negative_case_counts']={f'{method}_vs_{ref}':sum(x['method']==method and x['reference']==ref for x in result['negative_cases']) for method in ('tam','tam_dr') for ref in ('base','assumed_inverse')}
    result['cases_per_comparison']=len(curves)*len(cells)*len(conditions)
    return result


NOTES = [
    'Evaluation seeds change rollout randomness; all adapters are fixed checkpoints. These are not independent training seeds.',
    'Curves are fresh parameter draws; the test does not establish transfer to entirely unseen function families or actuator dynamics below OSC.',
    'Cluster intervals are unavailable with fewer than two complete curves. Descriptive rates and paired differences remain available; a singleton curve cannot establish curve-level uncertainty.',
    'When available, the 95% percentile intervals resample whole curves globally across all fixed cells. They reflect curve variation conditional on these policies, tasks, checkpoints and sampled episodes; the number of curve clusters is recorded in the manifest.',
    'Oracle-capacity knows the true grouped multiplicative capacity, but not Gaussian draws and does not cancel current ripple. It is a diagnostic reference, not a stochastic task-success upper bound.',
    'Healthy diagnostics use the nominal training degradation model and remain separate from stressed averages. Their zero corrections and identical paired trajectories are checked directly.',
    'Action diagnostics use the same complete stressed curve blocks as success results. Mean absolute correction averages the six arm action axes; maximum correction covers all seven action components, including the gripper.',
    'Episode-weighted diagnostics give each rollout equal weight. Step-weighted corrections reconstruct the sum using each episode mean times its actual number of policy-control steps. Command-weighted boundary saturation uses six arm components per step; longer failed rollouts receive more weight. These are not physics torque-control weights.',
    'Boundary saturation means |command| >= .99999 before synthetic degradation. Saved rows cannot distinguish an actively clipped command from one naturally near the boundary; true clipping frequency is unavailable. Capacity summaries describe the actual grouped six-axis capacity, constant within an episode.',
    'Negative cases are listed without post-hoc exclusion or significance claims. Their denominator is fixed curve x cell x condition cases, pooling the planned evaluation seeds.',
]


def write_outputs(data,summary,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'aggregate.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    columns=['cell','curve','eval_seed','condition','method','episode','seed','success','steps','max_correction','mean_correction','saturation_fraction','max_gate','mean_actual_capacity','seconds','initial_hash','observation_hash','telemetry_hash','trajectory_hash']
    with (outdir/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(data['rows'])
    with (outdir/'negative_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cell','curve','condition','method','reference','delta_pp']);w.writeheader();w.writerows(summary['negative_cases'])
    lines=[f'# {summary["status"]}', '',f'Validated {len(data["jobs"])} / {data["expected_jobs"]} required jobs. Complete stressed curve blocks: {len(data["complete_curves"])} / {len(data["manifest"]["curves"])}.', '']
    if not data['complete']:lines+=['**Preliminary only. Missing jobs prevent a final completion claim. Summaries use only complete curve blocks spanning every fixed cell and evaluation seed.**','']
    ci_available=summary['bootstrap']['intervals_available']
    if not ci_available:
        ncurves=summary['bootstrap']['complete_curve_count']
        lines += [f'**Cluster intervals unavailable: {ncurves} complete curve'+('' if ncurves==1 else 's')+' available; at least two are required. Rates and gains below are descriptive only.**','']
    def table(title,stats):
        suffix=' [95% CI]' if ci_available else ''
        lines.extend([f'## {title}','','| Method | Success %'+suffix+' | Gain vs base pp'+suffix+' | Gain vs assumed inverse pp'+suffix+' |','|---|---:|---:|---:|'])
        for method,r in stats.items():
            entries=[]
            for value,ci in [('sr_percent','sr_ci95_percent'),('delta_base_pp','delta_base_ci95_pp'),('delta_assumed_pp','delta_assumed_ci95_pp')]:
                if r[ci] is None:entries.append(f'{r[value]:.1f}')
                else:
                    lo,hi=r[ci];entries.append(f'{r[value]:.1f} [{lo:.1f}, {hi:.1f}]')
            lines.append('| '+LABELS[method]+' | '+' | '.join(entries)+' |')
        lines.append('')
    lines += ['## Information available to each method','',
        '| Method | Capacity-model information | Role |',
        '|---|---|---|',
        '| Frozen base | None | Reference policy |',
        '| TeAR | Learned from nominal simulation inverse labels; telemetry at deployment | Main learned adapter |',
        '| DR-TeAR | Learned from randomized simulation inverse labels; telemetry at deployment | Existing learned variant |',
        '| Assumed-model inverse | Nominal training-model functions and telemetry; no actual held-out curves | Model-based comparison under mismatch |',
        '| Oracle-capacity inverse | Actual held-out capacity functions and telemetry | Privileged reference, not a deployable-information peer |','',
        'Both analytic methods use the correct evaluation-axis grouping. The oracle has extra test-model information; the assumed inverse does not. Neither cancels random noise or current ripple, and neither is an upper bound on task success.','']
    if summary['overall']:
        table('Stressed results: fixed-cell macro average',summary['overall'])
        for cell,stats in summary['per_cell'].items():table(cell,stats)
        lines += ['## Action and capacity diagnostics','',
            'Correction means and command-boundary rates below use the same stressed episodes as success results. E = equal episode weights; S = policy-control-step weights; C = six arm action components per step. Maximum correction includes the gripper. Capacity is the six-axis grouped capacity. Boundary saturation is not a count of commands changed by clipping.','']
        for scope,stats in [('Overall',summary['overall']),*summary['per_cell'].items()]:
            lines += [f'### {scope} diagnostics','',
                '| Method | Correction E / S | Max correction | Boundary saturation % E / C | Capacity E / S [episode min, max] | Policy steps |',
                '|---|---:|---:|---:|---:|---:|']
            for method,r in stats.items():
                d=r['action_diagnostics']
                lines.append(f'| {LABELS[method]} | {d["mean_abs_correction_episode_weighted"]:.4f} / {d["mean_abs_correction_step_weighted"]:.4f} | {d["max_abs_correction_any_action_component"]:.4f} | {100*d["boundary_saturation_fraction_episode_weighted"]:.2f} / {100*d["boundary_saturation_fraction_command_weighted"]:.2f} | {d["mean_capacity_episode_weighted"]:.3f} / {d["mean_capacity_step_weighted"]:.3f} [{d["min_episode_mean_capacity"]:.3f}, {d["max_episode_mean_capacity"]:.3f}] | {d["control_steps"]} |')
            lines.append('')
        r=summary['overall']['tam_dr']
        if r['delta_original_tam_ci95_pp'] is None:ci_text='cluster intervals unavailable'
        else:
            lo,hi=r['delta_original_tam_ci95_pp'];ci_text=f'paired curve-cluster 95% CI [{lo:+.1f}, {hi:+.1f}]'
        lines += [f'DR minus original TAM: {r["delta_original_tam_pp"]:+.1f} pp, {ci_text}. This is a comparison of existing variants, subject to the configuration differences below.','']
        lines+=['## Negative paired cases','',f'Denominator: {summary["cases_per_comparison"]} curve × cell × condition cases per comparison.','']
        for key,count in summary['negative_case_counts'].items():lines.append(f'- {key}: {count}')
        lines+=['','Full list: `negative_cases.csv`. No losing cases are removed from the averages.','']
    else:lines+=['No complete stressed curve block is available; inferential summaries are withheld.','']
    h=summary['healthy'];lines+=['## Healthy identity diagnostics','',f'{h["validated_jobs"]} / {h["expected_jobs"]} jobs; {h["episodes_per_method"]} paired episodes per method. All available comparisons passed exact trajectory, outcome, zero-correction and zero-gate checks.','']
    for cell,r in h['per_cell'].items():lines.append(f'- {cell}: {r["episodes"]} episodes; success {r["sr_percent"] if r["sr_percent"] is not None else "pending"}%.')
    lines+=['','## Interpretation and protocol','']+[f'- {note}' for note in NOTES]
    lines+=['','## Coverage and provenance','',f'Manifest SHA256: `{data["manifest_sha256"]}`.',f'Worker SHA256: `{data["worker_sha256"]}`.',f'Protocol SHA256: `{data["protocol_sha256"]}`.','Source snapshot hashes were verified against the manifest.','']
    if data['missing']:lines+=['Missing jobs:','']+[f'- {" / ".join(map(str,k))}' for k in data['missing']]
    (outdir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    status='COMPLETE' if data['complete'] else 'PRELIMINARY: INCOMPLETE COVERAGE'
    latex=[f'% {status}',r'\begin{table}[t]',r'\centering',r'\begin{tabular}{lrr}',r'\toprule',r'Method & SR (\%) & $\Delta$ base (pp), 95\% CI \\',r'\midrule']
    if not ci_available:
        latex=[line.replace(r', 95\% CI','') for line in latex]
        latex.insert(1,'% Cluster intervals unavailable: fewer than two complete curves.')
    for method,r in summary['overall'].items():
        if method=='oracle_capacity':latex.append(r'\midrule')
        ci_text=''
        if r['delta_base_ci95_pp'] is not None:
            lo,hi=r['delta_base_ci95_pp'];ci_text=f' [{lo:+.1f}, {hi:+.1f}]'
        latex.append(f'{LABELS[method]} & {r["sr_percent"]:.1f} & {r["delta_base_pp"]:+.1f}'+ci_text+' '+r'\\')
    latex += [r'\bottomrule',r'\end{tabular}',r'\caption{'+('PRELIMINARY. ' if not data['complete'] else '')+r'Fresh degradation-curve evaluation. Macro averages over fixed task--policy cells; confidence intervals use a paired global curve-cluster bootstrap. The assumed inverse uses nominal training curves only. The separate oracle row is privileged: it knows actual test capacities, unlike TeAR. Oracle-capacity does not cancel noise or ripple and is not a task-success upper bound.}',r'\label{tab:heldout_capacity}',r'\end{table}']
    if not ci_available:
        latex=[line.replace('confidence intervals use a paired global curve-cluster bootstrap.', 'cluster intervals unavailable with fewer than two complete curves; values are descriptive only.') for line in latex]
    (outdir/'table.tex').write_text('\n'.join(latex)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    if summary['overall']:
        entries=[]
        for scope,stats in [('Overall',summary['overall']),*summary['per_cell'].items()]:
            for method in ('tam','tam_dr'):entries.append((f'{scope}: {LABELS[method]}',method,stats[method]))
        for y,(label,method,r) in enumerate(entries):
            x=r['delta_base_pp'];color='#2166ac' if method=='tam' else '#b35806'
            if r['delta_base_ci95_pp'] is not None:
                lo,hi=r['delta_base_ci95_pp'];ax.plot([lo,hi],[y,y],color=color,lw=2)
            ax.plot(x,y,'o',color=color)
        ax.set_yticks(range(len(entries)),[x[0] for x in entries]);ax.invert_yaxis();ax.axvline(0,color='gray',ls='--',lw=1)
        ax.set_xlabel('Success gain over paired frozen base (percentage points)')
        ax.grid(axis='x',alpha=.2)
    else:ax.text(.5,.5,'No complete stressed curve blocks yet',ha='center',transform=ax.transAxes);ax.set_axis_off()
    ax.set_title(('Paired curve-cluster 95% intervals' if ci_available else 'Descriptive gains — cluster intervals unavailable')+(' — PRELIMINARY' if not data['complete'] else ''))
    fig.tight_layout();fig.savefig(outdir/'forest.pdf');fig.savefig(outdir/'forest.png',dpi=200);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=HERE/'manifest.json')
    ap.add_argument('--results',type=Path,default=HERE/'results')
    ap.add_argument('--out',type=Path,default=HERE/'summary')
    ap.add_argument('--partial',action='store_true');ap.add_argument('--bootstrap',type=int,default=10000)
    args=ap.parse_args();fail(args.bootstrap>=100,'At least 100 bootstrap replicates required')
    data=load_results(args.manifest,args.results,args.partial);s=summarize(data,args.bootstrap)
    write_outputs(data,s,args.out)
    print(f'{s["status"]}: {len(data["jobs"])}/{data["expected_jobs"]} jobs; report: {args.out/"REPORT.md"}')

if __name__=='__main__':main()
