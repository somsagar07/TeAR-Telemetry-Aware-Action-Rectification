"""Paired held-out rollouts; only checkpoints and frozen local source are read."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
# Required by deterministic CUDA matrix multiplication; set before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from . import protocol as p
from thermal_adapters.checkpoint import load_adapter as load_tear


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def load_adapter(entry, device):
    return load_tear(entry['path'], config=entry['config'], device=device,
                     alpha=entry['alpha'], gamma_range=entry['gamma_range'])


def array_hash(*arrays):
    h=hashlib.sha256()
    for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def episode(env,policy,adapter,method,task,condition,curve,run_seed,ep,horizon):
    seed=p.stable_seed(task,curve,condition,run_seed,ep)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    raw=env.reset()
    initial=array_hash(env.sim.get_state().flatten())
    obs=legacy.build_robomimic_obs(raw)
    obs_hash=array_hash(*[obs[k] for k in sorted(obs)])
    policy.start_episode()
    transformer='Transformer' in type(policy.policy).__name__
    fb=legacy.FrameBuffer(10 if transformer else 1);fb.push(obs)
    T,C,V=[a.astype(np.float32) for a in p.telemetry(condition,p.stable_seed(seed,'telemetry'))]
    tele_hash=array_hash(T,C,V)
    expected_models=episode.expected_models
    actual_models=episode.actual_models
    q_assumed=p.capacity(expected_models,T,C,V)
    q_actual=p.capacity(actual_models,T,C,V)
    trajectory=hashlib.sha256()
    max_correction=0.;correction_sum=0.;saturated=0;max_gate=0.
    start=time.monotonic();success=False;sample_trace=[]
    for step in range(horizon):
        stacked=fb.stacked()
        po=stacked if transformer else {k:v[-1] for k,v in stacked.items()}
        # Give stochastic policies the same independent random stream at every
        # paired timestep, without consuming environment randomness.
        state_np=np.random.get_state();state_py=random.getstate()
        ps=p.stable_seed(seed,'policy',step)
        np.random.seed(ps);random.seed(ps);torch.manual_seed(ps)
        try:
            with torch.no_grad():a_base=np.asarray(policy(ob=po),dtype=np.float32).reshape(-1).clip(-1,1)
        finally:
            np.random.set_state(state_np);random.setstate(state_py)
        action=a_base.copy()
        if method=='assumed_inverse':action=p.inverse(a_base,q_assumed).astype(np.float32)
        elif method=='oracle_capacity':action=p.inverse(a_base,q_actual).astype(np.float32)
        elif adapter is not None:
            state=legacy.state_from_obs({k:v[-1] for k,v in stacked.items()})
            if len(state)!=adapter.state_dim:raise ValueError(f'state dimension {len(state)} != {adapter.state_dim}')
            inputs=[torch.as_tensor(x,device=episode.device,dtype=torch.float32)[None] for x in (a_base,T,state,C,V)]
            with torch.no_grad():
                action=adapter(*inputs)[0][0].cpu().numpy()
                gate=adapter.per_joint_gate(inputs[1],inputs[3],inputs[4])
            max_gate=max(max_gate,float(gate.max()))
        d=np.abs(action-a_base)
        max_correction=max(max_correction,float(d.max()));correction_sum+=float(d[:6].mean())
        saturated+=int(np.count_nonzero(np.abs(action[:6])>=.99999))
        executed=p.degrade(action,actual_models,T,C,V,p.stable_seed(seed,'noise'),step).astype(np.float32)
        raw,_,_,_=env.step(executed)
        trajectory.update(a_base.tobytes());trajectory.update(action.tobytes())
        trajectory.update(env.sim.get_state().flatten().tobytes())
        if step<3:sample_trace.append({'base':a_base.tolist(),'command':action.tolist()})
        fb.push(legacy.build_robomimic_obs(raw))
        if legacy.is_success(env):success=True;break
    if condition=='healthy' and method in ('tam','tam_dr','tam_exp') and max_correction!=0:
        raise AssertionError(f'nominal correction {max_correction}')
    return dict(episode=ep,seed=seed,success=int(success),steps=step+1,initial_hash=initial,
        observation_hash=obs_hash,telemetry_hash=tele_hash,trajectory_hash=trajectory.hexdigest(),
        max_correction=max_correction,mean_correction=correction_sum/(step+1),
        saturation_fraction=saturated/(6*(step+1)),max_gate=max_gate,
        mean_actual_capacity=float(q_actual[:6].mean()),seconds=time.monotonic()-start,
        first_actions=sample_trace)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--cell',required=True)
    ap.add_argument('--curve',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--episodes',type=int)
    ap.add_argument('--conditions',nargs='+');ap.add_argument('--methods',nargs='+')
    a=ap.parse_args();manifest=json.loads(Path(a.manifest).read_text());cell=manifest['cells'][a.cell]
    if p.file_hash(cell['ckpt'])!=cell['ckpt_sha256']:raise ValueError('Base checkpoint changed since manifest freeze')
    for method in (a.methods or manifest['methods']):
        if method in ('tam','tam_dr','tam_exp') and p.file_hash(cell[method]['path'])!=cell[method]['sha256']:
            raise ValueError(f'{method} checkpoint changed since manifest freeze')
    p.verify_sources(a.manifest)
    global legacy
    from scripts import eval_transfer_any_base as legacy
    p.verify_sources(a.manifest)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu';episode.device=device
    policy,_=legacy.policy_from_checkpoint(ckpt_path=cell['ckpt'],device=device,verbose=False)
    methods=a.methods or manifest['methods'];conditions=a.conditions or manifest['conditions']
    n=a.episodes or manifest['episodes_per_seed']
    adapters={m:load_adapter(cell[m],device) for m in methods if m in ('tam','tam_dr','tam_exp')}
    spec=manifest['curves'][a.curve]
    # All selected checkpoints use agentview only. Rendering an unused wrist
    # camera doubles graphics work without changing any policy observation.
    env=legacy.suite.make(legacy.TASK_TO_ENV[cell['task']],robots='Panda',
        controller_configs=legacy.suite.load_controller_config(default_controller='OSC_POSE'),
        has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,
        camera_names=['agentview'],camera_heights=84,camera_widths=84,
        reward_shaping=False,ignore_done=True,horizon=cell['horizon'],control_freq=20)
    result=dict(cell=a.cell,curve=a.curve,eval_seed=a.seed,episodes=n,manifest_sha256=p.file_hash(a.manifest),
                methods={},pairing_verified=False,device=device,worker_sha256=p.file_hash(__file__),
                protocol_sha256=p.file_hash(p.__file__))
    try:
        for condition in conditions:
            episode.actual_models=p.make_models(p.TRAIN_SPEC if condition=='healthy' else spec)
            episode.expected_models=p.make_models(p.TRAIN_SPEC)
            reference=None
            for method in methods:
                rows=[]
                for ep in range(n):
                    row=episode(env,policy,adapters.get(method),method,cell['task'],condition,a.curve,a.seed,ep,cell['horizon'])
                    rows.append(row)
                    print(f'{a.cell} {a.curve} seed={a.seed} {condition} {method} ep={ep} success={row["success"]} steps={row["steps"]} sec={row["seconds"]:.1f}',flush=True)
                if reference is None:reference=rows
                else:
                    for x,y in zip(reference,rows):
                        for key in ('initial_hash','observation_hash','telemetry_hash'):
                            if x[key]!=y[key]:raise AssertionError(f'Pairing failed {condition} {method} ep={y["episode"]}: {key}')
                        if condition=='healthy' and x['trajectory_hash']!=y['trajectory_hash']:
                            raise AssertionError(f'Nominal trajectory mismatch {method} ep={y["episode"]}')
                result['methods'].setdefault(method,{})[condition]={'sr':float(np.mean([r['success'] for r in rows])),'rows':rows}
                atomic_json(a.out+'.partial',result)
        result['pairing_verified']=True;atomic_json(a.out,result)
    finally:
        env.close()


if __name__=='__main__':main()
