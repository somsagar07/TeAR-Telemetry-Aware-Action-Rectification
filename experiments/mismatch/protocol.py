"""Fixed-protocol capacity baselines and independent degradation randomness."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from env.thermal_model import ThermalModel
from env.telemetry_model import CurrentModel, VoltageModel

TRAIN_SPEC = {ch: ['linear', {}] for ch in ('T','C','V')}
METHODS = ['base', 'assumed_inverse', 'tam', 'tam_dr', 'oracle_capacity']
STRESSED = ['hot', 'T_mod', 'TC_mod', 'TCV_mod']


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], 'little')


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def make_models(spec):
    return tuple(cls.from_predefined(spec[ch][0],params=spec[ch][1],n_joints=7)
                 for cls,ch in zip((ThermalModel,CurrentModel,VoltageModel),('T','C','V')))


def joint_capacity(models,T,C,V):
    return np.array([[m.compute_degradation(float(x),j) for j,x in enumerate(vals)]
                     for m,vals in zip(models,(T,C,V))])


def axis_capacity(factors):
    pos=np.prod(factors[:,:4].mean(axis=1))
    rot=np.prod(factors[:,3:7].mean(axis=1))
    return np.array([pos]*3+[rot]*3+[1.])


def capacity(models,T,C,V):
    return axis_capacity(joint_capacity(models,T,C,V))


def inverse(action,q):
    # No artificial .05 floor on the composed capacity: bounded output already
    # makes the command feasible, and each actual channel capacity is positive.
    out=np.clip(np.asarray(action)/np.maximum(q,1e-12),-1,1)
    out[6]=action[6]
    return out


def degrade(action,models,T,C,V,noise_seed,step):
    # Source implementation uses np.random. Isolate it from env and policy RNG,
    # and reset current ripple phase to the episode-local timestep explicitly.
    rng_state=np.random.get_state()
    try:
        np.random.seed(stable_seed(noise_seed,step))
        models[1]._step_counter=step
        a=models[0].apply_thermal_physics(action,T)
        a=models[1].apply_current_physics(a,C)
        return models[2].apply_voltage_physics(a,V)
    finally:
        np.random.set_state(rng_state)


def telemetry(condition,seed):
    rng=np.random.RandomState(seed)
    if condition=='healthy': return np.full(7,30.),np.full(7,.3),np.full(7,.96)
    if condition=='cool': return rng.uniform(20,42,7),rng.uniform(.1,.5,7),rng.uniform(.92,1.,7)
    if condition=='hot': return rng.uniform(56,75,7),rng.uniform(.1,.5,7),rng.uniform(.92,1.,7)
    constants={'T_mod':(58.,.3,1.),'TC_mod':(55.,.75,1.),'TCV_mod':(55.,.75,.72)}
    return tuple(np.full(7,x) for x in constants[condition])


def verify_sources(manifest_path):
    """Reject changed sources or execution from a different checkout."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    if (root / manifest['code_root']).resolve() != HERE:
        raise ValueError('Manifest belongs to another source checkout')
    expected = {(root / name).resolve(): digest
                for name, digest in manifest['source_sha256'].items()}
    for path, digest in expected.items():
        if file_hash(path) != digest:
            raise ValueError(f'Source changed since manifest freeze: {path}')
    repo = HERE.parents[1]
    prefixes = ('env', 'thermal_adapters', 'tam_v2', 'experiments.mismatch')
    for name, module in tuple(sys.modules.items()):
        if not (name == 'scripts.eval_transfer_any_base' or
                any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes)):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(repo) or path not in expected:
            raise ValueError(f'Imported source is outside the frozen checkout: {name}')
