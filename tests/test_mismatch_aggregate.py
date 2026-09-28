import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.mismatch import aggregate as a

class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.results = self.root/'results'; self.results.mkdir()
        for name in ('worker.py','protocol.py','source.py'):
            (self.root/name).write_text(name)
        self.m = dict(cells={'cell':{}},curves={'c0':{},'c1':{}},eval_seeds=[1],
                      episodes_per_seed=2,conditions=['hot'],methods=a.METHODS,
                      source_sha256={'source.py':a.file_hash(self.root/'source.py')})
        self.mp=self.root/'manifest.json';self.mp.write_text(json.dumps(self.m))
        for curve in self.m['curves']: self.write_job(curve,['hot'])
        self.write_job('c0',['healthy'])
    def tearDown(self): self.tmp.cleanup()
    def write_job(self,curve,conditions):
        d=dict(cell='cell',curve=curve,eval_seed=1,episodes=2,pairing_verified=True,
               manifest_sha256=a.file_hash(self.mp),worker_sha256=a.file_hash(self.root/'worker.py'),
               protocol_sha256=a.file_hash(self.root/'protocol.py'),methods={})
        for method in a.METHODS:
            d['methods'][method]={}
            for cond in conditions:
                rows=[]
                for ep in range(2):
                    r=dict(episode=ep,seed=a.stable_seed('cell',curve,cond,1,ep),success=int(method!='base' and cond!='healthy'),
                           initial_hash=str(ep),observation_hash=str(ep),telemetry_hash=str(ep),trajectory_hash=str(ep),
                           max_correction=0.,steps=10,mean_correction=0.,saturation_fraction=0.,max_gate=0.,mean_actual_capacity=1.,seconds=1.)
                    rows.append(r)
                d['methods'][method][cond]={'sr':sum(x['success'] for x in rows)/2,'rows':rows}
        name=f'{curve}_{conditions[0]}.json';(self.results/name).write_text(json.dumps(d))
    def load(self,partial=False):return a.load_results(self.mp,self.results,partial=partial)
    def mutate(self,fn):
        p=self.results/'c0_hot.json';d=json.loads(p.read_text());fn(d);p.write_text(json.dumps(d))
    def test_complete_and_paired_effect(self):
        data=self.load();self.assertTrue(data['complete'])
        s=a.summarize(data,bootstrap=200)
        self.assertEqual(s['overall']['tam']['delta_base_pp'],100.)
        self.assertEqual(s['overall']['tam']['delta_base_ci95_pp'],[100.,100.])
        self.assertEqual(s['healthy']['episodes_per_method'],2)
    def test_missing_job_rejected_and_partial_flagged(self):
        (self.results/'c1_hot.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing'):self.load()
        d=self.load(True);self.assertFalse(d['complete']);self.assertEqual(d['complete_curves'],['c0'])
        self.assertEqual(a.summarize(d,200)['overall']['tam']['action_diagnostics']['episodes'],2)
    def test_hash_rejected(self):
        self.mutate(lambda d:d.update(manifest_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'manifest'):self.load(True)
    def test_source_change_rejected(self):
        (self.root/'source.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Source'):self.load()
    def test_pairing_rejected(self):
        self.mutate(lambda d:d['methods']['tam']['hot']['rows'][0].update(initial_hash='wrong'))
        with self.assertRaisesRegex(ValueError,'Pairing'):self.load()
    def test_episode_count_rejected(self):
        self.mutate(lambda d:d['methods']['tam']['hot']['rows'].pop())
        with self.assertRaisesRegex(ValueError,'episode'):self.load(True)
    def test_missing_condition_rejected(self):
        self.mutate(lambda d:d['methods']['tam'].clear())
        with self.assertRaisesRegex(ValueError,'condition'):self.load(True)
    def test_healthy_mismatch_rejected(self):
        p=self.results/'c0_healthy.json';d=json.loads(p.read_text())
        d['methods']['tam']['healthy']['rows'][0]['trajectory_hash']='bad';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'Healthy'):self.load()
    def test_duplicate_job_rejected(self):
        (self.results/'duplicate.json').write_text((self.results/'c0_hot.json').read_text())
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.load()
    def test_protocol_change_rejected(self):
        (self.root/'protocol.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'protocol hash'):self.load()
    def test_artifacts_and_preliminary_status(self):
        (self.results/'c1_hot.json').unlink()
        d=self.load(True);s=a.summarize(d,bootstrap=200)
        out=self.root/'summary';a.write_outputs(d,s,out)
        for name in ('aggregate.json','episodes.csv','negative_cases.csv','REPORT.md','table.tex','forest.pdf','forest.png'):
            self.assertGreater((out/name).stat().st_size,20)
        self.assertIn('PRELIMINARY',(out/'REPORT.md').read_text())
        self.assertIn('Action and capacity diagnostics',(out/'REPORT.md').read_text())
        self.assertIn('not a count of commands changed by clipping',(out/'REPORT.md').read_text())
        self.assertIn('PRELIMINARY',(out/'table.tex').read_text())
        self.assertFalse(json.loads((out/'aggregate.json').read_text())['complete'])
    def test_diagnostics_respect_variable_episode_lengths(self):
        for path in self.results.glob('*_hot.json'):
            d=json.loads(path.read_text())
            for method in a.METHODS:
                for r in d['methods'][method]['hot']['rows']:
                    short=r['episode']==0
                    r.update(steps=1 if short else 9,
                             mean_correction=.2 if short else .8,
                             max_correction=.3 if short else .9,
                             saturation_fraction=1. if short else 0.,
                             mean_actual_capacity=.2 if short else .8)
            path.write_text(json.dumps(d))
        s=a.summarize(self.load(),200)
        for stats in [s['overall'],s['per_cell']['cell']]:
            r=stats['tam']['action_diagnostics']
            self.assertAlmostEqual(r['mean_abs_correction_episode_weighted'],.5)
            self.assertAlmostEqual(r['mean_abs_correction_step_weighted'],.74)
            self.assertAlmostEqual(r['boundary_saturation_fraction_episode_weighted'],.5)
            self.assertAlmostEqual(r['boundary_saturation_fraction_command_weighted'],.1)
            self.assertAlmostEqual(r['mean_capacity_episode_weighted'],.5)
            self.assertAlmostEqual(r['mean_capacity_step_weighted'],.74)
            self.assertEqual(r['max_abs_correction_any_action_component'],.9)
            self.assertEqual(r['arm_action_component_count'],120)
    def test_missing_diagnostic_denominator_rejected(self):
        self.mutate(lambda d:d['methods']['tam']['hot']['rows'][0].pop('steps'))
        with self.assertRaisesRegex(ValueError,'diagnostic'):self.load()
    def test_single_curve_has_descriptions_but_no_cluster_intervals(self):
        (self.results/'c1_hot.json').unlink()
        d=self.load(True);s=a.summarize(d,200)
        self.assertFalse(s['bootstrap']['intervals_available'])
        self.assertEqual(s['bootstrap']['complete_curve_count'],1)
        for stats in [s['overall'],s['per_cell']['cell']]:
            self.assertEqual(stats['tam']['sr_percent'],100.)
            for method,r in stats.items():
                for key,value in r.items():
                    if 'ci95' in key:self.assertIsNone(value)
        out=self.root/'single';a.write_outputs(d,s,out)
        report=(out/'REPORT.md').read_text();tex=(out/'table.tex').read_text()
        self.assertIn('1 complete curve',report)
        self.assertIn('intervals unavailable',report)
        self.assertNotIn('100.0 [100.0, 100.0]',report)
        self.assertIn('intervals unavailable',tex)
    def test_global_curve_bootstrap_preserves_pairing(self):
        x=np.array([[0.,1.],[1.,0.]])
        # The across-cell mean equals .5 for every sampled curve, so resampling
        # globally produces zero interval width despite opposite cell effects.
        boot=a.curve_bootstrap(x,1000,8)
        self.assertTrue(np.all(boot.mean(axis=1)==.5))

if __name__=='__main__':unittest.main()
