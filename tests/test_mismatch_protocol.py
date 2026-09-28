import sys
import unittest
from pathlib import Path

import numpy as np

from experiments.mismatch import protocol as p


class ProtocolTests(unittest.TestCase):
    def test_grouping_is_product_of_group_means(self):
        factors = np.array([[.2,.4,.6,.8,1,.3,.5], [.9,.8,.7,.6,.5,.4,.3], [.3,.5,.7,.9,.4,.6,.8]])
        got = p.axis_capacity(factors)
        np.testing.assert_allclose(got[:3], np.prod(factors[:, :4].mean(1)))
        np.testing.assert_allclose(got[3:6], np.prod(factors[:, 3:7].mean(1)))
        self.assertEqual(got[6], 1)

    def test_assumed_inverse_uses_actual_training_curves_and_preserves_gripper(self):
        models = p.make_models(p.TRAIN_SPEC)
        q = p.capacity(models, np.full(7, 59.), np.full(7, .8), np.full(7, .7))
        np.testing.assert_allclose(q[:6], .525*.55*.55)
        action = np.array([.01,-.02,.03,-.04,.05,-.06,.7])
        out = p.inverse(action, q)
        np.testing.assert_allclose(out*q, action)
        self.assertEqual(out[6], action[6])

    def test_healthy_identity(self):
        q = p.capacity(p.make_models(p.TRAIN_SPEC), np.full(7,30),np.full(7,.3),np.full(7,.96))
        a = np.linspace(-1,1,7)
        np.testing.assert_array_equal(p.inverse(a,q), a)

    def test_noise_stream_does_not_change_global_rng(self):
        np.random.seed(18)
        expected = np.random.random(10)
        np.random.seed(18)
        models = p.make_models(p.TRAIN_SPEC)
        p.degrade(np.ones(7)*.3,models,np.ones(7)*65,np.ones(7)*.8,np.ones(7)*.7,123,0)
        np.testing.assert_array_equal(np.random.random(10),expected)

    def test_same_episode_noise_and_ripple_independent_of_prior_rollout(self):
        models = p.make_models(p.TRAIN_SPEC)
        args=(np.ones(7)*.3,models,np.ones(7)*65,np.ones(7)*.8,np.ones(7)*.7)
        expected=p.degrade(*args,123,0)
        for t in range(9): p.degrade(*args,999,t)
        np.testing.assert_array_equal(p.degrade(*args,123,0), expected)

    def test_noiseless_reachable_action_recovered(self):
        models=p.make_models(p.TRAIN_SPEC)
        models[0].torque_noise_scale=0; models[1].noise_scale=0; models[1].ripple_scale=0; models[2].lag_scale=0
        T=np.linspace(45,70,7);C=np.linspace(.5,.9,7);V=np.linspace(.7,1,7)
        a=np.array([.01,-.02,.03,-.04,.05,-.06,.7])
        q=p.capacity(models,T,C,V)
        np.testing.assert_allclose(p.degrade(p.inverse(a,q),models,T,C,V,1,0),a,atol=1e-8)


if __name__ == '__main__': unittest.main()
