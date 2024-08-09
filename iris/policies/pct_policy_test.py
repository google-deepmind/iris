# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym import spaces
from iris.policies import pct_policy
import jax
import numpy as np
from absl.testing import absltest


class PCTPolicyTest(absltest.TestCase):

  def test_pct_encoder(self):
    batch_size = 10
    num_points = 100
    in_dim = 3
    out_dim = 8
    pc = np.random.normal(size=(batch_size, num_points, in_dim))
    pct_enc = pct_policy.PCTEncoder(emb_dim=out_dim, attention_type='perf-relu')
    params = pct_enc.init(jax.random.PRNGKey(0), pc)
    emb = pct_enc.apply(params, pc)
    self.assertEqual(emb.shape, (batch_size, out_dim))

  def test_pct_encoder_with_mask(self):
    batch_size = 10
    num_points = 100
    masked_points = 20
    in_dim = 3
    out_dim = 8
    pc = np.random.normal(size=(batch_size, num_points, in_dim))
    mask = np.concatenate(
        [np.ones((batch_size, num_points-masked_points)),
         np.zeros((batch_size, masked_points))], axis=-1)
    mask = mask.astype(bool)
    pct_enc = pct_policy.PCTEncoder(emb_dim=out_dim)
    params = pct_enc.init(jax.random.PRNGKey(0), pc)
    emb = pct_enc.apply(params, pc, mask)
    self.assertEqual(emb.shape, (batch_size, out_dim))

    emb_no_mask = pct_enc.apply(params, pc[:, :-masked_points], None)
    np.testing.assert_allclose(emb_no_mask, emb, atol=1e-1, rtol=1e-1)

  def test_policy_act(self):
    """Tests the act function for PCT policy."""

    ob_space = spaces.Dict({
        'object_position': spaces.Box(
            low=-5, high=5, shape=(3,)
        ),
        'object_bounding_box': spaces.Box(
            low=-1, high=1, shape=(3,)
        ),
        'object_point_cloud': spaces.Box(low=-1, high=1, shape=(3, 100, 1)),
    })

    policy = pct_policy.PCTPolicy(
        ob_space=ob_space,
        ac_space=spaces.Box(low=-3, high=3, shape=(7,)),
        auxiliary_observations=['object_position', 'object_bounding_box',],
    )

    n = policy.get_weights().shape[0]
    policy.update_weights(new_weights=np.zeros(n))

    x = ob_space.sample()

    jax_act = policy.act(x)
    np.testing.assert_array_almost_equal(jax_act, np.zeros(7), 2)


if __name__ == '__main__':
  absltest.main()
