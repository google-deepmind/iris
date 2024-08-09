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

"""Policy class that computes action by running jax neural networks."""
import functools
from typing import Dict, Union

from flax import linen as nn
import gym
from gym.spaces import utils
from iris.policies import jax_policy
import jax.numpy as jnp
import numpy as np
from scenic.projects.meshdynamics import model
from scenic.projects.pointcloud import models


class PCTEncoder(nn.Module):
  """Point Cloud Transformer Encoder.

  Attributes:
    in_dim: Point cloud feature dimension.
    feature_dim: Point cloud encoder feature dim.
    emb_dim: Point cloud encoder output dim.
    emb_idx: Point cloud index to extract the embedding. If None, all points
      will be returned.
    encoder_feature_dim: Point cloud encoder feature dim.
    kernel_size: Point cloud encoder kernel size.
    num_attention_layers: Number of attention layers in the point cloud encoder.
    attention_type: str defining attention algorithm; possible values
      are 'regular', 'perf-softmax', 'perf-relu'
    rpe_masking_type: str defining applied RPE mechanism; possible values
      are 'nomask', 'fft', 'flt'
  """

  in_dim: int = 3
  feature_dim: int = 16
  emb_dim: int = 8
  emb_idx: int | None = 0
  encoder_feature_dim: int = 16
  kernel_size: int = 1
  num_attention_layers: int = 2
  num_pre_conv_layers: int = 2
  attention_type: str = 'regular'
  rpe_masking_type: str = 'nomask'
  pseudolocal_sigma: float = 0.05

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      mask: jnp.ndarray | None = None,
      train: bool = True,
      coords: jnp.ndarray | None = None,
  ):
    """Runs PCT encoder and generates point cloud embedding.

    Args:
      x: A point clound of shape (batch_sizes, N, in_dim).
      mask: Optional boolean mask of shape (batch_sizes, N) applied to the point
        cloud.
      train: Whether the module is called during training.
      coords: xyz-coordinates of the points of shape (batch_sizes, N, 3).

    Returns:
      The embedding array of shape (batch_sizes, emb_dim).
    """
    batch_dim = True
    if jnp.ndim(x) == 2:
      batch_dim = False
      x = x[None, :, :]
      if mask is not None:
        mask = mask[None, :]
      if coords is not None:
        coords = coords[None, :, :]

    if self.attention_type == 'regular':
      x = model.PointCloudTransformerEncoder(
          in_dim=self.in_dim,
          feature_dim=self.feature_dim,
          out_dim=self.emb_dim,
          encoder_feature_dim=self.encoder_feature_dim,
          kernel_size=self.kernel_size,
          num_attention_layers=self.num_attention_layers,
          num_pre_conv_layers=self.num_pre_conv_layers,
      )(x, mask)
    else:
      if self.attention_type == 'pseudolocal-performer':
        attention_fn_configs = dict()
        attention_fn_configs['attention_kind'] = 'performer'
        attention_fn_configs['performer'] = {
            'masking_type': 'pseudolocal',
            'rf_type': 'hyper',
            'num_features': 128,
            'sigma': self.pseudolocal_sigma,
        }
      else:
        kernel_name_translate = {'perf-softmax': 'softmax', 'perf-relu': 'relu'}
        rpe_mask_name_translate = {
            'fft': 'fftmasked',
            'flt': 'sharpmasked',
            'nomask': 'nomask',
        }
        num_features_translate = {'perf-softmax': 64, 'perf-relu': 0}
        use_rand_proj_translate = {'perf-softmax': True, 'perf-relu': False}

        attention_fn_configs = dict()
        attention_fn_configs['attention_kind'] = 'performer'
        attention_fn_configs['performer'] = {
            'masking_type': rpe_mask_name_translate[self.rpe_masking_type],
            'kernel_transformation': kernel_name_translate[self.attention_type],
            'num_features': num_features_translate[self.attention_type],
            'rpe_method': None,
            'num_realizations': 10,
            'num_sines': 1,
            'use_random_projections': use_rand_proj_translate[
                self.attention_type
            ],
            'seed': 41,
        }

      x = models.PointCloudTransformerEncoder(
          in_dim=self.in_dim,
          feature_dim=self.feature_dim,
          out_dim=self.emb_dim,
          encoder_feature_dim=self.encoder_feature_dim,
          kernel_size=self.kernel_size,
          num_attention_layers=self.num_attention_layers,
          num_pre_conv_layers=self.num_pre_conv_layers,
          attention_fn_configs=attention_fn_configs,
      )(x, mask=mask, coords=coords)

    if self.emb_idx is not None:
      x = x[:, self.emb_idx, :]

    if not batch_dim:
      return x[0]

    return x


class PCTPolicyNet(nn.Module):
  """Point Cloud Transformer Net.

  Attributes:
    auxiliary_observations: List of names of observations other than the point
      cloud.
    in_dim: Point cloud feature dimension.
    feature_dim: Point cloud encoder feature dim.
    emb_dim: Point cloud encoder output dim.
    encoder_feature_dim: Point cloud encoder feature dim.
    kernel_size: Point cloud encoder kernel size.
    num_attention_layers: Number of attention layers in the point cloud encoder.
    fc_layer_dims: Fully connected layer dims.
    act_dim: Output action dim.
    attention_type: str defining attention algorithm; possible values
      are 'regular', 'perf-softmax', 'perf-relu'
    rpe_masking_type: str defining applied RPE mechanism; possible values
      are 'nomask', 'fft', 'flt'
  """

  auxiliary_observations: list[str]
  point_cloud_obs_name: str = 'object_point_cloud'
  point_cloud_mask_name: str = 'object_point_cloud_mask'
  in_dim: int = 3
  feature_dim: int = 16
  emb_dim: int = 8
  encoder_feature_dim: int = 16
  kernel_size: int = 1
  num_attention_layers: int = 2
  fc_layer_dims: tuple[int, ...] = (8, 8)
  act_dim: int = 7
  attention_type: str = 'regular'
  rpe_masking_type: str = 'nomask'

  @nn.compact
  def __call__(self, ob: dict[str, jnp.ndarray], train: bool = True):
    """Runs PCT policy network and generates actions.

    Args:
      ob: A dictionary of observations. Value with the key point_cloud_obs_name
        should be a point clound of shape (N, in_dim). auxiliary_observations
        keys should be present in the ob dict.
      train: Whether the module is called during training.

    Returns:
      The action array of size act_dim.
    """
    mask = (
        ob[self.point_cloud_mask_name]
        if self.point_cloud_mask_name in ob
        else None
    )
    x = PCTEncoder(
        in_dim=self.in_dim,
        feature_dim=self.feature_dim,
        emb_dim=self.emb_dim,
        encoder_feature_dim=self.encoder_feature_dim,
        kernel_size=self.kernel_size,
        num_attention_layers=self.num_attention_layers,
        attention_type=self.attention_type,
        rpe_masking_type=self.rpe_masking_type,
        )(ob[self.point_cloud_obs_name], mask)
    all_inputs = [x]
    for name in self.auxiliary_observations:
      all_inputs.append(ob[name])
    x = jnp.concatenate(all_inputs, axis=-1)

    for dim in self.fc_layer_dims:
      x = nn.tanh(nn.Dense(dim)(x))
    x = nn.tanh(nn.Dense(self.act_dim)(x))
    return x


class PCTPolicy(jax_policy.JaxPolicy):
  """Policy class that computes action by running PCT jax models."""

  def __init__(
      self, ob_space: gym.Space, ac_space: gym.Space,
      auxiliary_observations: list[str], seed: int = 42
  ) -> None:
    """Initializes a jax policy. See the base class for more details."""
    init_x = ob_space.sample()
    init_x['object_point_cloud'] = init_x['object_point_cloud'].squeeze().T
    super().__init__(
        ob_space=ob_space, ac_space=ac_space,
        model=functools.partial(
            PCTPolicyNet,
            act_dim=utils.flatdim(ac_space),
            auxiliary_observations=auxiliary_observations),
        init_x=init_x, seed=seed)

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action.

    Args:
      ob: The observations in reinforcement learning.

    Returns:
      The action in reinforcement learning.
    """
    ob['object_point_cloud'] = ob['object_point_cloud'].squeeze().T
    action = self.model.apply(
        self._tree_weights,
        ob,
        mutable=['batch_stats'])[0]
    action = utils.unflatten(self._ac_space, action)
    return action
