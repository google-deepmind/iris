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

"""Algorithm class for Predictive Information Augmented Random Search."""

from typing import Any, Callable, Dict, Optional, Sequence, Union

from absl import logging
import gym
from gym import spaces
from gym.spaces import utils
from iris import worker_util
from iris.algorithms import ars_algorithm
from iris.policies import keras_pi_policy
import numpy as np
import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import gym_wrapper
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import composite
from tf_agents.utils import eager_utils


class PIARS(ars_algorithm.AugmentedRandomSearch):
  """Augmented random search on predictive representations."""

  def __init__(
      self,
      env: Union[gym.Env, Callable[[], gym.Env]],
      policy: Union[
          keras_pi_policy.KerasPIPolicy,
          Callable[..., keras_pi_policy.KerasPIPolicy],
      ],
      env_args: Optional[Dict[str, Any]] = None,
      policy_args: Optional[Dict[str, Any]] = None,
      learn_representation: bool = True,
      **kwargs
  ) -> None:
    """Initializes the augmented random search algorithm.

    Args:
      env: Gym RL environment object to run rollout with.
      policy: Policy object to map observations to actions.
      env_args: Arguments for env constructor.
      policy_args: Arguments for policy constructor.
      learn_representation: Whether to learn representation.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)
    self._env = env
    self._env_args = env_args
    self._policy = policy
    self._policy_args = policy_args
    self.learn_representation = learn_representation
    self._representation_params = np.empty(0)
    self._representation_learner = RepresentationLearner(
        self._env,
        self._env_args,
        self._policy,
        self._policy_args,
        "dummy_reverb_server_addr",
    )

  def initialize(self, state: Dict[str, Any]) -> None:
    """Initializes the algorithm from initial worker state."""
    super().initialize(state)
    self._representation_params = state["init_representation_params"]

    # Initialize representation learner
    if self.learn_representation:
      self._representation_learner = RepresentationLearner(
          self._env,
          self._env_args,
          self._policy,
          self._policy_args,
          state["reverb_server_addr"],
      )
      self._representation_learner.policy.update_weights(state["init_params"])
      self._representation_learner.policy.update_representation_weights(
          state["init_representation_params"]
      )
      self._representation_learner.policy.reset()

  def process_evaluations(
      self, eval_results: Sequence[worker_util.EvaluationResult]
  ) -> None:
    """Processes the list of Blackbox function evaluations return from workers.

    Gradient is computed by taking a weighted sum of directions and
    difference of their value from the current value. The current parameter
    vector is then updated in the gradient direction with specified step size.

    Args:
      eval_results: List containing Blackbox function evaluations based on the
        order in which the suggestions were sent. ARS performs antithetic
        gradient estimation. The suggestions are sent for evaluation in pairs.
        The eval_results list should contain an even number of entries with the
        first half entries corresponding to evaluation result of positive
        perturbations and the last half corresponding to negative perturbations.
    """
    super().process_evaluations(eval_results)

    # Train representations
    obs_norm_state = None
    if self._obs_norm_data_buffer is not None:
      obs_norm_state = self.state["obs_norm_state"]
    if self.learn_representation:
      self._representation_learner.train(obs_norm_state)

  def get_param_suggestions(
      self, evaluate: bool = False
  ) -> Sequence[Dict[str, Any]]:
    """Suggests a list of inputs to evaluate the Blackbox function on.

    Suggestions are sampled from a gaussian distribution around the current
    parameter vector. For each suggestion, a dict containing keyword arguments
    for the worker is sent.

    Args:
      evaluate: Whether to evaluate current optimization variables for reporting
        training progress.

    Returns:
      A list of suggested inputs for the workers to evaluate.
    """
    suggestions = super().get_param_suggestions(evaluate)

    # Pull representation weights from representation learner
    if self.learn_representation:
      self._representation_params = (
          self._representation_learner.policy.get_representation_weights()
      )
      for suggestion in suggestions:
        suggestion["representation_params"] = self._representation_params

    return suggestions

  def _get_state(self) -> Dict[str, Any]:
    state = super()._get_state()
    if self.learn_representation:
      state["representation_params"] = self._representation_params
    return state

  def _set_state(self, new_state: Dict[str, Any]) -> None:
    super()._set_state(new_state)
    if self.learn_representation:
      self._representation_params = new_state["representation_params"]
    # Set policy weights in representation learner
    self._representation_learner.policy.update_weights(
        new_state["params_to_eval"]
    )
    self._representation_learner.policy.update_representation_weights(
        new_state["representation_params"]
    )
    self._representation_learner.policy.reset()


class RepresentationLearner(object):
  """Representation learner."""

  def __init__(
      self,
      env,
      env_args,
      policy,
      policy_args,
      reverb_server_address,
      rollout_length=5,
      batch_size=512,
      weight_decay=1e-5,
      gamma=0.99,
      num_supports=51,
      min_support=-10.0,
      max_support=10.0,
      use_pi_loss=True,
      use_imitation_loss=False,
      use_value_loss=False,
      learning_rate=1e-4,
      grad_clip=0.5,
      grad_step=2,
  ):
    env_args = {} if env_args is None else env_args
    policy_args = {} if policy_args is None else policy_args
    self._env = env(**env_args) if not isinstance(env, gym.Env) else env

    if not isinstance(policy, keras_pi_policy.KerasPIPolicy):
      self.policy = policy(
          ob_space=self._env.observation_space,
          ac_space=self._env.action_space,
          **policy_args
      )
    else:
      self.policy = policy

    obs_spec = gym_wrapper.spec_from_gym_space(self._env.observation_space)
    action_spec = gym_wrapper.spec_from_gym_space(self._env.action_space)
    time_step_spec = ts.time_step_spec(observation_spec=obs_spec)
    policy_step_spec = policy_step.PolicyStep(action=action_spec)
    collect_data_spec = trajectory.from_transition(
        time_step_spec, policy_step_spec, time_step_spec
    )
    collect_data_spec = tensor_spec.from_spec(collect_data_spec)
    self.reverb_rb = reverb_replay_buffer.ReverbReplayBuffer(
        collect_data_spec,
        sequence_length=(rollout_length + 1),
        table_name="uniform_table",
        server_address=reverb_server_address,
    )
    self._dataset = self.reverb_rb.as_dataset(
        sample_batch_size=batch_size, num_steps=(rollout_length + 1)
    )
    self._data_iter = iter(self._dataset)
    # For distributional value function
    self._num_supports = num_supports
    self._min_support = min_support
    self._max_support = max_support
    self.supports = tf.linspace(min_support, max_support, num_supports)

    self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self._rollout_length = rollout_length
    self._gamma = gamma
    self._weight_decay = weight_decay
    self.use_pi_loss = use_pi_loss
    self.use_imitation_loss = use_imitation_loss
    self.use_value_loss = use_value_loss
    self.grad_clip = grad_clip
    self.grad_step = grad_step
    # TODO: Checkpoint globel step
    self.global_step = 0
    self.reverb_checkpoint_period = 20

  def train(self, obs_norm_state=None):
    """Train representation from replay data.

    Args:
      obs_norm_state: Observation normalizer state (mean and std).
    """

    for _ in range(self.grad_step):
      traj, _ = next(self._data_iter)
      discount = traj.discount
      reward = tf.nest.map_structure(
          lambda t: composite.slice_to(t, axis=1, end=-1), traj.reward
      )
      action = tf.nest.map_structure(
          lambda t: composite.slice_to(t, axis=1, end=-1), traj.action
      )
      discount = tf.nest.map_structure(
          lambda t: composite.slice_to(t, axis=1, end=-1), discount
      )

      obs = traj.observation
      # observation normalization
      if obs_norm_state is not None:
        obs_mean = obs_norm_state["mean"]
        obs_mean = utils.unflatten(self._env.observation_space, obs_mean)
        obs_std = obs_norm_state["std"]
        obs_std = utils.unflatten(self._env.observation_space, obs_std)
        obs = tf.nest.map_structure(lambda x, y: x - y, obs, obs_mean)
        obs = tf.nest.map_structure(lambda x, y: x / (y + 1e-8), obs, obs_std)

      # Separate vision input and other observations.
      obs_flat = []
      for image_label in self.policy._image_input_labels:  # pylint: disable=protected-access
        vision_input = obs[image_label]
        obs_flat.append(vision_input)

      other_ob = obs.copy()
      for image_label in self.policy._image_input_labels:  # pylint: disable=protected-access
        del other_ob[image_label]

      # Flatten other observations.
      other_input = flatten_nested(self.policy._other_ob_space, other_ob)  # pylint: disable=protected-access
      obs_flat.append(other_input)

      loss, _ = self.train_single_step(obs_flat, reward, action, discount)
    self.global_step += 1
    if self.global_step % self.reverb_checkpoint_period == 0:
      logging.info("Start checkpointing reverb data.")
      self.reverb_rb.py_client.checkpoint()
    print("train/loss: {}".format(np.mean(loss.numpy())))

  @tf.function
  def train_single_step(self, obs, reward, action, discount):
    """One gradient step."""
    trainable_variables = self.policy.h_model.trainable_weights
    trainable_variables += self.policy.f_model.trainable_weights
    trainable_variables += self.policy.g_model.trainable_weights
    if self.use_pi_loss:
      trainable_variables += self.policy.px_model.trainable_weights
      trainable_variables += self.policy.py_model.trainable_weights
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(trainable_variables)
      loss, metrics = self.loss(obs, reward, action, discount)
      loss_reg = (
          tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables])
          * self._weight_decay
      )
      loss += loss_reg
    tf.debugging.check_numerics(loss, "loss is inf or nan.")
    grads = tape.gradient(loss, trainable_variables)
    grads_and_vars = list(zip(grads, trainable_variables))
    if self.grad_clip is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(
          grads_and_vars, self.grad_clip
      )
    self._optimizer.apply_gradients(grads_and_vars)
    return loss, metrics

  @tf.function
  def rollout(self, obs, actions):
    """Latent rollout."""
    s, _ = self.policy.h_model(obs)
    outputs = []
    for i in range(self._rollout_length):
      p, v = self.policy.f_model(s)
      u_next, s_next = self.policy.g_model([s, actions[:, i, ...]])
      outputs.append((p, v, u_next, s))
      s = s_next
    p, v = self.policy.f_model(s)
    outputs.append((p, v, None, s))
    return outputs

  @tf.function
  def loss(self, obs, rewards, actions, discount):
    """Representation and dynamics loss."""
    # Ex. obs: [(B, T, 24, 32, 1), (B, T, 24, 32, 1), (B, T, 68)]
    #     obs0: [(B, 24, 32, 1), (B, 24, 32, 1), (B, 68)]
    obs0 = tf.nest.map_structure(
        lambda t: composite.slice_to(t, axis=1, end=1), obs
    )
    obs0 = [tf.squeeze(x, axis=1) for x in obs0]
    latent_traj = self.rollout(obs0, actions)

    loss_pi = 0.0  # Predictive Information Loss
    loss_p = 0.0  # Imitation Loss
    loss_v = 0.0  # Value loss
    loss_r = 0.0  # Reward loss

    def infonce(hidden_x, hidden_y, temperature=0.1):
      hidden_x = tf.math.l2_normalize(hidden_x, -1)
      hidden_y = tf.math.l2_normalize(hidden_y, -1)
      batch_size = tf.shape(hidden_x)[0]
      labels = tf.one_hot(tf.range(batch_size), batch_size)
      logits = tf.matmul(hidden_x, hidden_y, transpose_b=True) / temperature
      hyz = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
      iyz = tf.math.log(tf.cast(batch_size, tf.float32)) - hyz
      return iyz, logits, labels

    if self.use_pi_loss:
      k = self._rollout_length
      obs_k = [x[:, k, ...] for x in obs]
      # Latent state (from visual + other observations) for the first time step
      hx = latent_traj[0][-1]
      # Latent state (from visual observations) for the last time step
      _, hy_vision = self.policy.h_model(obs_k)
      # A trick from https://arxiv.org/abs/2011.10566
      hy_vision = tf.stop_gradient(hy_vision)
      zx = self.policy.px_model(hx)
      zy = self.policy.py_model(hy_vision)
      iyz, _, _ = infonce(zx, zy, temperature=0.1)
      loss_pi = -iyz

    # Compute target values
    if self.use_value_loss:
      # Value distribution for the last time step
      last_value_distribution = latent_traj[-1][1]
      target_value_supports = [self.supports]  # not used in loss
      for i in range(self._rollout_length - 2, -1, -1):
        r_next = rewards[:, i : i + 1, ...]
        d_next = discount[:, i : i + 1, ...]
        target_support = (
            target_value_supports[-1] * d_next * self._gamma + r_next
        )
        target_value_supports.append(target_support)
      target_value_supports.reverse()

      vd = tf.nn.softmax(latent_traj[-1][1])
      pred_value_sum = tf.reduce_sum(vd * self.supports[None, ...], axis=-1)

    for i in range(self._rollout_length - 1):
      p = latent_traj[i][0]
      z = latent_traj[i][1]
      u_next = latent_traj[i][2]

      if self.use_imitation_loss:
        loss_p += tf.reduce_sum(
            tf.math.square(p - tf.stop_gradient(actions[:, i, ...])), -1
        )
      if self.use_value_loss:
        loss_v += self.distributional_value_loss(
            value_logits=z,
            value_supports=self.supports,
            target_value_logits=last_value_distribution,
            target_value_supports=target_value_supports[i],
        )
        vd = tf.nn.softmax(z)
        pred_value_sum += tf.reduce_sum(vd * self.supports[None, ...], axis=-1)
      # reward loss
      loss_r += tf.reduce_sum(
          tf.math.square(u_next - tf.stop_gradient(rewards[:, i : i + 1])), -1
      )
    loss = loss_r + loss_v + loss_p + loss_pi
    loss = tf.reduce_mean(loss)
    metrics = {
        "loss_r": tf.reduce_mean(loss_r),
        "loss_v": tf.reduce_mean(loss_v),
        "loss_p": tf.reduce_mean(loss_p),
        "loss_pi": tf.reduce_mean(loss_pi),
    }
    if self.use_value_loss:
      metrics["value"] = tf.reduce_mean(pred_value_sum) / self._rollout_length
    if self.use_pi_loss:
      metrics["iyz"] = tf.reduce_mean(iyz)
    return loss, metrics

  def distributional_value_loss(
      self,
      value_logits,
      value_supports,
      target_value_logits,
      target_value_supports,
  ):
    """Computes the distributional value loss."""
    target_value_probs = tf.nn.softmax(target_value_logits)
    projected_target_probs = tf.stop_gradient(
        categorical_dqn_agent.project_distribution(
            target_value_supports, target_value_probs, value_supports
        )
    )

    value_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=value_logits, labels=projected_target_probs
    )

    return value_loss


def flatten_nested(space, x):
  """Flatten nested."""
  if isinstance(space, spaces.Box):
    x = np.asarray(x, dtype=np.float32)
    inner_dims = list(space.shape)
    outer_dims = list(x.shape)[: -len(inner_dims)]
    x = np.reshape(x, outer_dims + [np.prod(inner_dims)])
    return x
  elif isinstance(space, spaces.Tuple):
    return np.concatenate(
        [flatten_nested(s, x_part) for x_part, s in zip(x, space.spaces)],
        axis=-1,
    )
  elif isinstance(space, spaces.Dict):
    return np.concatenate(
        [flatten_nested(space.spaces[key], item) for key, item in x.items()],
        axis=-1,
    )
  elif isinstance(space, spaces.MultiBinary):
    x = np.asarray(x)
    space = np.asarray(space)
    inner_dims = list(space.shape)
    outer_dims = list(x.shape)[: -len(inner_dims)]
    x = np.reshape(x, outer_dims + [np.prod(inner_dims)])
    return x
  elif isinstance(space, spaces.MultiDiscrete):
    x = np.asarray(x)
    space = np.asarray(space)
    inner_dims = list(space.shape)
    outer_dims = list(x.shape)[: -len(inner_dims)]
    x = np.reshape(x, outer_dims + [np.prod(inner_dims)])
    return x
  else:
    raise NotImplementedError
