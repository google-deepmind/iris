
<img align="left" width="100" height="100" src="assets/iris-logo.svg" alt="Iris Logo">

# Iris: Synchronous and Distributed Blackbox Optimization at Scale
[![Continuous Integration](https://github.com/google-deepmind/iris/actions/workflows/core_test.yml/badge.svg)](https://github.com/google-deepmind/iris/actions?query=branch%3Amain)

## Overview
Iris is a library for performing synchronous and distributed zeroth-order
optimization at scale. It is meant primarily to train large neural networks with
evolutionary methods, but can be applied to optimize any high dimensional
blackbox function.

## Installation

```bash

pip install google-iris==0.0.2a0
```

## Getting Started

To launch a local optimization, run:

```bash

python3 -m iris.launch \
--lp_launch_type=local_mp \
--experiment_name=iris_example \
--config=configs/simple_example_config.py \
--logdir=/tmp/bblog \
--num_workers=16 \
--num_eval_workers=10 \
--alsologtostderr
```

## Associated Publications

* [Achieving Human Level Competitive Robot Table Tennis](https://arxiv.org/abs/2408.03906) (ICRA 2025 - Best Paper Award Finalist)
* [SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust Attention](https://arxiv.org/abs/2312.01990) (ICRA 2024 - Best Robotic Manipulation Award)
* [Embodied AI with Two Arms: Zero-shot Learning, Safety and Modularity](https://arxiv.org/abs/2404.03570) (IROS 2024 - Robocup Best Paper Award)
* [Agile Catching with Whole-Body MPC and Blackbox Policy Learning](https://arxiv.org/abs/2306.08205) (L4DC 2023)
* [Discovering Adaptable Symbolic Algorithms from Scratch](https://arxiv.org/abs/2307.16890) (IROS 2023, Best Paper Finalist)
* [Visual-Locomotion: Learning to Walk on Complex Terrains with Vision](https://proceedings.mlr.press/v164/yu22a.html) (CoRL 2022)
* [ES-ENAS: Efficient Evolutionary Optimization for Large Hybrid Search Spaces](https://arxiv.org/abs/2101.07415) (arXiv, 2021)
* [Hierarchical Reinforcement Learning for Quadruped Locomotion](https://arxiv.org/abs/1905.08926) (RSS 2021)
* [Rapidly Adaptable Legged Robots via Evolutionary Meta-Learning](https://arxiv.org/abs/2003.01239) (IROS 2020)
* [Robotic Table Tennis with Model-Free Reinforcement Learning](https://arxiv.org/abs/2003.14398) (IROS 2020)
* [ES-MAML: Simple Hessian-Free Meta Learning](https://arxiv.org/abs/1910.01215) (ICLR 2020)
* [Provably Robust Blackbox Optimization for Reinforcement Learning](https://arxiv.org/abs/1903.02993) (CoRL 2019)
* [Structured Evolution with Compact Architectures for Scalable Policy Optimization](https://arxiv.org/abs/1804.02395) (ICML 2018)
* [Optimizing Simulations with Noise-Tolerant Structured Exploration](https://arxiv.org/abs/1805.07831) (ICRA 2018)
* [On Blackbox Backpropagation and Jacobian Sensing](https://proceedings.neurips.cc/paper_files/paper/2017/file/9c8661befae6dbcd08304dbf4dcaf0db-Paper.pdf) (NeurIPS 2017)

**Disclaimer:** This is not an officially supported Google product.