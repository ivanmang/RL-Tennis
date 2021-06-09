# Project 3 Report

## Introduction
In this project, an agent is trained to move a double-joined arm to target locations under the  [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.
It is trained using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm. The environment contains 2 identical agents, each have its own independent observations and actions.

## Learning Algorithm
Multi-Agent Deep Deterministic Policy Gradient ([MADDPG](https://arxiv.org/pdf/1706.02275.pdf)) algorithm is a algorithm which contains mutiple [DDPG](https://arxiv.org/abs/1509.02971) agents.
