#!/usr/bin/env python3

import tensorflow as tf
from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
import numpy as np

def train(env_id, num_timesteps, seed, save, gamma, lam,
          desired_kl):
    env = make_mujoco_env(env_id, seed)

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        ret = learn(env, policy=policy, vf=vf,
                    gamma=gamma, lam=lam,
                    desired_kl=desired_kl,
                    timesteps_per_batch=2500,
                    num_timesteps=num_timesteps, animate=False)

        env.close()
        np.savetxt(save, np.array([ret]))

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, gamma=args.gamma, lam=args.lam,
          save=args.save, desired_kl=args.desired_kl,
          num_timesteps=args.num_timesteps,
          seed=args.seed)

if __name__ == "__main__":
    main()
