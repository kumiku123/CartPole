import gymnasium as gym
import numpy as np


env = gym.make("CartPole-v1")

input_dim = 4
hidden_dim = 16
learning_rate = 1e-2
gamma = 0.99
max_episodes = 500

w1 = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
w2 = np.random.randn(hidden_dim) / np.sqrt(hidden_dim)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def policy_forward(x):
    h = np.dot(w1, x)
    h[h < 0] = 0
    logit = np.dot(w2, h)
    p = sigmoid(logit)
    return p, h


def discount_rewards(r):
    discounted = np.zeros_like(r, dtype=np.float64)
    running_add = 0

    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted[t] = running_add

    return discounted


episode_number = 0
running_reward = None

while episode_number < max_episodes:
    obs, info = env.reset()
    xs = []  # 每一步 observation/state
    hs = []  # 保存 hidden layer activation
    dlogps = []  # 当前动作的 policy gradient 方向
    drs = []  # 每一步 reward

    done = False
    reward_sum = 0

    while not done:
        x = obs
        p, h = policy_forward(x)
        action = 1 if np.random.uniform() < p else 0
        y = 1 if action == 1 else 0
        dlogp = y - p

        xs.append(x)
        hs.append(h)
        dlogps.append(dlogp)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward_sum += reward
        drs.append(reward)

    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)

    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr) + 1e-8

    epdlogp *= discounted_epr
    dw2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, w2)
    dh[eph <= 0] = 0
    dw1 = np.dot(dh.T, epx)

    w1 += learning_rate * dw1
    w2 += learning_rate * dw2

    episode_number += 1
    running_reward = (
        reward_sum
        if running_reward is None
        else running_reward * 0.99 + reward_sum * 0.01
    )

    if episode_number % 10 == 0:
        print(
            "Episode {}: reward total was {}, running reward {:.2f}".format(
                episode_number, reward_sum, running_reward
            )
        )

env.close()
