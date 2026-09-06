import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# =========================
# Environment
# =========================
env = gym.make("CartPole-v1")

input_dim = 4
hidden_dim = 16
learning_rate = 1e-2
gamma = 0.99
max_episodes = 500


# =========================
# Policy parameters
# =========================
w1 = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
w2 = np.random.randn(hidden_dim) / np.sqrt(hidden_dim)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def policy_forward(x):
    h = np.dot(w1, x)

    # ReLU
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


# =========================
# Experiment records
# =========================

results = []


# =========================
# Training
# =========================

episode_number = 0
running_reward = None

while episode_number < max_episodes:

    obs, info = env.reset()

    xs = []
    hs = []
    dlogps = []
    drs = []

    actions = []
    states = []

    done = False
    reward_sum = 0

    while not done:

        x = obs

        p, h = policy_forward(x)

        # Sample action
        action = 1 if np.random.uniform() < p else 0

        y = 1 if action == 1 else 0
        dlogp = y - p

        # Save training data
        xs.append(x)
        hs.append(h)
        dlogps.append(dlogp)
        drs.append(1.0)

        # Save analysis data
        actions.append(action)
        states.append(x.copy())

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        reward_sum += reward


    # =========================
    # Policy gradient update
    # =========================

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


    # =========================
    # Episode-level analysis
    # =========================

    states = np.array(states)
    actions = np.array(actions)

    episode_length = len(actions)

    # Action switching
    if episode_length > 1:
        switches = np.sum(actions[1:] != actions[:-1])
        switch_rate = switches / (episode_length - 1)
    else:
        switch_rate = 0.0

    # State variables
    max_cart_velocity = np.max(np.abs(states[:, 1]))
    max_pole_angle = np.max(np.abs(states[:, 2]))
    max_angular_velocity = np.max(np.abs(states[:, 3]))

    mean_pole_angle = np.mean(np.abs(states[:, 2]))
    mean_angular_velocity = np.mean(np.abs(states[:, 3]))


    # =========================
    # Running reward
    # =========================

    running_reward = (
        reward_sum
        if running_reward is None
        else running_reward * 0.99 + reward_sum * 0.01
    )


    # =========================
    # Save episode results
    # =========================

    results.append({
        "episode": episode_number + 1,
        "reward": reward_sum,
        "episode_length": episode_length,
        "running_reward": running_reward,
        "switch_rate": switch_rate,
        "max_cart_velocity": max_cart_velocity,
        "max_pole_angle": max_pole_angle,
        "max_angular_velocity": max_angular_velocity,
        "mean_pole_angle": mean_pole_angle,
        "mean_angular_velocity": mean_angular_velocity
    })


    episode_number += 1


    # =========================
    # Print every episode
    # =========================

    print(
        f"Episode {episode_number:3d} | "
        f"Reward {reward_sum:6.1f} | "
        f"Length {episode_length:3d} | "
        f"Switch {switch_rate:.3f} | "
        f"Max Angle {max_pole_angle:.3f} | "
        f"Max Angular Vel {max_angular_velocity:.3f}"
    )


env.close()


# =========================
# Save results
# =========================

df = pd.DataFrame(results)

df.to_csv("baseline_results.csv", index=False)

print("\nResults saved to baseline_results.csv")


# =========================
# Correlation analysis
# =========================

metrics = [
    "switch_rate",
    "max_pole_angle",
    "max_angular_velocity",
    "max_cart_velocity",
    "mean_pole_angle",
    "mean_angular_velocity"
]

print("\n=== Pearson Correlation with Reward ===")

for metric in metrics:
    corr = df["reward"].corr(df[metric], method="pearson")
    print(f"{metric:25s}: {corr:.4f}")


print("\n=== Spearman Correlation with Reward ===")

for metric in metrics:
    corr = df["reward"].corr(df[metric], method="spearman")
    print(f"{metric:25s}: {corr:.4f}")


# =========================
# Plot 1: Reward
# =========================

plt.figure()
plt.plot(df["episode"], df["reward"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward")
plt.show()


# =========================
# Plot 2: Running reward
# =========================

plt.figure()
plt.plot(df["episode"], df["running_reward"])
plt.xlabel("Episode")
plt.ylabel("Running Reward")
plt.title("Running Reward")
plt.show()


# =========================
# Plot 3: Switch rate
# =========================

plt.figure()
plt.plot(df["episode"], df["switch_rate"])
plt.xlabel("Episode")
plt.ylabel("Action Switch Rate")
plt.title("Action Switching Rate")
plt.show()


# =========================
# Plot 4: Maximum pole angle
# =========================

plt.figure()
plt.plot(df["episode"], df["max_pole_angle"])
plt.xlabel("Episode")
plt.ylabel("Maximum |Pole Angle|")
plt.title("Maximum Pole Angle")
plt.show()


# =========================
# Plot 5: Maximum angular velocity
# =========================

plt.figure()
plt.plot(df["episode"], df["max_angular_velocity"])
plt.xlabel("Episode")
plt.ylabel("Maximum |Angular Velocity|")
plt.title("Maximum Angular Velocity")
plt.show()