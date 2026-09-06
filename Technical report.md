# Safe Reinforcement Learning under Environmental Perturbations
## A Preliminary Investigation with CartPole

**Author:** Tang Ruijie  
**Project Type:** Independent Research Project  
**Research Area:** Reinforcement Learning / AI Safety  
**Status:** Preliminary Study  
**Repository:** https://github.com/kumiku123/CartPole

---

## Abstract

This project investigates how safety considerations can be incorporated into reinforcement learning (RL) agents while preserving or improving task performance. The project began with a basic policy-gradient implementation on the CartPole-v1 environment, where the original objective was primarily to maximize cumulative reward.

Rather than immediately designing a Safe RL algorithm, this study first investigates which observable aspects of an agent's behavior and state dynamics are associated with task performance and may provide useful signals for defining safety constraints.

The initial analysis considers action-switching frequency, pole angle, angular velocity, and cart velocity. Correlation analysis is used as an exploratory tool to examine relationships between these variables and episode reward. The analysis revealed that some intuitive assumptions about safe control were not supported by the baseline data. In particular, action-switching frequency did not decrease as performance improved, whereas angular-velocity measures showed stronger negative associations with reward.

These observations motivate a second stage of the project: identifying meaningful safety boundaries through failure analysis, then incorporating these signals into Safe RL objectives or constraints and evaluating the resulting agent under environmental perturbations.

---

# 1. Research Motivation

## 1.1 From Reinforcement Learning to Safe Reinforcement Learning

The initial project was a basic reinforcement learning implementation of CartPole-v1.

The original agent was trained to maximize cumulative reward:

$$
\max_{\pi} \mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T} \gamma^t R_t
\right]
$$

This objective is sufficient for demonstrating how an RL agent can learn a control policy, but it does not explicitly distinguish between:

- effective behavior,
- stable behavior,
- potentially unsafe behavior.

As RL agents become more capable and are deployed in more complex environments, optimizing task performance alone may not be sufficient. An agent may achieve a high task reward while using undesirable strategies, approaching physical limits, or reacting poorly to unexpected environmental changes.

This motivates the central research question of this project:

> **How can an RL agent improve its performance while remaining safe and stable when the environment changes?**

---

## 1.2 Why Study Safety Before Designing the Safety Algorithm?

A natural first attempt would be to add a safety penalty directly to the reward:

$$
R'_t = R_t - \lambda C_t
$$

However, this immediately raises a more fundamental question:

> **What should the safety cost $C_t$ actually measure?**

It is tempting to define safety using intuitive variables such as action frequency or pole angle. However, such assumptions may not correspond to actual failure or instability.

Therefore, this project takes a measurement-first approach:

$$
\text{Baseline RL}
\rightarrow
\text{Behavioral Analysis}
\rightarrow
\text{Candidate Safety Signals}
\rightarrow
\text{Safety Boundary}
\rightarrow
\text{Safe RL}
$$

The first objective is therefore not to improve the agent, but to understand its behavior sufficiently well to define what "safe" should mean in this environment.

---

# 2. Baseline Environment

## 2.1 CartPole-v1

The environment used in this study is `CartPole-v1` from Gymnasium.

The agent receives a four-dimensional observation:

$$
s_t =
[
x_t,\dot{x}_t,\theta_t,\dot{\theta}_t
]
$$

where:

- $x_t$: cart position
- $\dot{x}_t$: cart velocity
- $\theta_t$: pole angle
- $\dot{\theta}_t$: pole angular velocity

The agent chooses between two discrete actions:

$$
a_t \in \{0,1\}
$$

representing forces applied in opposite horizontal directions.

The episode reward is approximately proportional to the number of timesteps for which the pole remains balanced.

Therefore:

$$
R_{\text{episode}} \approx T
$$

where $T$ is the episode duration.

---

# 3. Baseline Agent

## 3.1 Policy Architecture

The baseline agent uses a small neural policy consisting of:

- input dimension: 4
- hidden dimension: 16
- one ReLU hidden layer
- sigmoid output
- stochastic binary action sampling

The forward pass is:

$$
h = \mathrm{ReLU}(W_1s_t)
$$

followed by:

$$
z = W_2h
$$

and:

$$
p_t = \sigma(z)
$$

where:

$$
p_t = P(a_t=1|s_t)
$$

The action is sampled according to:

$$
a_t =
\begin{cases}
1 & \text{if } u < p_t\\
0 & \text{otherwise}
\end{cases}
$$

where:

$$
u\sim U(0,1)
$$

---

## 3.2 Policy Gradient

The implementation uses a policy-gradient approach.

For each episode, discounted returns are calculated as:

$$
G_t =
r_t + \gamma r_{t+1}
+\gamma^2r_{t+2}+\cdots
$$

with:

$$
\gamma=0.99
$$

The discounted returns are normalized before being applied to the policy gradient update.

The model parameters are then updated using:

$$
W \leftarrow W+\alpha\nabla_\theta J(\theta)
$$

with:

$$
\alpha=10^{-2}
$$

The baseline therefore represents a simple performance-oriented RL system without explicit safety constraints.

---

# 4. Initial Safety Hypotheses

Before implementing a safety mechanism, several candidate safety-related variables were identified.

## Hypothesis H1 — Action Switching

Frequent changes between left and right actions might indicate unstable control.

A simple action-switching metric is:

$$
SwitchRate =
\frac{
\sum_{t=2}^{T}
\mathbf{1}(a_t\neq a_{t-1})
}{
T-1
}
$$

Initial intuition:

> Higher switching frequency may indicate more aggressive or unstable control.

---

## Hypothesis H2 — Pole Angle

The pole angle is directly related to the failure condition of CartPole.

A candidate metric is:

$$
\max_t|\theta_t|
$$

Initial intuition:

> Smaller maximum pole angle may indicate more stable control.

---

## Hypothesis H3 — Angular Velocity

The angular velocity describes how rapidly the pole's orientation changes.

Candidate metrics include:

$$
\max_t|\dot{\theta}_t|
$$

and:

$$
\frac{1}{T}
\sum_{t=1}^{T}
|\dot{\theta}_t|
$$

Initial intuition:

> Larger angular velocity may indicate more aggressive or unstable system dynamics.

---

# 5. Experiment 1 — Initial Small-Scale Analysis

## 5.1 Objective

The first experiment investigated whether candidate behavioral variables were associated with task performance.

The initial implementation recorded:

- episode reward
- episode length
- action-switching rate
- maximum pole angle

The results were initially printed every ten episodes.

Therefore, although training covered 500 episodes, the first exploratory analysis only contained approximately 50 recorded observations.

---

## 5.2 Data Analysis

Two types of correlation were considered:

### Pearson Correlation

Pearson correlation was used to measure linear association:

$$
r =
\frac{\mathrm{Cov}(X,Y)}
{\sigma_X\sigma_Y}
$$

### Spearman Correlation

Spearman correlation was also calculated to detect monotonic relationships without assuming a linear relationship.

The primary comparison was:

$$
Reward
\leftrightarrow
Candidate\ Safety\ Variable
$$

---

## 5.3 Initial Visualization

### Figure 1 — Episode Reward

> ![[Figure_1.png]]



---

### Figure 2 — Running Reward

![[Figure_2.png]]

---

### Figure 3 — Action Switching Rate

![[Figure_3.png]]

---

### Figure 4 — Maximum Pole Angle

![[Figure_4.png]]

---

## 5.4 Initial Findings

The first exploratory analysis suggested that task performance increased over training, but the relationship between reward and candidate safety indicators was not as straightforward as initially expected.

This motivated a more detailed experiment rather than immediately introducing a safety penalty.

---

# 6. Experiment 2 — Full Episode-Level Data Collection

## 6.1 Motivation

The initial logging strategy recorded results only every tenth episode.

This created a measurement limitation:

$$
500\ episodes
\rightarrow
50\ observations
$$

The limited temporal resolution made it difficult to distinguish systematic trends from noise and prevented more detailed statistical analysis.

The experiment was therefore redesigned to record every episode.

---

## 6.2 Additional Variables

The revised experiment recorded all 500 episodes and added:

- maximum cart velocity
- maximum pole angle
- maximum angular velocity
- mean pole angle
- mean angular velocity
- action-switching rate
- episode length
- episode reward

The resulting dataset was saved as:

![[baseline_results 1.csv]]

# 7. Results of the Baseline Analysis

## 7.1 Reward and Action Switching

The baseline data showed a positive association between reward and action-switching frequency.

The Pearson correlation was approximately:

$$
r \approx 0.36
$$

and the Spearman correlation was approximately:

$$
\rho \approx 0.44
$$

This was contrary to the initial hypothesis that high-performing control would necessarily use fewer action changes.

The observation therefore suggests:

> **Action-switching frequency alone is not a sufficient proxy for safety or stability in this environment.**

A high-performing policy may still use relatively frequent switching as part of its control strategy.

---

## 7.2 Reward and Maximum Pole Angle

The relationship between reward and maximum pole angle was not sufficiently consistent to support the assumption that lower maximum pole angle directly implies higher performance.

The Pearson correlation was approximately:

$$
r \approx -0.41
$$

while the Spearman correlation was close to zero:

$$
\rho \approx -0.03
$$

This difference also demonstrates why relying on a single correlation statistic can be misleading.

The result suggests:

> **Maximum pole angle alone should not be treated as a sufficient safety metric.**

---

## 7.3 Reward and Angular Velocity

Angular velocity produced a more informative pattern.

The Spearman correlation between reward and maximum angular velocity was approximately:

$$
\rho \approx -0.43
$$

More notably, the relationship between reward and mean angular velocity was approximately:

$$
\rho \approx -0.72
$$

The training data also showed a substantial reduction in mean angular velocity as average performance improved.

This suggests that, in this baseline experiment:

$$
Performance\uparrow
\quad\text{was associated with}\quad
Mean\ Angular\ Velocity\downarrow
$$

This observation shifted the investigation from simple action-level metrics toward **state-dynamics and system-stability metrics**.

---

# 8. What Was Surprising?

The most significant surprise was that action-switching frequency behaved differently from the original intuition.

The initial expectation was:

$$
SwitchRate\downarrow
\Rightarrow
Control\ Stability\uparrow
\Rightarrow
Performance\uparrow
$$

However, the data instead showed:

$$
Performance\uparrow
\quad\text{while}\quad
SwitchRate\uparrow
$$

This directly challenged the initial assumption that less frequent action changes necessarily indicate safer or more stable control.

The second important observation was that maximum pole angle also did not show a clear monotonic relationship with reward.

In contrast, angular-velocity measures showed a stronger negative association with performance, especially mean angular velocity.

These results motivated a shift in perspective:

> Rather than defining safety using a single intuitive action-level metric, the analysis should investigate the underlying dynamics and stability of the controlled system.

This represented an important change in the research direction from:

$$
Action\ behavior
$$

toward:

$$
State\ dynamics
$$

---

# 9. Methodological Reflection

## 9.1 Initial Logging Strategy Was Too Coarse

The initial experiment only recorded results every tenth episode.

Although training covered 500 episodes, this produced only approximately 50 recorded observations for analysis.

This was suboptimal because it:

- reduced the resolution of the learning trajectory;
- made it harder to distinguish systematic trends from noise;
- limited statistical analysis;
- made unusual episodes more difficult to identify.

The experiment was therefore revised to record every episode.

---

## 9.2 Safety Proxies Were Initially Based Too Strongly on Intuition

The initial hypothesis that frequent action switching might indicate unsafe or unstable behavior was based primarily on intuition rather than empirical evidence.

The baseline results did not support this simple relationship.

In retrospect, this demonstrates an important methodological principle:

> **Candidate safety indicators should be empirically evaluated before being incorporated into the objective or constraints of a Safe RL algorithm.**

Rather than assuming that a particular observable variable represents safety, the project should first determine whether that variable is actually associated with instability, constraint violations, or failure.

---

## 9.3 Correlation with Reward Is Not Equivalent to Safety

The second methodological limitation is that correlation with reward does not directly establish a safety relationship.

For a candidate variable $X$:

$$
Correlation(X, Reward)
$$

only measures the relationship between $X$ and task performance.

It does not establish:

$$
X \equiv Safety
$$

A variable may correlate strongly with reward while having little relationship with failure probability.

Therefore, the next stage should investigate:

$$
P(Failure\mid X)
$$

and, where appropriate:

$$
P(Safety\ Violation\mid X)
$$

rather than relying solely on correlation with reward.

---

# 10. Current Research Interpretation

The preliminary results suggest that safety in CartPole may be better represented by **system dynamics and stability** than by a single action-level statistic.

A candidate conceptual hierarchy is:

$$
Action
\rightarrow
State\ Transition
\rightarrow
System\ Dynamics
\rightarrow
Stability
\rightarrow
Failure\ Risk
$$

This leads to a refined research question:

> **Which observable properties of an RL agent's control behavior and state dynamics meaningfully predict unsafe or unstable operation?**

This question will guide the next phase of the project.

---

# 11. Unresolved Questions

The main unresolved question is:

> **What constitutes a meaningful safety boundary for an RL control system?**

It remains unclear whether:

- angular velocity,
- pole angle,
- cart velocity,
- action dynamics,

or a combination of these variables best predicts failure or unsafe behavior.

The current analysis has identified candidate variables, but it has not yet established a causal or predictive relationship between these variables and failure.

The next step is therefore to determine whether candidate safety variables can identify regions of the state-action space associated with increased failure risk.

---

# 12. Planned Safe RL Extension

Once meaningful safety-related variables have been identified, they can be incorporated into the RL objective.

One possible formulation is:

$$
R'_t =
R_t-\lambda C_t
$$

where $C_t$ represents a safety cost.

A candidate safety cost may take the form:

$$
C_t =
\alpha C_{\theta}
+
\beta C_{\dot{\theta}}
+
\gamma C_{action}
$$

where:

- $C_{\theta}$ represents a pole-angle-related safety cost;
- $C_{\dot{\theta}}$ represents an angular-velocity-related safety cost;
- $C_{action}$ represents a control-related cost.

The exact form and weighting should not be chosen arbitrarily. They should be informed by the failure analysis conducted on the baseline environment.

An alternative formulation is constrained reinforcement learning:

$$
\max_\pi
\mathbb E_\pi
\left[
\sum_t\gamma^tR_t
\right]
$$

subject to:

$$
\mathbb E_\pi
\left[
\sum_t\gamma^tC_t
\right]
\leq d
$$

This formulation makes it possible to investigate whether an agent can preserve safety constraints while maintaining or improving task performance.

---

# 13. Environmental Perturbations

After establishing a baseline safety mechanism, the environment will be made progressively more challenging.

The purpose is to test whether the proposed safety mechanism remains effective when the environment differs from the conditions under which the original policy was learned.

## 13.1 Wind Disturbances

A disturbance force can be introduced:

$$
F_{wind}(t)
$$

Possible variants include:

- constant wind;
- random wind;
- time-varying wind.

These disturbances can be used to test whether the agent can detect and respond to unexpected external forces while remaining within its safety boundaries.

---

## 13.2 Changing Dynamics

Selected environment parameters can be varied during training or evaluation.

For example:

$$
g_t \neq constant
$$

The purpose is to introduce a degree of distribution shift:

$$
P_{train}(E)\neq P_{test}(E)
$$

and examine whether the safety mechanism continues to provide useful protection.

---

## 13.3 Uneven or Changing Terrain

A more ambitious extension may modify the environment dynamics to represent uneven surfaces or changing slopes.

This is motivated by potential future applications to physical control systems, where changing terrain, external disturbances, and actuator limitations can create real safety risks.

However, CartPole remains a highly simplified simulation. Results obtained in this environment should therefore be interpreted as evidence about control principles rather than direct evidence of real-world robotic or vehicle safety.

---

# 14. Future Research: Continual Learning

After establishing the Safe RL baseline, a further extension will investigate continual learning.

Suppose the environment changes over time:

$$
E_1\rightarrow E_2\rightarrow E_3
$$

The agent correspondingly updates its policy:

$$
\pi_1\rightarrow\pi_2\rightarrow\pi_3
$$

The central question becomes:

> **Can the agent continuously improve its capabilities while preserving previously established safety constraints?**

This introduces a potential trade-off between:

$$
Capability\ Improvement
\quad vs \quad
Safety\ Retention
$$

A particularly important failure mode is catastrophic forgetting of previously learned safety behavior.

The agent may become better adapted to a new environment while simultaneously becoming less compliant with constraints learned earlier.

This creates a natural connection between continual learning and alignment:

> increasing capability should not require sacrificing previously established safety properties.

---
