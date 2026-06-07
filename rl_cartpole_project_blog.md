# 从 CartPole 开始理解强化学习：一个 AI Agent 项目的学习日志

> Status: In Progress 
> Project type: Reinforcement Learning learning log  
> Current stage: First NumPy Policy Gradient implementation completed and tested on CartPole

## 为什么从 CartPole 开始

我最近开始了解/学习强化学习（Reinforcement Learning, RL），并计划通过一个 GitHub Blog 记录自己的学习过程、实验计划和思考。第一阶段我选择从 `CartPole` 开始，而不是一开始就进入更复杂的游戏或机器人任务。

原因很简单：CartPole 是一个足够经典、足够小、同时又能体现强化学习核心思想的环境。在这个任务里，一个 `agent` 需要通过不断与 `environment` 交互，学习如何选择动作，让小车上的杆尽可能长时间保持平衡。这个过程可以帮助我理解强化学习中几个最基础的概念：

- `agent`：做决策的智能体。
- `environment`：智能体所处的环境。
- `state`：环境当前的状态信息。
- `action`：智能体可以采取的动作。
- `reward`：环境对动作结果的反馈。
- `policy`：智能体根据状态选择动作的策略。

对我来说，CartPole 不只是一个入门 toy problem。它提供了一个清晰的起点，让我能够从一个具体环境出发，逐步理解 AI agent 如何通过反馈学习，而不是只停留在教材或视频里的抽象定义。

## 项目目标

目前这个项目还处于早期阶段，但我已经完成了第一版基础实验：使用 NumPy 手写了一个简单的 `Policy Gradient` agent，并在 `CartPole-v1` 环境中跑通训练流程。因此，这篇文章既是 learning log，也是我对第一个 RL 实验模型的技术记录。

这个阶段的目标主要有四个：

- 理解 CartPole 环境的基本设置和强化学习问题形式。
- 使用 NumPy 实现一个最小版本的 `Policy Gradient / REINFORCE` 算法。
- 记录从 action sampling、discounted rewards 到 parameter update 的完整过程。
- 通过写作训练自己把技术概念解释清楚，而不是只会运行代码。

我希望这个项目最终不仅能展示我学习 RL 的过程，也能展示我对 research-driven learning 的兴趣：先理解问题，再设计实验，然后用结果反过来修正自己的理解。

## CartPole 问题是什么

CartPole 的任务可以想象成：一辆小车在水平轨道上移动，小车上有一根杆。智能体每一步都需要决定让小车向左还是向右移动，目标是让杆尽可能长时间不倒下。

在 Gymnasium 的 CartPole 环境中，状态通常由四个数值组成：

- 小车的位置（cart position）。
- 小车的速度（cart velocity）。
- 杆的角度（pole angle）。
- 杆的角速度（pole angular velocity）。

动作空间非常简单：

- `0`：向左推小车。
- `1`：向右推小车。

奖励机制也比较直观：只要杆还没有倒下，并且小车还没有超出边界，agent 每坚持一步就会获得 reward。任务结束通常发生在杆的角度过大、小车位置超出限制，或者达到最大步数。

这个设置虽然简单，但它已经包含了强化学习的关键问题：agent 不能只看眼前一步的 reward，而需要学习一种更长期有效的 `policy`。例如，小车当前向右移动不一定总是坏事，关键在于这个动作是否能在未来帮助杆保持平衡。

## 我目前已经完成和正在准备什么

目前我已经完成或正在准备的内容包括：

- 确定第一个强化学习项目主题为 CartPole。
- 阅读和整理强化学习的基础概念，包括 agent、environment、state、action、reward 和 policy。
- 准备使用 GitHub Blog 记录项目过程，而不是只把代码放在仓库里。
- 使用 NumPy 跑通了第一版 Policy Gradient 训练程序，文件为 `CartPole.py`。
- 通过训练日志观察 `reward total` 和 `running reward`，确认 agent 的表现可以随着训练逐步改善。

我选择先写下这篇文章，是因为我希望项目从一开始就有清楚的 research log。很多时候，学习技术很容易变成“跑通代码”，但我更希望记录自己为什么选择这个环境、如何理解这个问题，以及下一步准备怎样验证自己的想法。

## 第一版算法：Policy Gradient

在第一版实现中，我选择了 `Policy Gradient`，而不是 Q-learning 或 DQN。主要原因是：CartPole 的状态是连续的，如果直接做 Q-learning，需要额外把连续状态离散化；而 Policy Gradient 可以直接学习从 state 到 action probability 的映射，更适合我理解“策略如何被 reward 调整”这个核心问题。

这版程序没有使用 PyTorch，而是用 NumPy 手写了最基础的两层 policy network：

```text
state(4维) -> hidden layer(16维) -> sigmoid -> P(action=1)
```

其中 CartPole 的 state 包括小车位置、小车速度、杆的角度和杆的角速度。网络最后输出一个概率 `p`，表示当前状态下选择 `action=1` 的概率。如果随机数小于 `p`，agent 就选择向右推；否则选择向左推。

### 1. Policy forward：从 state 到 action probability

程序中的 `policy_forward` 函数完成前向传播：

```python
def policy_forward(x):
    h = np.dot(w1, x)
    h[h < 0] = 0
    logit = np.dot(w2, h)
    p = sigmoid(logit)
    return p, h
```

这里的逻辑是：

- `w1` 把 4 维 state 映射到 16 维 hidden layer。
- ReLU 把负数激活设为 0。
- `w2` 把 hidden layer 转成一个 logit。
- `sigmoid` 把 logit 转成 0 到 1 之间的动作概率。

我一开始容易把这个过程理解成“模型直接给出正确动作”，但现在更准确的理解是：policy network 给出的不是确定答案，而是一个 action distribution。agent 需要从这个分布中采样，这样它才有探索不同动作的机会。

### 2. 记录 episode：为什么不每一步立刻更新

在每一个 episode 中，我记录了四类信息：

- `xs`：每一步的 observation/state。
- `hs`：hidden layer 的 activation。
- `dlogps`：当前动作对应的 policy gradient 方向。
- `drs`：每一步获得的 reward。

程序不是每一步结束后立刻更新参数，而是等整个 episode 结束后再更新。原因是：某一步动作到底好不好，通常要看后面发生了什么。CartPole 中，如果一个 episode 坚持了更长时间，说明这一局中很多动作整体上更有价值。

### 3. discounted rewards：把即时 reward 变成长期回报

CartPole 每一步只要没有失败，就会得到 `reward = 1`。如果只看即时 reward，每一步似乎都一样。但强化学习关心的是长期结果，所以我实现了 `discount_rewards`：

```python
def discount_rewards(r):
    discounted = np.zeros_like(r, dtype=np.float64)
    running_add = 0

    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted[t] = running_add

    return discounted
```

它对应的思想是：

```text
G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
```

也就是说，一个动作不仅要看它当前得到的 reward，还要看它之后带来了多长时间的成功。`gamma = 0.99` 表示未来 reward 仍然很重要，只是越远的未来会稍微打折。

### 4. 用 reward 调整 policy gradient

episode 结束后，程序会把 `dlogp` 乘上标准化后的 discounted rewards：

```python
epdlogp *= discounted_epr
```

这是我理解 Policy Gradient 时最关键的一步。`dlogp` 本身只表示“这一步选择了什么动作，以及这个动作相对于当前概率应该往哪个方向调整”。但是只有乘上 discounted reward 后，程序才知道这个动作是否值得被鼓励。

我的理解是：

- 如果某一步动作后面带来了更高的长期回报，就提高这类动作在类似状态下被选择的概率。
- 如果某一步动作后面表现较差，就降低这类动作在类似状态下被选择的概率。

这一步把“动作选择”和“长期结果”连接了起来。

### 5. 参数更新：从结果回到模型

最后，程序计算 `dw2` 和 `dw1`，分别更新 hidden layer 到输出层、输入层到 hidden layer 的权重：

```python
dw2 = np.dot(eph.T, epdlogp).ravel()
dh = np.outer(epdlogp, w2)
dh[eph <= 0] = 0
dw1 = np.dot(dh.T, epx)

w1 += learning_rate * dw1
w2 += learning_rate * dw2
```

这里使用的是 gradient ascent，因为我们希望提高高回报动作出现的概率。和 supervised learning 中常见的 `loss minimization` 不同，这里更直观的说法是：沿着让好动作更可能出现的方向更新 policy。

## 实验结果与观察

这份程序目前设置为训练 `500` 个 episode，并且每 10 个 episode 打印一次：

```text
Episode 10: reward total was ..., running reward ...
Episode 20: reward total was ..., running reward ...
...
```

其中 `reward total` 是单个 episode 的总 reward，`running reward` 是一个平滑后的长期指标。由于初始权重是随机的，每次运行的具体数值都会不同，但我观察到训练过程中 running reward 有上升趋势，这说明 agent 开始学到一些让杆保持平衡的策略。

目前这还不是一个充分稳定或最优的 RL agent。更准确地说，它是我完成的第一个“从零理解并实现”的强化学习实验模型。它让我真正看到了：

- policy 不是固定规则，而是可以通过 reward 学出来的。
- reward 不能只看当前一步，而要通过 discounted return 连接到未来。
- agent 的学习过程本质上是在不断调整 action probability。
- 跑通代码只是第一步，更重要的是理解每一个变量为什么存在。

## 下一步计划

接下来我计划继续完善这个实验：

- 增加随机策略 baseline，和 Policy Gradient 的表现进行对比。
- 记录多次运行的平均 reward，而不是只看一次训练曲线。
- 尝试调整 `hidden_dim`、`learning_rate` 和 `gamma`，观察训练稳定性。
- 把 reward 曲线画出来，让实验结果更直观。
- 后续再考虑使用 PyTorch 实现同样的算法，比较手写 NumPy 和深度学习框架实现之间的差异。

## 研究反思：从 RL 到 Human-AI Collaboration

我对强化学习感兴趣，不只是因为它是一类机器学习算法，也因为它提供了一种理解智能体行为的方式。在 CartPole 中，agent 通过环境反馈不断调整自己的行为；在更复杂的现实场景中，人类和 AI 系统也会在反馈、建议、修正和协作中共同形成决策。

所以我开始思考一些更广泛的问题：

- AI agent 如何通过反馈学习更好的策略？
- 人类在与 AI 系统协作时，会如何调整自己的判断和行为？
- AI 的建议会增强人的批判性思维，还是可能让人过度依赖系统？
- 在创造力、学习和决策任务中，人和 AI 的分工应该如何设计？

这些问题也与我目前对 human-AI collaboration、AI-assisted decision-making 和 critical thinking 的兴趣有关。虽然 CartPole 本身是一个非常基础的 RL 环境，但它让我从一个具体任务出发，开始理解“智能体如何学习”和“反馈如何影响行为”这两个更大的研究主题。

## English Summary

This is an in-progress reinforcement learning project based on the CartPole environment. I implemented a simple Policy Gradient agent in NumPy, where a two-layer policy network maps the 4-dimensional CartPole state to the probability of choosing an action. After each episode, discounted returns are used to weight the policy gradient signal, allowing actions followed by higher long-term rewards to become more likely in future decisions.

## References / Learning Materials

- Gymnasium Documentation: https://gymnasium.farama.org/
- CartPole Environment Documentation: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*. http://incompleteideas.net/book/the-book-2nd.html
- Andrej Karpathy, “Deep Reinforcement Learning: Pong from Pixels”: http://karpathy.github.io/2016/05/31/rl/
- OpenAI Spinning Up, “Introduction to RL”: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
