### 概述
在这个项目中，你会需要实现一个 Q-learning算法来解决一个增强学习问题 -- 走迷宫。我们已经写好了 `testlearner.py` 来自动测试你的 Q-Learner. 

你的任务包括，

- 实现 Q-learner 类
- 针对走迷宫问题，测试／调试 你的Q-learner
 
### Github Repo
- 使用 github 更新你的 `qlearning_robot` 目录
- 在 `Qlearner.py` 实现 `QLearner` 类.
- 运行 `testqlearner.py` 来调试你的 `Q-learner`
- 我们在 `testworlds` 目录下提供了一些迷宫可以用来测试。


### 实现 Q-Learner
你不可以导入任何额外的库，你需要按照下面定义的 API，在 `QLearner.py` 中实现 `QLearner` 类。 

**构造函数 QLearner()** 应该预留空间存放 所有状态和行为的 Q-table Q[s, a], 并将整个矩阵初始化为 0. 构造函数的每一个参数如下定义：

- `num_states` integer, 所有状态个数。
- `num_actions` integer, 所有行为个数。
- `alpha` float, 更新Q-table时的学习率，范围 0.0 ~ 1.0, 常用值 0.2。
- `gamma` float, 更新Q-table时的衰减率，范围 0.0 ~ 1.0, 常用值 0.9。
- `rar` float, 随机行为比例, 每一步随机选择行为的概率。范围 0.0（从不随机） ~ 1.0（永远随机）, 常用值 0.5。
- `radr` float, 随机行为比例衰减率, 每一步都更新 rar = rar * radr. 0.0（直接衰减到0） ~ 1.0（从不衰减）, 常用值 0.99。
- `verbose` boolean, 如果为真，你的类可以打印调试语句，否则，禁止所有打印语句。

**query(s_prime, r)** 是 Q-Learner 的核心方法。他应该记录最后的状态 s 和最后的行为 a，然后使用新的信息 s_prime 和 r 来更新 Q-Table。 学习实例是四元组 `<s, a, s_prime, r>`. query() 应该返回一个 integer, 代表下一个行为。注意这里应该以 rar 的概率随机选择一个行为，并根据 radr 来更新 rar的值。

参数定义：

- `s_prime` integer, 新的状态
- `r` float, 即时奖励／惩罚，可以为正，可以为负。
 
**querysetstate(s)** query() 方法的特殊版本。设置状态为 s，并且返回下一个行为 a （和 query() 方法规则一致，例如包括以一定概率随机选择行为）。但是这个方法不更新 Q-table，不更新 rar。我们主要会在两个地方用到它： 1）设置初始状态 2) 使用学习后的策略，但不更新它

这里是一个使用 API 的例子

```
import QLearner as ql

learner = ql.QLearner(num_states = 100, \ 
    num_actions = 4, \
    alpha = 0.2, \
    gamma = 0.9, \
    rar = 0.98, \
    radr = 0.999, \
    verbose = False)

s = 99 # 初始状态

a = learner.querysetstate(s) # 状态s下的执行行为 a

s_prime = 5 # 在状态 s，执行行为 a 之后，进入新状态 s_prime

r = 0 # 在状态 s，执行行为 a 之后，获得即使奖励／惩罚 r

next_action = learner.query(s_prime, r)
```

### 走迷宫问题测试
我们会用如下方法测试你的 Q-Learner. 注意你的 Q-Learner 不应该知道任何有关走迷宫的信息。测试程序为 `testqlearner.py`, 你不需要做任何修改。迷宫的纬度是 10 * 10， 每一个迷宫都存储在csv文件中，用 integer 表示每个位置的属性，具体含义如下

- 0: 空地.
- 1: 障碍物.
- 2: 机器人的起始点.
- 3: 目标终点.
- 5: 陷阱.

一个迷宫 (world01.csv) 如下图所示 

```
3,0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0,0
0,0,1,1,1,1,1,0,0,0
0,5,1,0,0,0,1,0,0,0
0,5,1,0,0,0,1,0,0,0
0,0,1,0,0,0,1,0,0,0
0,0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0,0
0,0,0,0,2,0,0,0,0,0
```

在这个例子中，机器人从最后一行的中间位置开始，目标为第0行第0列，中间连续的障碍物组成一面墙阻挡路线，同时左边有很多陷阱。我们的目标是让机器人找到起点到终点奖励最高（或惩罚最低）的路线。 每一步的奖励／惩罚定义如下：

- **reward = -1** 如果机器人走进了一个空地，或遇到了障碍物。

- **reward = -100** 如果机器人走进了陷阱。

- **reward = 1** 如果机器人走到了终点。
 
**注意:** 每个行为都有一定概率不被执行。例如，如果 query() 返回的行为是向上走，会有一定的概率不往上走，而走其他方向。因此，一个 “聪明的” qlearner 应该尽可能得远离陷阱。

我们把这个问题用以下方式对应到增强学习：

- 状态: 状态是机器人的位置，由以下方式计算: `column_number * 10 + row_number` 
- 行为: 有四个可能的行为： 
	- 0: 向上走, 
	- 1: 向右走, 
	- 2: 向下走, 
	- 3: 向左走.
	
- 奖励: 奖励的方式如上所述 
- 转移表: 转移表可以从 csv 地图和行为中推断得出。

注意，奖励和转移表都是 qlearner 不知道的信息。 

`testqlearner.py` 的伪代码如下： 

```
Instantiate the learner with the constructor QLearner()
s = initial_location
a = querysetstate(s)
s_prime = new location according to action a
r = -1.0
while not converged:
    a = query(s_prime, r) 
    s_prime = new location according to action a
    if s_prime == goal:
        r = +1
        s_prime = start location
    else if s_prime == quicksand:
        r = -100
    else:
        r = -1
```

qlearner 每一步都会获得 -1 或 -100 的奖励，直到到达终点。衡量一个策略时，我们会从起点到终点行走500次，统计500次行走奖励的中位数，越高越好。