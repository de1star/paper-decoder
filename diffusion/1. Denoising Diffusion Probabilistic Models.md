# DDPM 论文阅读笔记

论文：Denoising Diffusion Probabilistic Models  
链接：https://arxiv.org/abs/2006.11239  
作者：Jonathan Ho, Ajay Jain, Pieter Abbeel

## 1. 阅读主线

这篇论文的核心不是发明一个复杂的生成器，而是把图像生成问题转化成一个反复去噪的问题。

整体逻辑是：

```text
真实图像 x0
  -> 前向过程逐步加噪
  -> 最终得到近似纯噪声 xT
  -> 训练模型学习反向去噪
  -> 从随机噪声一步步生成图像
```

最关键的转化是：

```text
生成建模问题
=> 多步去噪问题
=> 噪声预测的监督学习问题
```

## 2. Abstract + Introduction

论文想解决的问题是：扩散模型在理论上很自然，但此前没有证明它能生成高质量图像。

这篇论文的主要贡献是：

1. 证明扩散模型可以生成高质量图像。
2. 使用噪声预测参数化，也就是让网络预测 `epsilon`。
3. 使用简化训练目标 `L_simple`，把训练变成噪声预测的 MSE。
4. 建立扩散模型、denoising score matching、Langevin dynamics 之间的联系。
5. 通过实验说明 `epsilon prediction + L_simple` 是生成质量提升的关键。

## 3. 基础符号

```text
x0: 原始真实图像
xt: 第 t 步加噪后的图像
xT: 最后一步，接近纯高斯噪声
T: 总扩散步数，论文中常用 T = 1000

beta_t: 第 t 步加入的噪声强度
alpha_t = 1 - beta_t
alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t

epsilon: 标准高斯噪声，epsilon ~ N(0, I)
epsilon_theta(xt, t): 神经网络预测的噪声
```

`sqrt` 是 square root，也就是平方根：

```text
sqrt(x) = √x
```

例如：

```text
sqrt(4) = 2
sqrt(9) = 3
sqrt(0.25) = 0.5
```

在高斯采样中，分布里写的是方差，实际采样时乘的是标准差：

```text
标准差 = sqrt(方差)
```

## 4. 前向过程：forward process

前向过程是固定的，不需要训练。

论文定义：

```text
q(xt | x_{t-1})
= N(xt; sqrt(1 - beta_t)x_{t-1}, beta_t I)
```

因为：

```text
alpha_t = 1 - beta_t
```

所以也可以写成：

```text
q(xt | x_{t-1})
= N(xt; sqrt(alpha_t)x_{t-1}, (1 - alpha_t)I)
```

等价采样形式：

```text
xt = sqrt(alpha_t)x_{t-1} + sqrt(1 - alpha_t)epsilon_t
```

含义是：

```text
当前图像 = 保留一部分上一时刻图像 + 加入一部分高斯噪声
```

如果 `beta_t` 很小，每一步只加一点噪声；重复很多步后，图像结构逐渐消失，最后接近纯噪声。

## 5. `alpha_bar_t` 是什么

`alpha_bar_t` 写成数学符号是：

```text
\bar{alpha}_t
```

它表示从第 1 步到第 `t` 步累计保留下来的原图比例。

定义：

```text
alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t
```

直觉上：

```text
alpha_t: 单步保留多少信号
alpha_bar_t: 从 x0 到 xt 总共还保留多少原始信号
```

例如，如果每一步都保留 `0.9`：

```text
alpha_bar_3 = 0.9 * 0.9 * 0.9 = 0.729
```

也就是说，到第 3 步时，原始信号大约还剩 `72.9%`。

## 6. 为什么可以一步得到任意 `xt`

论文给出：

```text
q(xt | x0)
= N(xt; sqrt(alpha_bar_t)x0, (1 - alpha_bar_t)I)
```

这表示：给定原始图像 `x0`，第 `t` 步加噪后的图像 `xt` 服从一个高斯分布。

其中：

```text
sqrt(alpha_bar_t)x0
```

是高斯分布的均值，也就是保留下来的原图成分。

```text
(1 - alpha_bar_t)I
```

是协方差，表示噪声强度。

根据高斯采样规则：

```text
如果 x ~ N(mu, sigma^2 I)
那么 x = mu + sigma epsilon
其中 epsilon ~ N(0, I)
```

所以：

```text
q(xt | x0)
= N(xt; sqrt(alpha_bar_t)x0, (1 - alpha_bar_t)I)
```

等价于：

```text
xt = sqrt(alpha_bar_t)x0 + sqrt(1 - alpha_bar_t)epsilon
```

直觉上：

```text
xt = 保留下来的原图 + 加进去的随机噪声
```

## 7. 这个公式为什么成立

这个公式不是额外假设，而是从每一步的加噪过程推导出来的。

DDPM 真正先定义的是：

```text
x_t = sqrt(alpha_t)x_{t-1} + sqrt(1 - alpha_t)epsilon_t
```

第 1 步：

```text
x_1 = sqrt(alpha_1)x_0 + sqrt(1 - alpha_1)epsilon_1
```

第 2 步：

```text
x_2 = sqrt(alpha_2)x_1 + sqrt(1 - alpha_2)epsilon_2
```

代入 `x_1`：

```text
x_2
= sqrt(alpha_2)[sqrt(alpha_1)x_0 + sqrt(1 - alpha_1)epsilon_1]
  + sqrt(1 - alpha_2)epsilon_2
```

整理：

```text
x_2
= sqrt(alpha_1 alpha_2)x_0
  + sqrt(alpha_2)sqrt(1 - alpha_1)epsilon_1
  + sqrt(1 - alpha_2)epsilon_2
```

后两项是两个独立高斯噪声的线性组合，仍然是高斯噪声。

它的总方差是：

```text
alpha_2(1 - alpha_1) + (1 - alpha_2)
= 1 - alpha_1 alpha_2
```

所以可以合并成：

```text
sqrt(alpha_2)sqrt(1 - alpha_1)epsilon_1
+ sqrt(1 - alpha_2)epsilon_2
= sqrt(1 - alpha_1 alpha_2)epsilon
```

于是：

```text
x_2 = sqrt(alpha_1 alpha_2)x_0
    + sqrt(1 - alpha_1 alpha_2)epsilon
```

推广到第 `t` 步：

```text
x_t = sqrt(alpha_bar_t)x_0
    + sqrt(1 - alpha_bar_t)epsilon
```

成立的根源是：

```text
DDPM 的前向过程是一个线性高斯马尔可夫链。
```

## 8. 独立高斯噪声线性组合仍是高斯

这不是公理，而是高斯分布的一个定理，也可以理解为高斯分布的闭包性质。

如果：

```text
epsilon_1 ~ N(0, I)
epsilon_2 ~ N(0, I)
```

并且二者独立，那么对任意常数 `a, b`：

```text
a epsilon_1 + b epsilon_2
```

仍然是高斯分布：

```text
a epsilon_1 + b epsilon_2 ~ N(0, (a^2 + b^2)I)
```

因为：

```text
Var(a epsilon_1 + b epsilon_2)
= a^2 Var(epsilon_1) + b^2 Var(epsilon_2)
= (a^2 + b^2)I
```

所以它可以写成：

```text
a epsilon_1 + b epsilon_2 = sqrt(a^2 + b^2)epsilon
```

其中：

```text
epsilon ~ N(0, I)
```

独立性很关键。如果不独立，中间会出现协方差项。

## 9. 前向过程里的独立性假设

DDPM 的前向过程确实有一个结构性假设：

```text
q(x_{1:T} | x0) = ∏_{t=1}^T q(xt | x_{t-1})
```

这表示前向加噪过程是一个马尔可夫链。

它包含两层意思：

```text
q(xt | x_{t-1}, x_{t-2}, ..., x0)
= q(xt | x_{t-1})
```

也就是当前状态只依赖上一个状态。

同时，每一步注入的噪声可以看成独立采样：

```text
epsilon_1, epsilon_2, ..., epsilon_T 独立同分布
epsilon_t ~ N(0, I)
```

更精确地说：

```text
DDPM 假设前向扩散过程是一个线性高斯马尔可夫链，
因此每一步注入的高斯噪声相互独立，
并且当前状态只依赖上一个状态。
```

## 10. 反向过程：reverse process

前向过程 `q` 是固定加噪过程；反向过程 `p_theta` 是要学习的去噪过程。

论文定义：

```text
p_theta(x_{t-1} | xt)
= N(x_{t-1}; mu_theta(xt, t), Sigma_theta(xt, t))
```

意思是：

```text
给定当前更脏的图 xt，
模型认为上一时刻更干净的图 x_{t-1}
服从一个高斯分布。
```

其中：

```text
mu_theta(xt, t): 模型预测的均值
Sigma_theta(xt, t): 模型预测或指定的协方差
```

模型不是直接说：

```text
x_{t-1} = 某个确定图像
```

而是说：

```text
x_{t-1} 大概率在 mu_theta(xt, t) 附近，
不确定性由 Sigma_theta(xt, t) 控制。
```

生成时：

```text
xT ~ N(0, I)
xT -> xT-1 -> ... -> x0
```

每一步都使用：

```text
x_{t-1} ~ p_theta(x_{t-1} | xt)
```

## 11. 真实后验与模型后验

训练时，如果已知 `x0` 和 `xt`，可以推导出真实后验：

```text
q(x_{t-1} | xt, x0)
= N(x_{t-1}; mu_tilde_t(xt, x0), beta_tilde_t I)
```

含义是：

```text
如果知道原图 x0，又知道当前 noisy image xt，
理论上可以算出上一时刻 x_{t-1} 的真实分布。
```

但生成时没有 `x0`，所以模型只能学习近似：

```text
p_theta(x_{t-1} | xt)
≈ q(x_{t-1} | xt, x0)
```

这就是训练目标的来源。

## 12. Variational Bound

论文使用变分上界训练模型。

它把负对数似然上界拆成：

```text
L = LT + Σ L_{t-1} + L0
```

三部分含义：

```text
LT:
q(xT | x0) 是否接近标准高斯 p(xT)

L_{t-1}:
模型反向去噪分布 p_theta(x_{t-1} | xt)
是否接近真实后验 q(x_{t-1} | xt, x0)

L0:
最后一步是否能还原离散图像 x0
```

最关键的是中间项：

```text
L_{t-1}
= KL(q(x_{t-1} | xt, x0) || p_theta(x_{t-1} | xt))
```

它要求：

```text
真实反向一步应该怎么走，
模型预测的反向一步就应该尽量接近它。
```

由于这两个分布都是高斯分布，KL 可以化简成均值之间的平方误差。

## 13. 为什么改成预测噪声

由前向公式：

```text
xt = sqrt(alpha_bar_t)x0 + sqrt(1 - alpha_bar_t)epsilon
```

可以反解出：

```text
x0 = (xt - sqrt(1 - alpha_bar_t)epsilon) / sqrt(alpha_bar_t)
```

如果模型能预测噪声：

```text
epsilon ≈ epsilon_theta(xt, t)
```

那么就可以估计原图：

```text
x0_hat
= (xt - sqrt(1 - alpha_bar_t)epsilon_theta(xt, t))
  / sqrt(alpha_bar_t)
```

所以预测噪声本质上就是学习去噪。

DDPM 中反向均值也可以由预测噪声构造：

```text
mu_theta(xt, t)
= 1 / sqrt(alpha_t)
  * (xt - beta_t / sqrt(1 - alpha_bar_t)
     * epsilon_theta(xt, t))
```

因此模型表面上是在预测反向分布，实际训练中主要是在学习：

```text
这张 noisy image 里混进了多少噪声？
```

## 14. 简化目标 `L_simple`

论文最重要的训练目标是：

```text
L_simple(theta)
= E_{t, x0, epsilon} [
    ||epsilon - epsilon_theta(
        sqrt(alpha_bar_t)x0 + sqrt(1 - alpha_bar_t)epsilon,
        t
    )||^2
  ]
```

简写为：

```text
L_simple = E ||epsilon - epsilon_theta(xt, t)||^2
```

逐项解释：

```text
epsilon:
真实加进去的噪声

epsilon_theta(xt, t):
神经网络预测的噪声

epsilon - epsilon_theta(xt, t):
真实噪声和预测噪声的差

||epsilon - epsilon_theta(xt, t)||^2:
平方误差，也就是 MSE

E:
对 x0、t、epsilon 的随机采样取平均
```

训练过程：

```text
1. 取一张真实图片 x0
2. 随机选一个时间步 t
3. 随机采样噪声 epsilon
4. 构造 xt = sqrt(alpha_bar_t)x0 + sqrt(1 - alpha_bar_t)epsilon
5. 把 xt 和 t 输入网络
6. 网络输出预测噪声 epsilon_theta(xt, t)
7. 用 MSE 惩罚预测噪声和真实噪声的差距
```

一句话：

```text
L_simple 是噪声预测任务的均方误差。
```

它把复杂的生成建模问题变成了一个直接的监督学习问题：

```text
我给你加噪图，你告诉我加了什么噪声。
```

需要注意的是，`L_simple` 不是原始 ELBO 的严格等价形式，而是一个重新加权后的目标。论文发现它牺牲了一些 likelihood 表现，但显著提升了样本质量。

## 15. Algorithm 1：训练过程

训练算法可以概括为：

```text
repeat:
    x0 ~ data
    t ~ Uniform({1, ..., T})
    epsilon ~ N(0, I)

    xt = sqrt(alpha_bar_t)x0 + sqrt(1 - alpha_bar_t)epsilon

    loss = ||epsilon - epsilon_theta(xt, t)||^2

    update theta
```

关键点：

```text
训练时不用真的从 x0 一步步加噪到 xt，
而是直接用闭式公式采样任意时间步的 xt。
```

这就是 `q(xt | x0)` 公式的工程价值。

## 16. Algorithm 2：采样过程

采样从纯噪声开始：

```text
xT ~ N(0, I)
```

然后从 `T` 到 `1` 反复去噪：

```text
for t = T, T-1, ..., 1:
    预测 epsilon_theta(xt, t)
    根据采样公式得到 x_{t-1}
```

采样公式：

```text
x_{t-1}
= 1 / sqrt(alpha_t)
  * (xt - beta_t / sqrt(1 - alpha_bar_t)
     * epsilon_theta(xt, t))
  + sigma_t z
```

其中：

```text
epsilon_theta(xt, t):
模型预测当前图像里的噪声

xt - beta_t / sqrt(1 - alpha_bar_t) * epsilon_theta(xt, t):
从当前图像里减掉一部分噪声

1 / sqrt(alpha_t):
尺度校正

sigma_t z:
重新加入一点随机性
```

当 `t > 1` 时：

```text
z ~ N(0, I)
```

当 `t = 1` 时，通常不再加噪声。

训练和采样的区别：

```text
训练:
知道真实 epsilon，让模型学会预测它

采样:
不知道真实 epsilon，只能用模型预测的 epsilon 一步步去噪
```

## 17. Section 4：实验结论

实验部分主要证明：

```text
epsilon prediction + L_simple
```

是真正让 DDPM 生成质量变强的关键。

论文在 CIFAR-10 上达到：

```text
FID = 3.17
IS = 9.46
```

当时这是非常强的无条件生成结果。

消融实验比较了：

```text
预测均值 mu
预测噪声 epsilon
使用完整 variational bound L
使用简化目标 L_simple
学习方差 Sigma
固定方差 Sigma
```

结论：

```text
预测 epsilon + L_simple -> 样本质量最好
学习方差 -> 训练不稳定
完整 ELBO -> likelihood 稍好，但样本质量不如 L_simple
```

这说明：

```text
生成质量最好的目标，不一定是 likelihood 最优的目标。
```

## 18. 最小心智模型

把整篇论文压缩成三行：

```text
前向过程:
xt = sqrt(alpha_bar_t)x0 + sqrt(1 - alpha_bar_t)epsilon
```

```text
训练目标:
让网络从 xt 和 t 预测 epsilon
```

```text
采样过程:
从 xT ~ N(0, I) 开始，反复用预测到的 epsilon 去噪
```

最终理解：

```text
DDPM = 一个在所有噪声强度下训练的去噪模型。
```
