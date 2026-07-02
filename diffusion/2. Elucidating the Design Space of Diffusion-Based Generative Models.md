# Elucidating the Design Space of Diffusion-Based Generative Models 阅读笔记

论文：Elucidating the Design Space of Diffusion-Based Generative Models  
链接：https://arxiv.org/abs/2206.00364  
作者：Tero Karras, Miika Aittala, Timo Aila, Samuli Laine

## 1. 阅读主线

这篇论文通常被称为 EDM。它的核心不是发明一个全新的扩散模型，而是把已有扩散模型拆成一组可独立讨论、替换和调优的设计选择。

论文的显式问题是：

```text
扩散模型质量很强，但理论和实现框架太耦合。
VP、VE、DDPM、DDIM、score SDE 等方法看起来像互不兼容的整体。
```

真正的结构张力是：

```text
一个扩散模型看似是一整套模型，
但实际由采样器、噪声尺度、时间步、网络预条件化、训练噪声分布和 loss 权重组成。
如果这些组件被历史推导绑定在一起，就很难知道哪个组件真正重要。
```

EDM 的第一原则是：

```text
生成过程 = 从高噪声分布出发，沿 denoiser 指出的方向逐步走回数据分布。
```

所以这篇论文最重要的对象不是某个历史框架名称，而是：

```text
D(x; sigma)
```

也就是：给定一个噪声强度为 `sigma` 的带噪图像 `x`，denoiser 应该输出它对应的干净图像估计。

## 2. 论文要解决的问题

扩散模型的早期论文通常从某个随机过程出发，然后推导出对应的训练目标和采样算法。这保证了理论一致性，但也带来一个问题：读者容易以为所有公式必须绑定使用。

EDM 反过来做：

```text
先找训练和采样中真正会被实现的对象
再把不同方法翻译到同一个坐标系里
最后逐个替换组件，看哪些选择真正改善质量或速度
```

它把设计空间拆成三类：

1. 采样过程：ODE/SDE、Heun、时间步、随机 churn。
2. 网络参数化：`D_theta` 如何由原始网络 `F_theta` 构成。
3. 训练过程：训练时采样哪些 `sigma`，不同 `sigma` 的 loss 怎么加权。

## 3. 核心符号

```text
y: 干净训练图像。
n: 高斯噪声，n ~ N(0, sigma^2 I)。
x: 带噪图像，通常 x = y + n。

sigma: 噪声标准差。sigma 越大，图像越接近纯噪声。
sigma_data: 数据本身的标准差，论文常用 sigma_data = 0.5。

p_data(x): 真实数据分布。
p(x; sigma): 数据加上标准差 sigma 的高斯噪声后的分布。

D(x; sigma): 理想 denoiser。
D_theta(x; sigma): 神经网络实现的 denoiser。
F_theta: 原始网络主体。EDM 不直接让 F_theta 等于 D_theta，而是通过预条件化得到 D_theta。

t: ODE/SDE 的时间变量。
sigma(t): 时间 t 对应的噪声强度。
s(t): 信号缩放函数。

N: 采样步数。
NFE: neural function evaluations，即生成一张图需要调用多少次网络。
rho: 控制 EDM 时间步分布的参数，论文默认 rho = 7。
```

概率记号要区分三件事：

```text
x ~ N(mu, Sigma): x 服从一个高斯分布。
N(x; mu, Sigma): 在 x 处计算高斯密度。
x = mu + sigma * epsilon: 实际采样公式，其中 epsilon ~ N(0, I)。
```

这三种写法的语义不同：

```text
x ~ N(...): 声明 x 是从哪个分布来的。
N(x; ...): 把 x 代入这个分布，计算它在该分布下的密度。
x = mu + sigma * epsilon: 真正生成一个样本 x。
```

`N(x; mu, Sigma)` 的意义不是得到一个绝对概率，而是衡量“当前这个 `x` 在以 `mu` 为中心、协方差为 `Sigma` 的高斯分布下有多合理”。在 EDM 的推导里，它常被用来做相对权重。

例如：

```text
N(x; y_i, sigma^2 I)
```

表示：如果干净图像是 `y_i`，并给它加标准差为 `sigma` 的高斯噪声，那么观察到当前带噪样本 `x` 的密度有多大。密度越大，说明 `x` 越像是由 `y_i` 加噪得到的，于是 `y_i` 在 denoiser 的加权平均中权重越大。

## 4. 方法框架

EDM 先定义一族被高斯噪声平滑后的数据分布：

```text
p(x; sigma) = p_data 与 N(0, sigma^2 I) 卷积
```

这句话可以理解为：先从真实数据分布里取一张干净图像，再给它加高斯噪声。

```text
y ~ p_data
n ~ N(0, sigma^2 I)
x = y + n
```

所有可能的 `x` 组成的分布，就是 `p(x; sigma)`。

`卷积` 在这里可以直观理解成：

```text
把每个真实样本 y 都扩散成一个以它为中心的小高斯云，
然后把所有小高斯云叠加起来。
```

所以：

```text
p_data: 干净图片世界
p(x; sigma): 加了 sigma 强度噪声后的图片世界
```

直觉是：

```text
sigma = 0: 就是真实数据分布
sigma 很大: 几乎是纯高斯噪声
```

采样时从大噪声开始：

```text
x0 ~ N(0, sigma_max^2 I)
```

然后逐步降低噪声：

```text
sigma_0 = sigma_max > sigma_1 > ... > sigma_N = 0
```

目标是让每一步的样本满足：

```text
x_i ~ p(x_i; sigma_i)
```

这行不是说 `x_i` 等于某个固定值，而是在描述第 `i` 步样本的理想分布状态：

```text
第 i 步的样本 x_i，
应该看起来像一张真实图像加上 sigma_i 强度高斯噪声后的结果。
```

例如：

```text
sigma_0 很大: x_0 应该像纯噪声。
sigma_i 中等: x_i 应该像模糊、有结构但仍带噪的图像。
sigma_N = 0: x_N 应该像真实干净图像。
```

最终 `sigma=0` 时，样本就来自真实数据分布。

### 为什么 denoiser 是核心

如果训练一个理想 denoiser：

```text
D(x; sigma) = 预测 x = y + n 中的干净图像 y
```

那么它和 score function 有一个关键关系：

```text
grad_x log p(x; sigma) = (D(x; sigma) - x) / sigma^2
```

这意味着：只要会训练 denoiser，就间接得到了 score。

score 的直觉是：

```text
score 指向当前噪声分布中更高密度的方向。
反向采样时，它告诉样本应该往哪里移动才能更像数据。
```

更细地看，`grad_x log p(x; sigma)` 可以拆成三层：

```text
p(x; sigma): 当前噪声强度下，x 的概率密度。
log p(x; sigma): 对密度取对数，不改变哪里大哪里小，但更便于求导。
grad_x: 对 x 求梯度，也就是问“x 往哪个方向移动，log 密度增长最快”。
```

所以 `grad_x log p(x; sigma)` 不是一个概率数值，而是一个方向向量。它给每个 `x` 一个箭头，指向当前噪声分布中更高密度的位置。

一维高斯例子：

```text
p(x) = N(x; mu, sigma^2)
log p(x) = - (x - mu)^2 / (2 sigma^2) + constant
d/dx log p(x) = (mu - x) / sigma^2
```

如果 `x` 在 `mu` 右边，梯度为负，箭头指向左；如果 `x` 在 `mu` 左边，梯度为正，箭头指向右；如果 `x = mu`，梯度为 0。也就是说，score 会把点拉向高斯中心。

## 5. 关键公式与推导

### 5.1 Denoising score matching

训练 denoiser 的目标是：

```text
E_{y ~ p_data, n ~ N(0, sigma^2 I)} ||D(y + n; sigma) - y||_2^2
```

这句话的含义是：

```text
从真实数据取一张图 y
给它加噪声 n
把 x = y + n 输入 denoiser
要求 denoiser 输出 y
```

### 推导

假设训练集是有限样本 `{y_i}`，那么加噪后的分布可以写成高斯混合：

```text
p(x; sigma) = (1 / Y) sum_i N(x; y_i, sigma^2 I)
```

对固定的 `x`，最优 denoiser 是一个加权平均：

```text
D(x; sigma)
= sum_i N(x; y_i, sigma^2 I) y_i / sum_i N(x; y_i, sigma^2 I)
```

计算 `p(x; sigma)` 的 score：

```text
grad_x log p(x; sigma)
= grad_x p(x; sigma) / p(x; sigma)
```

而单个高斯密度的梯度满足：

```text
grad_x N(x; y_i, sigma^2 I)
= N(x; y_i, sigma^2 I) * (y_i - x) / sigma^2
```

代回去：

```text
grad_x log p(x; sigma)
= (weighted_average_of_y_i - x) / sigma^2
= (D(x; sigma) - x) / sigma^2
```

### 直觉

`D(x; sigma)` 是“这个噪声图最可能来自哪个干净图”的估计。`D(x; sigma) - x` 就是从当前噪声图走向干净图的方向。除以 `sigma^2` 是因为噪声越大，同样的偏移在密度梯度里应该被缩放得越小。

这个关系能成立的根本原因是：`p(x; sigma)` 不是任意分布，而是“真实数据加高斯噪声”得到的分布。在这个场景下，Tweedie 公式告诉我们：

```text
E[y | x] = x + sigma^2 * grad_x log p(x; sigma)
```

而理想 denoiser 正是均方误差下对干净图像的最优估计：

```text
D(x; sigma) = E[y | x]
```

所以：

```text
D(x; sigma) = x + sigma^2 * grad_x log p(x; sigma)
grad_x log p(x; sigma) = (D(x; sigma) - x) / sigma^2
```

换句话说，`D(x; sigma) - x` 是“从当前带噪点指向它最可能干净来源”的方向；`grad_x log p(x; sigma)` 是“从当前点指向更高概率密度区域”的方向。因为当前分布就是由干净数据加高斯噪声形成的，所以这两个方向一致，只差一个由噪声方差带来的尺度因子。

### 5.2 Probability-flow ODE

论文使用的概率流 ODE 是：

```text
dx = - dot_sigma(t) * sigma(t) * grad_x log p(x; sigma(t)) dt
```

其中 `dot_sigma(t)` 是 `sigma(t)` 对时间的导数。

代入 denoiser 与 score 的关系：

```text
grad_x log p(x; sigma) = (D(x; sigma) - x) / sigma^2
```

就可以把 ODE 变成一个只需要调用 denoiser 的采样方程。

### 直觉

ODE 的作用是：当时间变化导致噪声强度变化时，样本也同步移动，使得样本始终落在对应噪声强度的边缘分布 `p(x; sigma)` 上。

反向生成就是：

```text
从大 sigma 的噪声分布
沿 ODE 走到 sigma = 0
得到数据样本
```

### 5.3 加入信号缩放后的通用 ODE

有些方法会使用额外的信号缩放 `s(t)`。论文把它写进通用框架：

```text
dx =
[
  dot_s(t) / s(t) * x
  - s(t)^2 * dot_sigma(t) * sigma(t)
    * grad_x log p(x / s(t); sigma(t))
] dt
```

这个形式让 VP、VE、iDDPM、DDIM 等方法可以放到同一张表里比较。

## 6. 训练过程

EDM 不直接令神经网络输出 `D_theta(x; sigma)`，而是定义：

```text
D_theta(x; sigma)
= c_skip(sigma) * x
  + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))
```

这里：

```text
c_skip: 控制输入 x 通过 skip connection 保留多少。
c_in: 控制输入到 F_theta 的幅度。
c_out: 控制 F_theta 输出的幅度。
c_noise: 把 sigma 编码成网络条件输入。
```

### 为什么需要预条件化

如果直接训练网络处理所有 `sigma`，输入尺度会剧烈变化：

```text
x = y + n
Var(x) = sigma_data^2 + sigma^2
```

当 `sigma` 很大时，输入几乎全是噪声；当 `sigma` 很小时，输入几乎是干净图。让同一个网络在所有尺度下都稳定学习，需要把输入、目标和 loss 权重调到可控范围。

### 6.1 输入缩放 `c_in`

要求输入到 `F_theta` 的方差为 1：

```text
Var(c_in(sigma) * (y + n)) = 1
```

因为：

```text
Var(y + n) = sigma_data^2 + sigma^2
```

所以：

```text
c_in(sigma) = 1 / sqrt(sigma^2 + sigma_data^2)
```

直觉：不管噪声强度多大，网络看到的输入幅度都差不多。

### 6.2 Skip connection `c_skip`

EDM 选择 `c_skip` 的原则是：让 `F_theta` 的误差被放大得尽量少。

推导结果：

```text
c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)
```

直觉：

```text
sigma 很小: c_skip 接近 1，输入本来就很干净，直接保留大部分 x。
sigma 很大: c_skip 接近 0，输入几乎全是噪声，不应该信任 x。
```

### 6.3 输出缩放 `c_out`

在确定 `c_skip` 后，要求 `F_theta` 的训练目标方差为 1，得到：

```text
c_out(sigma)
= sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
```

直觉：`F_theta` 输出始终处在比较稳定的数值范围里。

### 6.4 Loss 权重 `lambda`

训练 `F_theta` 时的有效权重是：

```text
w(sigma) = lambda(sigma) * c_out(sigma)^2
```

为了让不同噪声级别的初始 loss 权重均衡，设：

```text
w(sigma) = 1
```

因此：

```text
lambda(sigma)
= 1 / c_out(sigma)^2
= (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
```

### 6.5 训练噪声分布

EDM 不均匀采样 `sigma`，而是使用 log-normal 分布：

```text
ln(sigma) ~ N(P_mean, P_std^2)
```

论文默认：

```text
P_mean = -1.2
P_std = 1.2
```

直觉：

```text
极低噪声: 噪声太小，学习去除它对感知质量帮助有限。
极高噪声: 图像信息太少，目标接近数据均值。
中等噪声: 最影响图像结构，是训练最值得投入的区域。
```

## 7. 推理 / 采样过程

### 7.1 确定性采样：Heun sampler

确定性采样只在初始化时采样一次噪声，之后沿 ODE 走到 `sigma=0`。

简化过程：

```text
sample x0 ~ N(0, sigma(t0)^2 s(t0)^2 I)

for i = 0 ... N-1:
    计算当前斜率 d_i = dx/dt at t_i
    先用 Euler 预测 x_{i+1}
    如果下一步 sigma 不为 0:
        在预测点重新计算斜率 d'_i
        用 0.5 * d_i + 0.5 * d'_i 修正

return x_N
```

Euler 只看当前斜率。Heun 多看一步预测终点的斜率，相当于用梯形法近似曲线面积。

直觉：

```text
Euler: 我现在看到方向，就直接走过去。
Heun: 我先试走一步，看终点方向变没变，再取起点和终点方向的平均。
```

代价是每步多一次网络调用，但局部误差从一阶方法的 `O(h^2)` 改善到 `O(h^3)`。在 NFE 固定时，它通常更划算。

### 7.2 EDM 时间步

EDM 使用：

```text
sigma_i =
(
  sigma_max^(1/rho)
  + i / (N - 1) * (sigma_min^(1/rho) - sigma_max^(1/rho))
)^rho
```

并设：

```text
sigma_N = 0
rho = 7
```

`rho` 控制低噪声区域的步长密度。论文发现 `rho=3` 近似均衡每步截断误差，但 `rho=5` 到 `10` 的采样质量更好，说明低噪声区域的误差对最终图像影响更大。

### 7.3 为什么选择 `sigma(t)=t` 和 `s(t)=1`

在 EDM 推荐设置下：

```text
sigma(t) = t
s(t) = 1
```

ODE 简化为：

```text
dx/dt = (x - D(x; t)) / t
```

一个重要结果是：

```text
从任意 x 和 t 做一步 Euler 到 t=0，
会直接得到 D_theta(x; t)。
```

这说明轨迹的切线始终指向 denoiser 输出。论文认为这样会让轨迹更接近线性，从而更容易被低阶数值积分器近似。

### 7.4 随机采样：stochastic churn

随机采样在 deterministic Heun 基础上加了一个“临时加噪再去噪”的步骤。

简化过程：

```text
sample x0 ~ N(0, t0^2 I)

for i = 0 ... N-1:
    如果 ti 在允许 churn 的噪声范围内:
        gamma_i = min(S_churn / N, sqrt(2) - 1)
    否则:
        gamma_i = 0

    t_hat_i = ti + gamma_i * ti
    x_hat_i = x_i + sqrt(t_hat_i^2 - ti^2) * epsilon_i

    从 (x_hat_i, t_hat_i) 用 ODE 走到 t_{i+1}
    如果 t_{i+1} 不为 0:
        用 Heun 做二阶修正

return x_N
```

关键点：它不是通用 SDE solver，而是专门为扩散采样设计的修正机制。

直觉：

```text
确定性 ODE 可能把早期误差一路带到最后。
临时加噪能把样本重新推回对应噪声层的合理区域。
随后再用 denoiser 把它拉回低噪声层。
```

但随机性不是越多越好。过强的 churn 会损失细节或造成颜色漂移，所以论文只在特定噪声范围内启用，并用 `S_churn`、`S_tmin`、`S_tmax`、`S_noise` 控制强度。

## 8. 实验结论

读实验时不要只看 FID，而要看每个实验在证明哪个设计选择有效。

### 8.1 采样器可以独立替换

Table 3 比较确定性采样改动：

```text
VP 原采样器: FID 2.85, NFE 256
VP + EDM sigma(t), s(t): FID 2.93, NFE 35

VE 原采样器: FID 5.45, NFE 8192
VE + EDM sigma(t), s(t): FID 3.73, NFE 27

iDDPM/DDIM 原采样器: FID 2.85, NFE 250
iDDPM/DDIM + Heun/time steps: FID 2.64, NFE 79
```

结论：采样过程确实可以作为相对独立的模块替换，尤其 VE 的 NFE 改善非常大。

### 8.2 stochastic churn 对旧模型有用，但不是总有用

Table 4 中，随机采样的最优设置进一步改善了旧模型：

```text
VP deterministic baseline: FID 2.93, NFE 35
VP Algorithm 2 optimal: FID 2.27, NFE 383

VE deterministic baseline: FID 3.73, NFE 27
VE Algorithm 2 optimal: FID 2.23, NFE 767

ImageNet-64 deterministic baseline: FID 2.64, NFE 79
ImageNet-64 Algorithm 2 optimal: FID 1.55, NFE 511
```

结论：随机性可以修正旧模型或困难任务中的采样误差，但它需要更多 NFE，并且在训练更好的模型上未必有益。

### 8.3 训练侧最关键的是 loss weighting 和噪声分布

Table 2 展示训练配置逐步改进。以 CIFAR-10 为例：

```text
baseline: conditional FID 2.48 / 3.11
加入 EDM loss function 后: conditional FID 1.88 / 1.86
再加入 non-leaky augmentation: conditional FID 1.79 / 1.79
```

其中 preconditioning 本身不一定立刻显著降低 FID。它更像是让训练变稳的基础，使得重新设计 loss 和噪声采样分布可以发挥作用。

### 8.4 最终结果

论文报告的代表结果：

```text
CIFAR-10 class-conditional FID: 1.79
CIFAR-10 unconditional FID: 1.97
ImageNet-64 使用预训练模型替换采样器: FID 从 2.07 改到 1.55
ImageNet-64 重新训练后: FID 1.36
```

采样速度方面，CIFAR-10 高质量结果可用约 `35` 次网络评估得到。

## 9. 本次疑问记录

### 问题 1：这篇论文是不是提出新网络？

不是。它主要提出的是一套设计空间拆解和强默认配置。网络架构仍可使用 DDPM++、NCSN++ 等已有结构。

### 问题 2：为什么 denoiser 比 score 更适合作为核心对象？

因为 denoiser 可以通过普通监督式去噪 loss 训练，而 score 可以由 denoiser 推出：

```text
score = (D(x; sigma) - x) / sigma^2
```

这把抽象的密度梯度问题转成了直观的去噪问题。

### 问题 3：为什么低噪声区域要更小步长？

高噪声区域的样本还很粗糙，误差对最终图像结构影响相对小。低噪声区域已经接近图像流形，小误差更容易变成可见伪影，所以需要更细的步长。

### 问题 4：为什么 stochastic sampling 有时有用，有时有害？

它能修正早期采样误差，但也会引入新的离散化误差。对旧模型或 ImageNet-64 这类更困难任务，它有明显帮助；对 EDM 训练得更好的 CIFAR-10 模型，确定性采样反而最好。

### 问题 5：`N(x; mu, Sigma)` 计算密度有什么意义？

它衡量“当前这个 `x` 在某个高斯分布下有多合理”。在 EDM 里，`N(x; y_i, sigma^2 I)` 可以理解为：如果干净图像是 `y_i`，并给它加 `sigma` 强度的高斯噪声，那么观察到当前 `x` 的相对可能性有多大。

这个密度主要用于比较不同候选干净图像对当前带噪样本的解释力。密度越大，说明 `x` 越像是由对应的 `y_i` 加噪得到的。

### 问题 6：`p(x; sigma) = p_data` 与高斯噪声卷积是什么意思？

它表示先从真实数据分布中取干净图像：

```text
y ~ p_data
```

再加高斯噪声：

```text
n ~ N(0, sigma^2 I)
x = y + n
```

所有可能的 `x` 形成的分布就是 `p(x; sigma)`。直觉上，这是把每张干净图像扩散成一个高斯云，再把所有高斯云叠加起来。

### 问题 7：`x_i ~ p(x_i; sigma_i)` 是什么意思？

它描述采样过程中第 `i` 步的理想状态：当前样本 `x_i` 应该服从噪声强度为 `sigma_i` 的带噪数据分布。

也就是说：

```text
sigma_i 很大: x_i 应该像纯噪声。
sigma_i 中等: x_i 应该像有结构但仍带噪的图像。
sigma_i = 0: x_i 应该像真实干净图像。
```

扩散采样的目标，就是让样本沿着这些分布逐步移动。

### 问题 8：`grad_x log p(x; sigma)` 到底是什么？

它是 score，表示当前位置 `x` 应该往哪个方向移动，才能让 `log p(x; sigma)` 增长最快。它不是概率，而是方向向量。

如果把 `log p(x; sigma)` 看成地形高度，那么：

```text
x: 当前站的位置
log p(x; sigma): 当前高度
grad_x log p(x; sigma): 最陡上坡方向
```

在扩散模型里，上坡方向就是“更像当前噪声级别下的数据分布”的方向。

### 问题 9：为什么 `grad_x log p(x; sigma) = (D(x; sigma) - x) / sigma^2` 可以成立？

因为 `p(x; sigma)` 是真实数据加高斯噪声形成的分布。在这个场景下，理想 denoiser 是：

```text
D(x; sigma) = E[y | x]
```

也就是看到带噪样本 `x` 后，对干净图像 `y` 的最优均方估计。Tweedie 公式给出：

```text
E[y | x] = x + sigma^2 * grad_x log p(x; sigma)
```

整理后就是：

```text
grad_x log p(x; sigma) = (D(x; sigma) - x) / sigma^2
```

直觉上，`D(x; sigma) - x` 指向最可能的干净来源；`grad_x log p(x; sigma)` 指向更高密度区域。因为这个密度本来就是由干净数据加高斯噪声形成的，所以两者方向一致。

## 10. 最小心智模型

可以把 EDM 记成下面这条链：

```text
1. 数据加噪形成一族分布 p(x; sigma)
2. 训练 denoiser D(x; sigma) 从噪声图预测干净图
3. denoiser 给出 score: (D - x) / sigma^2
4. score 定义从大噪声走向小噪声的 ODE/SDE
5. 采样质量取决于数值积分误差和网络预测误差
6. Heun、rho=7、sigma(t)=t 降低采样误差
7. c_in/c_skip/c_out/lambda 降低训练难度
8. log-normal p_train(sigma) 把训练集中在最有用噪声区间
```

一句话版本：

```text
EDM 把扩散模型从“一个复杂理论包”重写成“一个 denoiser 加一组工程上可调的数值与训练设计”。
```

## 11. 盲点与后续阅读

EDM 的强项是拆解和经验验证，但不是对所有选择给出全局最优证明。

需要保留的盲点：

1. `rho=7` 是强经验默认值，不是数学必然。
2. `P_mean=-1.2`、`P_std=1.2` 也依赖数据和分辨率。
3. stochastic churn 与训练目标的交互仍不完全清楚，论文结论里也把它列为未来问题。
4. 高分辨率生成可能需要重新调参数，论文明确提到很多默认值在高分辨率数据集上可能不直接适用。
5. 更好的 FID 不等于所有应用都更安全或更可靠，论文也提到高质量生成可能放大虚假信息、偏见和算力消耗问题。

后续阅读顺序建议：

```text
DDPM -> Score SDE -> DDIM -> ADM/iDDPM -> EDM
```

如果已经读过 DDPM，下一步最值得补的是：

```text
1. Song et al. 的 score-based generative modeling / SDE 框架
2. DDIM 的确定性采样
3. ADM/iDDPM 的高质量图像生成改进
```
