# High-Resolution Image Synthesis with Latent Diffusion Models 阅读笔记

论文：High-Resolution Image Synthesis with Latent Diffusion Models  
链接：https://arxiv.org/abs/2112.10752  
作者：Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer  
常用简称：LDM，Latent Diffusion Models

## 1. 阅读计划回顾

这篇论文的阅读重点不是“Stable Diffusion 是怎么画图的”，而是理解一个结构性转移：

```text
像素空间扩散
  -> 计算贵，但图像细节完整
潜空间扩散
  -> 先用 autoencoder 做轻度感知压缩
  -> 再在 latent map 上做 DDPM 风格去噪
  -> 最后一次 decode 回图像
```

建议按以下顺序阅读：

1. `Abstract`、`Introduction`、Fig. 1、Fig. 2：确认论文为什么要离开 pixel space。
2. `Section 3.1`：读懂 first-stage autoencoder 如何提供感知等价的 latent space。
3. `Section 3.2`：读懂 LDM objective 为什么几乎就是 DDPM objective 的变量替换。
4. `Section 3.3`：读懂 cross-attention 如何把 text / layout / class 等条件接入 UNet。
5. `Section 4.1`：把不同压缩因子 `f` 的实验当作整篇论文的核心证据。
6. `Section 4.2-4.5`：看 unconditional generation、text-to-image、super-resolution、inpainting 如何证明方法的通用性。
7. `Limitations`：回到 autoencoder 压缩瓶颈、采样速度和社会影响。

## 2. 阅读主线

显式问题是：

```text
扩散模型生成质量很强，但在像素空间训练和采样都太昂贵，
尤其是高分辨率图像需要大量重复的 UNet 前向 / 反向计算。
```

真正的结构张力是：

```text
图像像素里有大量人眼不敏感的高频细节。
likelihood-based 模型会花容量去建模这些细节；
但如果像 VQGAN + autoregressive transformer 那样过度压缩，
又会损伤重建质量和空间结构。
```

LDM 的第一原则是：

```text
把“感知压缩”和“语义生成”分开。

autoencoder 负责去掉感知上不重要的像素冗余；
diffusion model 负责在更小但仍保留二维结构的 latent space 中学习生成分布。
```

所以整篇论文的主线可以压缩成：

```text
真实图像 x
  -> encoder E
潜变量 z = E(x)
  -> 在 z 上加噪得到 z_t
  -> UNet 学习 epsilon_theta(z_t, t, condition)
  -> 从噪声 latent 逐步去噪得到 z_0
  -> decoder D
生成图像 x = D(z_0)
```

## 3. 论文要解决的问题

DDPM 已经把图像生成转化成了噪声预测问题：

```text
sample x
sample t
sample epsilon
construct x_t
predict epsilon
minimize MSE
```

这让训练目标非常稳定，但带来一个直接成本：每一步网络都在完整像素空间里运行。

如果图像是 `512 x 512 x 3`，模型在每次 denoising step 里都要处理完整 RGB tensor。扩散采样又需要几十到上千步，所以成本被重复放大。

LDM 的关键判断是：

```text
高分辨率图像生成不一定要在高分辨率像素空间中建模。
```

更准确地说，扩散模型真正需要学习的是图像的语义结构、布局、纹理统计和条件控制关系；许多像素级高频细节可以先交给一个高质量 autoencoder 处理。

## 4. 核心贡献

这篇论文的贡献可以读成四个层次：

1. 用 first-stage autoencoder 把图像压缩到感知等价的 latent space。
2. 在 latent map 上训练 diffusion model，显著降低训练和采样成本。
3. 保留二维空间结构，所以 latent prior 可以继续用卷积 UNet，而不是把 latent token 拉平成一维序列给 autoregressive transformer。
4. 用 cross-attention 建立通用条件机制，使 text-to-image、layout-to-image、class-conditional、inpainting、super-resolution 都能落到同一框架。

## 5. 核心符号

```text
x: 原始 RGB 图像，x in R^{H x W x 3}

E: encoder，把图像编码到 latent space
D: decoder，把 latent 还原成图像

z = E(x): 图像 x 的 latent representation
x_tilde = D(z) = D(E(x)): autoencoder 重建图像

z in R^{h x w x c}: latent map
f = H / h = W / w: 空间下采样因子
LDM-f: 使用下采样因子 f 的 Latent Diffusion Model

t: diffusion timestep
T: 总扩散步数
epsilon: 标准高斯噪声，epsilon ~ N(0, I)
z_t: 第 t 步加噪后的 latent

epsilon_theta(z_t, t): UNet 在 latent space 中预测噪声
y: 条件输入，例如文本、类别、语义图、低分辨率图像、mask 等
tau_theta(y): condition encoder，把 y 编码成可供 cross-attention 使用的表示

phi_i(z_t): UNet 第 i 层的中间特征，通常展平成 token-like feature
Q, K, V: cross-attention 中的 query、key、value
```

和 DDPM 的对应关系：

```text
DDPM:
    x_0 -> x_t -> epsilon_theta(x_t, t)

LDM:
    x -> z = E(x)
    z_0 -> z_t -> epsilon_theta(z_t, t)
```

也就是说，LDM 没有推翻 DDPM 的训练逻辑，而是替换了扩散过程所在的空间。

## 6. 方法框架

LDM 是一个两阶段系统。

第一阶段：感知压缩。

```text
x
  -> E(x)
z
  -> D(z)
x_tilde
```

这一阶段训练 autoencoder。论文使用 perceptual loss 和 patch-based adversarial objective，让重建结果更贴近真实图像流形，避免单纯 `L1` / `L2` loss 带来的模糊。

这里的关键不是扩散模型的 loss，而是 autoencoder 的重建质量控制。

`perceptual loss` 比较的不是原图和重建图的逐像素差异，而是它们经过预训练视觉网络后的特征差异：

```text
pixel loss:
    compare x and x_tilde directly

perceptual loss:
    compare phi(x) and phi(x_tilde)
```

其中 `phi` 可以理解成 VGG / LPIPS 这类视觉特征提取器。它更关心“人眼看来是否相似”，而不是每个像素是否精确对齐。

`patch-based adversarial objective` 来自 GAN 的思想，但判别器关注局部 patch：

```text
decoder:
    tries to reconstruct realistic image patches

patch discriminator:
    tries to distinguish real image patches from reconstructed patches
```

它的作用是逼 decoder 生成更真实的局部纹理、边缘和细节。单纯 `L1` / `L2` 常会鼓励“平均答案”，导致图像变糊；perceptual loss 和 patch adversarial loss 则帮助 autoencoder 保持视觉真实感。

第二阶段：latent diffusion。

```text
z = E(x)
  -> add noise
z_t
  -> epsilon_theta(z_t, t, condition)
predicted noise
```

这一阶段固定或复用已经训练好的 autoencoder，只训练 latent space 中的 diffusion prior。

关键是：`z` 仍然是二维 feature map，而不是一串没有空间结构的 token。因此 UNet 的卷积归纳偏置仍然有效。

## 7. 为什么不是直接用 VQGAN + Transformer

早期两阶段生成方法也会先压缩图像，再在 latent 上建模。例如 VQ-VAE、VQGAN、DALL-E 风格 autoregressive transformer。

但这些方法通常有一个压力：

```text
如果 latent token 太多，autoregressive transformer 成本太高；
如果 latent token 太少，压缩过猛，重建质量下降。
```

LDM 的不同点是：latent prior 不是 autoregressive transformer，而是 convolutional UNet diffusion model。

这带来两个结果：

1. 不必把 latent space 压得非常小。
2. 可以保留 latent 的二维空间结构。

所以 LDM 的压缩是“轻度感知压缩”，不是“为了让序列模型能跑而进行的强压缩”。

## 8. 关键公式一：DDPM 噪声预测目标

论文先回顾 DDPM 的简化目标：

$$
L_{DM}
=
\mathbb{E}_{x,\epsilon\sim\mathcal{N}(0,1),t}
\Big[
\|\epsilon-\epsilon_{\theta}(x_t,t)\|_2^2
\Big].
$$

### 公式逐项解释

先把公式读成一句话：

```text
随机抽一张真实图像 x，
随机抽一份高斯噪声 epsilon，
随机抽一个时间步 t，
构造带噪图像 x_t，
然后训练网络 epsilon_theta(x_t, t) 预测这份噪声 epsilon。
```

各项含义是：

```text
L_DM:
    diffusion model 的训练损失。

E_{x, epsilon ~ N(0,1), t}[...]:
    对很多随机样本取平均期望。
    训练时实际就是 minibatch loss 的平均。

x:
    从训练集中采样的真实图像。

epsilon ~ N(0,1):
    从标准高斯分布采样的噪声。

t:
    随机采样的扩散时间步。

x_t:
    第 t 步的带噪图像。

epsilon_theta(x_t, t):
    参数为 theta 的神经网络预测出来的噪声。

||epsilon - epsilon_theta(x_t, t)||_2^2:
    真实噪声和预测噪声之间的平方误差。
```

`epsilon_theta(x_t,t)` 不是一个已知公式，而是神经网络：

```text
input:
    x_t: 当前带噪图像
    t: 当前噪声强度 / 时间步

output:
    model's prediction of epsilon
```

训练时真实噪声 `epsilon` 是已知的，因为 `x_t` 是我们自己用 `x_0` 和 `epsilon` 构造出来的。模型学习的是：

```text
epsilon_theta(x_t, t) ≈ epsilon
```

这里必须输入 `t`，因为不同时间步的噪声强度不同：

```text
t small:
    image is still close to x_0

t large:
    image is close to pure noise
```

### 推导

从 Appendix B 的写法看，forward process 可以写成：

$$
q(x_t|x_0)=\mathcal{N}(x_t|\alpha_t x_0,\sigma_t^2 I).
$$

这个式子不是说 `x_t` 等于某个固定值，而是说：

```text
在给定干净图像 x_0 的条件下，
x_t 服从一个高斯分布。
```

这个高斯分布的均值和方差是：

```text
mean:
    alpha_t * x_0

variance:
    sigma_t^2 * I
```

也就是说，`x_t` 是“剩余图像信号 + 随机噪声”。

等价采样形式是：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\quad \epsilon\sim\mathcal{N}(0,I).
$$

这两种写法等价，因为：

```text
epsilon ~ N(0, I)
sigma_t * epsilon ~ N(0, sigma_t^2 I)
alpha_t * x_0 + sigma_t * epsilon ~ N(alpha_t x_0, sigma_t^2 I)
```

所以可以写成分布形式：

```text
q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I)
```

### alpha_t 和 sigma_t 怎么来

在 DDPM 里通常先定义每一步的噪声强度 `beta_t`。

为了避免符号混淆，把每一步的信号保留比例暂时记为：

```text
a_t = 1 - beta_t
```

逐步加噪可以写成：

$$
q(x_t|x_{t-1})
=
\mathcal{N}(\sqrt{a_t}x_{t-1}, \beta_t I).
$$

从 `x_0` 直接跳到 `x_t` 时，需要把前面所有步骤的信号保留比例乘起来：

$$
\bar{\alpha}_t
=
\prod_{s=1}^{t} a_s
=
\prod_{s=1}^{t}(1-\beta_s).
$$

于是有：

$$
\alpha_t = \sqrt{\bar{\alpha}_t},
\quad
\sigma_t = \sqrt{1-\bar{\alpha}_t}.
$$

所以计算链条是：

```text
beta_t
  -> a_t = 1 - beta_t
  -> alpha_bar_t = a_1 * a_2 * ... * a_t
  -> alpha_t = sqrt(alpha_bar_t)
  -> sigma_t = sqrt(1 - alpha_bar_t)
```

因此在常见 VP / DDPM 设定下：

$$
\alpha_t^2 + \sigma_t^2 = 1.
$$

直觉是：

```text
alpha_t:
    controls how much original signal remains

sigma_t:
    controls how much Gaussian noise is added

t increases:
    alpha_t decreases
    sigma_t increases
```

论文后面也用它们定义信噪比：

$$
\text{SNR}(t)=\frac{\alpha_t^2}{\sigma_t^2}.
$$

反向模型要从 `x_t` 估计干净样本 `x_0`，也可以等价地估计噪声：

$$
\epsilon_{\theta}(x_t,t)
=
\frac{x_t-\alpha_t x_{\theta}(x_t,t)}{\sigma_t}.
$$

因此重建误差可以改写成噪声预测误差：

$$
\|x_0-x_\theta(x_t,t)\|^2
=
\frac{\sigma_t^2}{\alpha_t^2}
\|\epsilon-\epsilon_\theta(x_t,t)\|^2.
$$

DDPM 的常用简化训练目标就是直接让网络预测加进去的噪声 `epsilon`。

### 直觉

训练时我们知道干净图像 `x_0`，也知道自己加了哪份噪声 `epsilon`。所以可以把生成建模问题变成监督学习：

```text
输入：带噪图像 x_t 和 timestep t
目标：预测这次加进去的噪声 epsilon
loss：预测噪声和真实噪声的 MSE
```

如果模型能在每个噪声强度下预测噪声，就能在采样时从纯噪声一步步移除噪声，回到数据分布。

## 9. 关键公式二：Latent Diffusion 目标

LDM 把上面的目标从 pixel space 改到 latent space：

$$
L_{LDM}
:=
\mathbb{E}_{\mathcal{E}(x),\epsilon\sim\mathcal{N}(0,1),t}
\Big[
\|\epsilon-\epsilon_{\theta}(z_t,t)\|_2^2
\Big].
$$

### 推导

先把图像编码成 latent：

$$
z = \mathcal{E}(x).
$$

然后在 latent 上执行同样的 forward noising：

$$
z_t=\alpha_t z+\sigma_t\epsilon.
$$

训练目标变成：

```text
给定 z_t 和 t，预测加到 z 上的噪声 epsilon。
```

也就是：

$$
\epsilon_\theta(z_t,t)\approx\epsilon.
$$

因此损失函数就是：

$$
\|\epsilon-\epsilon_\theta(z_t,t)\|_2^2.
$$

### 直觉

这条公式的重点不是数学变复杂了，而是变量换了。

```text
DDPM:
    在 x_t 上去噪

LDM:
    在 z_t 上去噪
```

如果 `E` 和 `D` 足够好，`z` 就是一个更便宜、更紧凑、但仍保留感知信息的图像表示。扩散模型不再为所有像素细节付费，而是在更小的 latent map 上学习生成。

### 为什么这能省计算

假设原图大小是 `H x W x 3`，autoencoder 下采样因子是 `f`，latent 空间大小大致是：

```text
h x w x c
where h = H / f, w = W / f
```

空间分辨率减少为：

```text
H * W -> (H / f) * (W / f) = H * W / f^2
```

所以 `f=4` 时，空间位置数约减少 `16` 倍；`f=8` 时，空间位置数约减少 `64` 倍。实际成本还受 channel 数、UNet 宽度、attention 位置等影响，但主要节省来自空间尺寸变小。

## 10. 关键公式三：条件 LDM

论文把条件输入 `y` 加入 denoising network：

$$
L_{LDM}
:=
\mathbb{E}_{\mathcal{E}(x),y,\epsilon\sim\mathcal{N}(0,1),t}
\Big[
\|\epsilon-\epsilon_{\theta}(z_t,t,\tau_{\theta}(y))\|_2^2
\Big].
$$

其中：

```text
y: 条件输入
tau_theta(y): 条件编码器输出
epsilon_theta(...): 接收 latent、timestep 和 condition 的 UNet
```

### 推导

无条件 LDM 是：

$$
\epsilon_\theta(z_t,t)\approx\epsilon.
$$

条件 LDM 要学习的是：

$$
p(z|y).
$$

所以 denoising network 需要知道条件 `y`：

$$
\epsilon_\theta(z_t,t,\tau_\theta(y))\approx\epsilon.
$$

训练目标仍然是噪声预测 MSE，只是模型输入多了条件表示。

### 直觉

条件输入不是直接生成图像，而是改变每一步去噪的方向。

例如 text-to-image：

```text
prompt = "a watercolor painting of a chair that looks like an octopus"
tau_theta(prompt) -> text representation
UNet 在每一步 denoising 时通过 cross-attention 查询这个 text representation
最终 latent 被推向符合 prompt 的图像区域
```

## 11. Cross-Attention 条件机制

论文使用 cross-attention 把各种条件接入 UNet：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)\cdot V.
$$

其中：

$$
Q=W_Q^{(i)}\cdot\varphi_i(z_t),
$$

$$
K=W_K^{(i)}\cdot\tau_\theta(y),
$$

$$
V=W_V^{(i)}\cdot\tau_\theta(y).
$$

### 直觉

最重要的一句话：

```text
Q 来自图像 latent 的 UNet feature；
K 和 V 来自条件编码器。
```

也就是说，图像生成过程中的每个空间位置都可以去“查询”条件信息。

对于文本条件：

```text
UNet feature: 当前 latent 图像的空间位置
text tokens: prompt 中的词或子词表示
cross-attention: 每个图像位置决定应该关注哪些 token
```

这比简单拼接更适合文本，因为文本不是一个和图像像素对齐的二维 map。

## 12. Concatenation 与 Cross-Attention 的区别

LDM 有两类条件接入方式。

第一类是 concatenation。

适合空间对齐条件：

```text
semantic map
mask
low-resolution image
corrupted image
```

这些条件本身有空间结构，可以 resize 到 latent 分辨率后和 `z_t` 拼接。

第二类是 cross-attention。

适合 token-like 或非空间对齐条件：

```text
text prompt
class label
layout token
object description
```

这些条件不天然对应图像中的每个像素位置，需要通过 attention 学习“哪些图像位置应该看哪些条件 token”。

## 13. 训练过程

无条件 LDM 训练可以写成：

```text
repeat:
    sample image x from dataset
    encode z = E(x)
    sample timestep t
    sample epsilon ~ N(0, I)
    construct z_t = alpha_t * z + sigma_t * epsilon
    predict epsilon_hat = epsilon_theta(z_t, t)
    loss = ||epsilon - epsilon_hat||^2
    update epsilon_theta
```

条件 LDM 训练可以写成：

```text
repeat:
    sample paired data (x, y)
    encode image z = E(x)
    encode condition c = tau_theta(y)
    sample timestep t
    sample epsilon ~ N(0, I)
    construct z_t = alpha_t * z + sigma_t * epsilon
    predict epsilon_hat = epsilon_theta(z_t, t, c)
    loss = ||epsilon - epsilon_hat||^2
    update epsilon_theta and tau_theta
```

训练时已知：

```text
x
z = E(x)
epsilon
t
y
```

模型要学会预测：

```text
epsilon
```

## 14. 推理 / 采样过程

采样时没有真实图像 `x`，也没有真实 latent `z`。只有随机噪声：

```text
z_T ~ N(0, I)
```

无条件采样：

```text
initialize z_T from Gaussian noise

for t = T ... 1:
    predict epsilon_hat = epsilon_theta(z_t, t)
    update z_t -> z_{t-1}

decode x = D(z_0)
return x
```

条件采样：

```text
encode condition c = tau_theta(y)
initialize z_T from Gaussian noise

for t = T ... 1:
    predict epsilon_hat = epsilon_theta(z_t, t, c)
    update z_t -> z_{t-1}

decode x = D(z_0)
return x
```

训练和采样的核心差别：

```text
训练时：
    干净 latent z 已知，噪声 epsilon 已知。

采样时：
    干净 latent z_0 未知，只能从 z_T 逐步反推。
```

## 15. f 是整篇论文的关键旋钮

`f` 是 autoencoder 的空间下采样因子：

```text
f = H / h = W / w
```

如果 `f` 太小，例如 `f=1` 或 `f=2`：

```text
latent space 接近 pixel space
计算仍然很贵
diffusion model 还要处理大量感知上不重要的细节
```

如果 `f` 太大，例如 `f=32`：

```text
autoencoder 压缩过猛
latent 丢掉太多感知细节
diffusion model 再强也无法恢复已经丢失的信息
```

论文的实验结论是：

```text
LDM-4 和 LDM-8 通常位于质量与效率的较好折中点。
```

这也是 Fig. 6 和 Fig. 7 的核心含义。

## 16. 感知压缩 vs 语义压缩

论文 Fig. 2 的关键不是图片，而是一个分工：

```text
感知压缩：
    去掉人眼不敏感的细节
    由 autoencoder 完成

语义压缩 / 语义建模：
    学习对象、布局、纹理、风格、类别和条件关系
    由 diffusion model 完成
```

如果没有 first-stage autoencoder，pixel-space diffusion 既要做感知压缩，又要做语义生成，训练和采样都会浪费计算。

如果 autoencoder 压缩过强，感知压缩会侵入语义和细节，导致生成上限被破坏。

LDM 的目标是找到中间点：

```text
压缩足够强：显著省计算
压缩足够轻：不破坏图像感知质量
```

## 17. 实验怎么读

不要把实验读成指标列表。每组实验都在回答一个设计问题。

### 17.1 压缩因子实验

问题：

```text
latent space 到底应该压缩多少？
```

论文比较 `f in {1, 2, 4, 8, 16, 32}`。

结论：

```text
f 太小：训练慢，像素空间成本仍然存在。
f 太大：压缩瓶颈明显，质量上限下降。
f=4/8：通常是较好的折中。
```

这组实验是整篇论文最重要的因果证据。

### 17.2 无条件图像生成

问题：

```text
从 pixel space 移到 latent space 后，基础生成质量会不会崩？
```

论文在 CelebA-HQ、FFHQ、LSUN-Churches、LSUN-Bedrooms 等数据集上评估 FID、Precision、Recall。

结论：

```text
LDM 在多个数据集上保持竞争力，有些场景达到或接近 state of the art。
这说明 latent diffusion 不是只适合条件生成，也能作为通用生成 prior。
```

### 17.3 Text-to-Image 与多模态条件

问题：

```text
cross-attention 是否足以让 diffusion model 接入文本等非图像条件？
```

论文使用 LAION 训练 text-to-image LDM，并在 MS-COCO 上评估。

结论：

```text
cross-attention 让 LDM 可以自然接入文本编码器；
classifier-free guidance 进一步显著提升样本质量。
```

这里已经能看到后续 Stable Diffusion 的核心结构：

```text
text encoder
  -> cross-attention in latent UNet
  -> latent diffusion sampling
  -> decoder to image
```

### 17.4 Super-Resolution

问题：

```text
LDM 是否能处理图像到图像任务，而不只是从噪声生成图像？
```

做法：

```text
把 low-resolution image 作为 spatial condition 拼接到 UNet 输入。
```

结论：

```text
LDM-SR 在 FID 上表现强，但 PSNR / SSIM 不一定最高。
```

这个结果提醒我们：感知质量指标和像素级重建指标并不总是一致。图像回归模型可能 PSNR 高，但视觉上更模糊。

### 17.5 Inpainting

问题：

```text
latent diffusion 能否与专门的图像修复模型竞争？
```

结论：

```text
LDM-4 相比 pixel-space conditional DM 至少有 2.7x speed-up，
同时 FID 也更好。
```

这说明 latent space 的收益不仅是无条件生成，也包括 masked image modeling 这类空间条件任务。

## 18. 与 DDPM 笔记的连接

如果已经读过 DDPM，可以这样理解 LDM：

```text
DDPM 解决的是：
    如何把生成建模转化成逐步去噪？

LDM 解决的是：
    逐步去噪应该发生在什么空间里，才更便宜、更可扩展？
```

DDPM 的核心变量是：

```text
x_0, x_t, epsilon, epsilon_theta(x_t, t)
```

LDM 的核心替换是：

```text
x_0 -> z_0 = E(x)
x_t -> z_t
epsilon_theta(x_t, t) -> epsilon_theta(z_t, t, condition)
```

所以 LDM 可以被看成：

```text
DDPM objective
  + perceptual autoencoder
  + latent-space UNet
  + cross-attention conditioning
```

## 19. 与 EDM 笔记的连接

EDM 关注的是扩散模型内部的设计空间：

```text
noise schedule
sampler
preconditioning
loss weighting
sigma distribution
```

LDM 关注的是扩散模型外部的表示空间：

```text
pixel space
vs
latent space
```

两篇论文互补：

```text
LDM 问：在哪个空间里扩散？
EDM 问：扩散和采样的每个组件如何设计？
```

如果把它们合起来看，一个现代 text-to-image 系统大致由两层设计组成：

```text
representation layer:
    autoencoder decides pixel space vs latent space

diffusion layer:
    denoiser, noise schedule, sampler, guidance decide generation dynamics
```

## 20. 最小心智模型

一句话：

```text
LDM = 先把图像压缩成仍有空间结构的 latent map，再在这个 latent map 上跑 DDPM。
```

三步心智模型：

```text
1. Autoencoder 学会把图像变成便宜但有意义的 z。
2. Diffusion model 学会在 z 空间从噪声还原出自然图像 latent。
3. Decoder 把生成出的 z 还原成 RGB 图像。
```

条件生成心智模型：

```text
condition y 不直接画图；
它通过 concat 或 cross-attention 改变每一步 denoising 的方向。
```

最重要的 trade-off：

```text
latent 越小，扩散越便宜；
latent 太小，图像细节上限越低。
```

## 21. 本次疑问记录

后续阅读时建议持续记录这些问题：

1. `z_t` 和 DDPM 里的 `x_t` 是否完全同构？
   - 训练目标同构，但所在空间不同；`z_t` 是 autoencoder latent 的带噪版本。

2. LDM 的 autoencoder 是否也参与 diffusion 训练？
   - 论文主线是先训练 autoencoder，再在固定 latent space 里训练 diffusion model。这样避免同时权衡重建质量和 latent prior 学习。

3. 为什么不把图像压得越小越好？
   - 因为压缩会丢信息。扩散模型只能学习 latent 分布，无法恢复 encoder 已经丢掉的细节。

4. Cross-attention 中为什么 `Q` 来自图像、`K/V` 来自文本？
   - 因为生成中的图像 latent 需要主动查询条件信息：当前空间位置应该关注 prompt 的哪些 token。

5. LDM 为什么仍然比 GAN 慢？
   - 因为 LDM 仍然是多步 denoising 采样，需要多次网络前向；latent space 只降低每次前向的成本，不消除 sequential sampling。

## 22. 盲点与限制

最重要的盲点是 autoencoder bottleneck。

```text
LDM 的生成上限受 first-stage autoencoder 的重建能力限制。
```

如果任务要求精确像素级信息，例如小字、医学影像、严格几何结构、可验证细节，latent compression 可能损伤关键证据。

第二个限制是采样仍然 sequential。

```text
LDM 比 pixel-space diffusion 便宜，
但仍然通常比 GAN 采样慢。
```

第三个限制是数据与社会影响。

论文明确提到生成模型可能被用于深度伪造、误导性内容、隐私泄露和偏见放大。LDM 降低训练和推理成本，一方面提高可访问性，另一方面也降低了滥用门槛。

## 23. 读完后应能回答的问题

读懂这篇论文后，应能回答：

1. 为什么 pixel-space diffusion 成本高？
2. 为什么 LDM 不等于简单降采样图片后再扩散？
3. `f=4` 和 `f=8` 为什么经常是好折中？
4. Eq. 2 和 DDPM 的 Eq. 1 本质差别在哪里？
5. Cross-attention 如何让文本条件影响图像生成？
6. 为什么 spatial conditioning 可以用 concatenation？
7. 为什么 LDM 的质量上限受 autoencoder 影响？
8. LDM、DDPM、EDM 分别解决扩散模型的哪个层面问题？

