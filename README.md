# paper-decoder

I read, I decode, I share.

## Diffusion

- [Denoising Diffusion Probabilistic Models](<diffusion/1. Denoising Diffusion Probabilistic Models.md>)：DDPM 把生成建模转化为在不同噪声强度下训练一个去噪网络，采样时从纯噪声开始反复预测噪声并去噪。
- [Elucidating the Design Space of Diffusion-Based Generative Models](<diffusion/2. Elucidating the Design Space of Diffusion-Based Generative Models.md>)：EDM 将扩散模型拆成 denoiser、采样 ODE/SDE、数值积分、预条件化和训练噪声分布等可调设计，强调工程化分解与经验验证。
- [High-Resolution Image Synthesis with Latent Diffusion Models](<diffusion/3. High-Resolution Image Synthesis with Latent Diffusion Models.md>)：LDM 先把图像压缩到保留空间结构的 latent map，再在 latent 空间运行扩散模型，用更低计算成本实现高分辨率条件生成。
- [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](<diffusion/4. Align your Latents High-Resolution Video Synthesis with Latent Diffusion Models.md>)：Video LDM 的关键不是从零训练视频生成器，而是在强图像 LDM 的 latent 空间中加入 temporal layers、decoder、预测、插帧和超分模块来对齐时间维度。
- [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](<diffusion/5. Stable Video Diffusion Scaling Latent Video Diffusion Models to Large Datasets.md>)：SVD 以 Stable Diffusion 的图像先验为基础，通过大规模高质量视频预训练和 finetuning 学到可迁移的 motion prior。
