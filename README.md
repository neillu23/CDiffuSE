# CDiffuSE
[![License](Apache License 2.0](https://github.com/neillu23/CDiffuSE/blob/main/LICENSE)
CDiffuSE leverages recent advances in diffusion probabilistic models, and proposes a novel speech enhancement algorithm that incorporates characteristics of the observed noisy speech signal into the diffusion and reverse processes. More specifically, we propose a generalized formulation of the diffusion probabilistic model named conditional diffusion probabilistic model that, in its reverse process, can adapt to non-Gaussian real noises in the estimated speech signal.
[Conditional Diffusion Probabilistic Model for Speech Enhancement](https://arxiv.org/abs/2202.05256).

## Audio samples
[16 kHz audio samples](https://drive.google.com/drive/u/0/folders/161St-rrq579r1VH7_fKOWDiBWi2MCzqw)

## Pretrained models will be released soon

### Training
Before you start training, you'll need to prepare a training dataset. The default dataset is VOICEBANK-DEMAND dataset. You can download them from [VOICEBANK-DEMAND](https://doi.org/10.7488/ds/2117)). By default, this implementation assumes a sample rate of 16 kHz. If you need to change this value, edit [params.py](https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/params.py).

You need to set the output path and data path under path.sh

```
output_path=path-to-output-directory
voicebank=path-to-voicebank-directory
```

Usage:
Train SE model or pretrain model with clean Mel-Spectrum conditioner
```
./train.sh [stage] [se or se_pre] [model_directory]
```

Train SE model based on the pretrain model with clean Mel-Spectrum conditioner
```
./train.sh [stage] se [model_directory] [pretrained_model_directory]/weights-[ckpt].pt

```

#### Multi-GPU training
By default, this implementation uses as many GPUs in parallel as returned by [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count). You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module.

### Validatoin and Inference API

Usage:
```
./valid.sh [stage] [checkpoint id] [se or se_pre] [model name]
./inference.sh [stage] [checkpoint id] [se or se_pre] [model name]

```

## References
- [Conditional Diffusion Probabilistic Model for Speech Enhancement](https://arxiv.org/abs/2202.05256)
- [A Study on Speech Enhancement Based on Diffusion Probabilistic Model](https://arxiv.org/abs/2107.11876)
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
