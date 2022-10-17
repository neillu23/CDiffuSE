# CDiffuSE
[Apache License 2.0](https://github.com/neillu23/CDiffuSE/blob/main/LICENSE)

CDiffuSE leverages recent advances in diffusion probabilistic models, and proposes a novel speech enhancement algorithm that incorporates characteristics of the observed noisy speech signal into the diffusion and reverse processes. More specifically, we propose a generalized formulation of the diffusion probabilistic model named conditional diffusion probabilistic model that, in its reverse process, can adapt to non-Gaussian real noises in the estimated speech signal.
[Conditional Diffusion Probabilistic Model for Speech Enhancement](https://arxiv.org/abs/2202.05256).

## Audio samples
[16 kHz audio samples](https://github.com/neillu23/CDiffuSE/tree/main/Sample%20Files)

## Pretrained models 
[CDiffuSE Pretrained models](https://drive.google.com/drive/folders/1QQCiJSc8yrXvuCnyA8vrRplPiStuoD6M?usp=sharing) are released.
The large model is the same as described in the paper. The base model was trained without the mel-spectrum pre-training phase, which is a little better than the ones on the paper.

### Training
Before you start training, you'll need to prepare a training dataset. The default dataset is VOICEBANK-DEMAND dataset. You can **download them from [VOICEBANK-DEMAND](https://doi.org/10.7488/ds/2117) and resample it to 16k Hz**. By default, this implementation assumes a sample rate of 16 kHz. If you need to change this value, edit [params.py](https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/params.py).

You need to set the output path and data path under path.sh

```
output_path=[path_to_output_directory]
voicebank=[path_to_voicebank_directory]
```

Usage:
Train SE model
```
./train.sh [stage] [model_directory]
```

Train SE model based on a pre-trained model.
```
./train.sh [stage] [model_directory] [pretrained_model_directory]/weights-[ckpt].pt
```

Note that the pre-training step with clean Mel-Spectrum conditioners is no longer needed in CDiffuSE. A randomly initialized CDiffuSE performs as well as one initialized from pre-trained parameters.

#### Multi-GPU training
By default, this implementation uses as many GPUs in parallel as returned by [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count). You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module.

### Validatoin and Inference API

Usage:
```
./valid.sh [stage] [model name] [checkpoint id] 
./inference.sh [stage] [model name] [checkpoint id]
```

The code of CDiffuSE is developed based on the code of [Diffwave](https://github.com/lmnt-com/diffwave) 

## References
- [Conditional Diffusion Probabilistic Model for Speech Enhancement](https://arxiv.org/abs/2202.05256)
- [A Study on Speech Enhancement Based on Diffusion Probabilistic Model](https://arxiv.org/abs/2107.11876)
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
