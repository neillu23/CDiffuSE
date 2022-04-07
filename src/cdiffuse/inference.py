# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import librosa
import torchaudio
import random
from argparse import ArgumentParser
import pdb
from sympy import symbols, Eq, solve

from torch.utils.tensorboard import SummaryWriter
from params import AttrDict, params as base_params
from model import DiffuSE

from os import path
from glob import glob
from tqdm import tqdm
random.seed(23)

models = {}


def load_model(model_dir=None, args=None, params=None, device=torch.device('cuda')):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    
    model = DiffuSE(args, AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model
  model = models[model_dir]
  model.params.override(params)
      
  return model
      

def inference_schedule(model, fast_sampling=False):
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    # print("beta",beta)
    # print("alpha_cum",talpha_cum)
    # print("gamma_cum",alpha_cum)
    sigmas = [0 for i in alpha]
    for n in range(len(alpha) - 1, -1, -1): 
      sigmas[n] = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    m = [0 for i in alpha] 
    gamma = [0 for i in alpha] 
    delta = [0 for i in alpha]  
    d_x = [0 for i in alpha]  
    d_y = [0 for i in alpha]  
    delta_cond = [0 for i in alpha]  
    delta_bar = [0 for i in alpha] 
    c1 = [0 for i in alpha] 
    c2 = [0 for i in alpha] 
    c3 = [0 for i in alpha] 
    oc1 = [0 for i in alpha] 
    oc3 = [0 for i in alpha] 
    
    for n in range(len(alpha)):
      m[n] = min(((1- alpha_cum[n])/(alpha_cum[n]**0.5)),1)**0.5
    m[-1] = 1    

    for n in range(len(alpha)):
      delta[n] = max(1-(1+m[n]**2)*alpha_cum[n],0)
      gamma[n] = sigmas[n]

    for n in range(len(alpha)):
      if n >0: 
        d_x[n] = (1-m[n])/(1-m[n-1]) * (alpha[n]**0.5)
        d_y[n] = (m[n]-(1-m[n])/(1-m[n-1])*m[n-1])*(alpha_cum[n]**0.5)
        delta_cond[n] = delta[n] - (((1-m[n])/(1-m[n-1])))**2 * alpha[n] * delta[n-1]
        delta_bar[n] = (delta_cond[n])* delta[n-1]/ delta[n]
      else:
        d_x[n] = (1-m[n])* (alpha[n]**0.5)
        d_y[n]= (m[n])*(alpha_cum[n]**0.5)
        delta_cond[n] = 0
        delta_bar[n] = 0 
    # print("m",np.array(m))
    # print("delta",np.array(delta))
    # print("d_x",np.array(d_x))
    # print("d_y",np.array(d_y))
    # print("delta_cond",np.array(delta_cond))
    # print("delta_bar",np.array(delta_bar))
    # print("sigma",np.array(sigmas))

    for n in range(len(alpha)):
      oc1[n] = 1 / alpha[n]**0.5
      oc3[n] = oc1[n] * beta[n] / (1 - alpha_cum[n])**0.5
      if n >0:
        c1[n] = (1-m[n])/(1-m[n-1])*(delta[n-1]/delta[n])*alpha[n]**0.5 + (1-m[n-1])*(delta_cond[n]/delta[n])/alpha[n]**0.5
        c2[n] = (m[n-1] * delta[n] - (m[n] *(1-m[n]))/(1-m[n-1])*alpha[n]*delta[n-1])*(alpha_cum[n-1]**0.5/delta[n])
        c3[n] = (1-m[n-1])*(delta_cond[n]/delta[n])*(1-alpha_cum[n])**0.5/(alpha[n])**0.5
      else:
        c1[n] = 1 / alpha[n]**0.5
        c3[n] = c1[n] * beta[n] / (1 - alpha_cum[n])**0.5
    return alpha, beta, alpha_cum,sigmas, T, c1, c2, c3, delta, delta_bar
      

def predict(spectrogram, model, noisy_signal, alpha, beta, alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar, device=torch.device('cuda')):
  with torch.no_grad():
    # Expand rank 2 tensors by adding a batch dimension.
    if len(spectrogram.shape) == 2:
      spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)
    
    
    audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    noisy_audio = torch.zeros(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    noisy_audio[:,:noisy_signal.shape[0]] = torch.from_numpy(noisy_signal).to(device)
    audio = noisy_audio
    gamma = [0.2]
    for n in range(len(alpha) - 1, -1, -1):
      if n > 0:
        predicted_noise =  model(audio, spectrogram, torch.tensor([T[n]], device=audio.device)).squeeze(1)
        audio = c1[n] * audio + c2[n] * noisy_audio - c3[n] * predicted_noise
        noise = torch.randn_like(audio)
        newsigma= delta_bar[n]**0.5 
        audio += newsigma * noise
      else:
        predicted_noise =  model(audio, spectrogram, torch.tensor([T[n]], device=audio.device)).squeeze(1)
        audio = c1[n] * audio - c3[n] * predicted_noise
        audio = (1-gamma[n])*audio+gamma[n]*noisy_audio
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate


def main(args):
  if args.se:
    base_params.n_mels = 513
  else:
    base_params.n_mels = 80
  specnames = []
  print("spectrum:",args.spectrogram_path)
  print("noisy_signal:",args.wav_path)
  for path in args.spectrogram_path:
    specnames += glob(f'{path}/*.wav.spec.npy', recursive=True)
  
  model = load_model(model_dir=args.model_dir ,args=args)
  alpha, beta, alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar = inference_schedule(model, fast_sampling=args.fast)


  output_path = os.path.join(args.output, specnames[0].split("/")[-2])
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  for spec in tqdm(specnames):
    spectrogram = torch.from_numpy(np.load(spec))
    noisy_signal, _ = librosa.load(os.path.join(args.wav_path,spec.split("/")[-1].replace(".spec.npy","")),sr=16000)
    wlen = noisy_signal.shape[0]
    audio, sr = predict(spectrogram, model, noisy_signal, alpha, beta, alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar)
    audio = audio[:,:wlen]
    # audio = snr_process(audio,noisy_signal)
    output_name = os.path.join(output_path, spec.split("/")[-1].replace(".spec.npy", ""))
    torchaudio.save(output_name, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('spectrogram_path', nargs='+',
      help='space separated list of directories from spectrogram file generated by diffwave.preprocess')
  parser.add_argument('wav_path',
      help='input noisy wav directory')
  parser.add_argument('--output', '-o', default='output/',
      help='output path name')
  parser.add_argument('--fast', dest='fast', action='store_true',
      help='fast sampling procedure')
  parser.add_argument('--full', dest='fast', action='store_false',
      help='fast sampling procedure')
  parser.add_argument('--se', dest='se', action='store_true')
  parser.add_argument('--se_pre', dest='se', action='store_false')
  parser.add_argument('--voicebank', dest='voicebank', action='store_true')
  parser.set_defaults(se=True)
  parser.set_defaults(fast=True)
  parser.set_defaults(voicebank=False)
  main(parser.parse_args())
