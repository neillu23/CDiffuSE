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
import librosa,os
import random
import scipy
import pdb 
from itertools import repeat
import numpy as np
import torch
import torchaudio

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from params import params

random.seed(23)

def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=400, SHIFT=160, _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    ### Normalize waveform
    y = y / np.max(abs(y)) # / 2.

    D = librosa.stft(y, n_fft=FRAMELENGTH, hop_length=SHIFT,win_length=FRAMELENGTH,window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1

    return Sxx, phase, len(y)


def transform(filename,indir,outdir):
  audio, sr = torchaudio.load(filename)
  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')
  audio = torch.clamp(audio[0], -1.0, 1.0)

  mel_args = {
      'sample_rate': sr,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': sr / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)
  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    np.save(f'{filename.replace(indir,outdir)}.spec.npy', spectrogram.cpu().numpy()) 

def spec_transform(filename,indir,outdir):
    spec, _, _ = make_spectrum(filename,FRAMELENGTH=params.n_fft, SHIFT=params.hop_samples)
    np.save(f'{filename.replace(indir,outdir)}.spec.npy', spec)



def main(args):
  if args.se:
    params.n_mels = 513
  else:
    params.n_mels = 80

  if args.se or args.voicebank:
    filenames = glob(f'{args.dir}/*.wav', recursive=True)
  else:
    filenames = glob(f'{args.dir}/*.Clean.wav', recursive=True)
#   filenames=sorted(filenames)
#   random.shuffle(filenames)


  if args.se:
    with ProcessPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(spec_transform, filenames, repeat(args.dir), repeat(args.outdir)), desc='Preprocessing', total=len(filenames)))
  else:
    with ProcessPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(transform, filenames, repeat(args.dir), repeat(args.outdir)), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffuSE')
  parser.add_argument('dir', 
      help='directory containing .wav files for training')
  parser.add_argument('outdir',
      help='output directory containing .npy files for training')
  parser.add_argument('--se', dest='se', action='store_true')
  parser.add_argument('--se_pre', dest='se', action='store_false')
  parser.add_argument('--train', dest='test', action='store_false')
  parser.add_argument('--test', dest='test', action='store_true')
  parser.add_argument('--voicebank', dest='voicebank', action='store_true')
  parser.set_defaults(se=True)
  parser.set_defaults(test=False)
  parser.set_defaults(voicebank=True)
  main(parser.parse_args())
