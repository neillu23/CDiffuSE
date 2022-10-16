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
import random
import torch
import librosa

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, wav_path, noisy_path, npy_paths, se, voicebank=False):
    super().__init__()
    # self.filenames = []
    self.wav_path = wav_path
    self.noisy_path = noisy_path
    self.specnames = []
    self.se = se
    self.voicebank = voicebank
    print(npy_paths,wav_path,noisy_path)
    for path in npy_paths:
      self.specnames += glob(f'{path}/*.wav.spec.npy', recursive=True)

  def __len__(self):
    return len(self.specnames)

  def __getitem__(self, idx):
    spec_filename = self.specnames[idx]
    if self.voicebank:
      spec_path = "/".join(spec_filename.split("/")[:-1])
      audio_filename = spec_filename.replace(spec_path, self.wav_path).replace(".spec.npy", "")
      noisy_filename = spec_filename.replace(spec_path, self.noisy_path).replace(".spec.npy", "")
    else:
      spec_path = "/".join(spec_filename.split("/")[:-2])+"/"
      if self.se:
        audio_filename = spec_filename.replace(spec_path, self.wav_path).replace(".wav.spec.npy", ".Clean.wav")
      else:
        audio_filename = spec_filename.replace(spec_path, self.wav_path).replace(".spec.npy", "")
      
    signal, _ = librosa.load(audio_filename, sr=16000)
    noisy_signal, _ = librosa.load(noisy_filename, sr=16000)  
      
    spectrogram = np.load(spec_filename)
    return {
        'audio': signal,
        'noisy': noisy_signal,
        'spectrogram': spectrogram.T
    }


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      # Filter out records that aren't long enough.
      if len(record['spectrogram']) < self.params.crop_mel_frames:
        del record['spectrogram']
        del record['audio']
        del record['noisy']
        continue

      start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
      end = start + self.params.crop_mel_frames
      record['spectrogram'] = record['spectrogram'][start:end].T

      start *= samples_per_frame
      end *= samples_per_frame
      record['audio'] = record['audio'][start:end]
      record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')
      record['noisy'] = record['noisy'][start:end]
      record['noisy'] = np.pad(record['noisy'], (0, (end-start) - len(record['noisy'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    noisy = np.stack([record['noisy'] for record in minibatch if 'noisy' in record])
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        'noisy': torch.from_numpy(noisy),
        'spectrogram': torch.from_numpy(spectrogram),
    }


def from_path(clean_dir, noisy_dir, data_dirs, params, se=True, voicebank=False, is_distributed=False):
  dataset = NumpyDataset(clean_dir, noisy_dir, data_dirs, se, voicebank)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
