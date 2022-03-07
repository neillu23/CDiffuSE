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
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from params import params

random.seed(23)



def main(args):
  ids = []
  with os.open(args.cvscp,"r") as rf:
    for line in rf.readlines():
      fileid = line.split()[0].split("_")[-1].split(".")[0]
      ids.append(fileid)

  for dirPath, dirNames, fileNames in os.walk(args.specdir):
    for f in fileNames:
      fn = os.path.join(dirPath, f)
      if f.split("_")[-1].split(".")[0] in ids:
        os.exec("mv {} {}".format(fn,path.join(args.specdir,"valid")))
      else:
        os.exec("mv {} {}".format(fn,path.join(args.specdir,"train")))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('specdir', 
      help='directory containing .spec files for training')
  parser.add_argument('cvscp',
      help='output directory containing .npy files for training')
  main(parser.parse_args())
