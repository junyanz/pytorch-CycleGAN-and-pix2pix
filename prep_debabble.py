import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import glob
import random
import librosa
import numpy as np
from tqdm import tqdm
#from tqdm.notebook import tqdm
import cv2

def melpectrogram(y, start_sample=0, n_samples=None, min_dB=-100,n_fft = 1024,hop_length=256,n_mels=80,fmin=40, fmax=8000):
    if n_samples is None: n_samples = len(y)
    y = y[start_sample:start_sample+n_samples]
    #FFT
    D = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    D = np.abs(D)
    #melspec
    S = librosa.feature.melspectrogram(S=D, sr=r, n_fft=n_fft, n_mels=n_mels, fmin=fmin)
    #amp to dB
    S = 20 * np.log10(np.maximum(1e-5, S))
    #normalize
    return  np.clip((S - min_dB) / -min_dB, 0, 1)

def mkpath(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

noise_gain_str = '2.5'
n_mels = 256
print('im here')
audio_directory = 'data/debabble/audio'
img_directory = f'data/debabble/mel_img{n_mels}_{noise_gain_str}'
mkpath(f'{img_directory}/A')
mkpath(f'{img_directory}/B')
#mkpath(f'{img_directory}/A/train')
#mkpath(f'{img_directory}/A/test')
#mkpath(f'{img_directory}/A/val')
#mkpath(f'{img_directory}/B/train')
#mkpath(f'{img_directory}/B/test')
#mkpath(f'{img_directory}/B/val')
files_A = glob.glob(f'{audio_directory}/A/*')

print(f'{audio_directory}/A/*')
import re
p = re.compile('speaker_([0-9]+)')
print(files_A)
for i, f in enumerate(tqdm(files_A)):
    speaker_num = int(p.search(f).group(1))
    fA = f
    fB = f.replace('/A/', '/B/').replace('_clean.wav', f'_{noise_gain_str}_noise.wav')
    yA, r = librosa.load(fA)
    yB, r = librosa.load(fB)
    melA = melpectrogram(yA, n_mels=n_mels)
    melB = melpectrogram(yB, n_mels=n_mels)
    # chop and save images
    n_specs = melA.shape[1]
    m = melA
    for j in range(n_specs // n_mels):
        slice_start = j * n_mels
        slice_end = slice_start + n_mels
        mA_slice = melA[:,slice_start:slice_end]
        mB_slice = melB[:,slice_start:slice_end]
        cv2.imwrite(f'{img_directory}/A/mel_{noise_gain_str}_{i:06d}_s{speaker_num:03d}_{j:02d}.png', mA_slice * 255.)
        cv2.imwrite(f'{img_directory}/B/mel_{noise_gain_str}_{i:06d}_s{speaker_num:03d}_{j:02d}.png', mB_slice * 255.)

raise
imgs_A = glob.glob(f'{img_directory}/A/*.png')
n_images = len(imgs_A)
random.shuffle(imgs_A)
n_val = n_images // 10
n_test = n_images // 100
print(imgs_A)
for i, fA in enumerate(tqdm(imgs_A)):
    fB = fA.replace('/A/', '/B/')
    if i < n_test:
        os.rename(fA, fA.replace('mel_0', 'test/mel_0'))
        os.rename(fB, fB.replace('mel_0', 'test/mel_0'))
    elif i < n_val:
        os.rename(fA, fA.replace('mel_0', 'val/mel_0'))
        os.rename(fB, fB.replace('mel_0', 'val/mel_0'))
    else:
        os.rename(fA, fA.replace('mel_0', 'train/mel_0'))
        os.rename(fB, fB.replace('mel_0', 'train/mel_0'))

print('ran')
