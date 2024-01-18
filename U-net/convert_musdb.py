# MUSDB 속성

#- __Track.name__ :  the track name, consisting of Track.artist and Track.title.
#- __Track.path__ : the absolute path of the mixture which might be handy to process with external applications.
#- __Track.audio__ : stereo mixture as an numpy array of shape (nb_samples, 2).
#- __Track.rate__ : the sample rate of the mixture.
#- __Track.sources__ : a dictionary of sources used for this track.
#- __Track.stems__ : an numpy tensor of all five stereo sources of shape (5, nb_samples, 2). The stems are always in the following order: ['mixture', 'drums', 'bass', 'other', 'vocals'],
#- __Track.targets__ : a dictionary of targets provided for this track.  ('mixture', 'drums', 'bass', 'other', 'vocals', 'accompaniment', 'linear_mixture')

import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf


sr = 22050
origin_dataset_dir = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/data/MUSDB18'
new_dataset_dir = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/data/convertMUSDB'

if not os.path.isdir(new_dataset_dir):
    os.mkdir(new_dataset_dir)

os.mkdir(os.path.join(new_dataset_dir, 'train'))
os.mkdir(os.path.join(new_dataset_dir, 'test'))

with tempfile.TemporaryDirectory() as tmpdir:
    for subdir in ('train', 'test'):
        origin_dir = os.path.join(origin_dataset_dir, subdir)
        files = [f for f in os.listdir(origin_dir) if os.path.splitext(f)[1] == '.mp4']
        
        for file in files:
            path = os.path.join(origin_dir, file)
            name = os.path.splitext(file)[0]
            wav_data = []

            # concatenate all channels to a single .wav file
            for ch in range(5):
                temp_fn = f'{name}.{ch}.wav'
                out_path = os.path.join(tmpdir, temp_fn)
                subprocess.run(['ffmpeg', '-i', path,'-map', f'0:{ch}', out_path])
                sound, _ = librosa.load(out_path, sr=sr, mono=True)
                wav_data.append(sound)
            wav_data = np.stack(wav_data, axis=1)
            out_path = os.path.join(new_dataset_dir, subdir, f'{name}.wav')
            sf.write(out_path, wav_data, sr)