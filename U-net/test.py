import soundfile as sf
import torch
import torchaudio
import librosa

from pathlib import Path
import museval
import numpy as np
import pandas as pd
import tqdm

from u_net import UNet
from separation import separate
from utils import padding, median_nan

import warnings
warnings.filterwarnings("ignore")

SAMPLING_RATE = 22050

musdb_dir = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/data/MUSDB'
output_dir = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/new_unet/separates/'
model_path = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/new_unet/outputs/model-475.pth'

cuda_check = torch.cuda.is_available()
device = torch.device(f'cuda:1' if cuda_check else 'cpu')

def test_main():
    result = pd.DataFrame(columns=['track','SDR','ISR', 'SIR', 'SAR'])

    # parsing track path
    tracks = []
    p = Path(musdb_dir, 'test')
    test_dir = Path(musdb_dir, 'test')
    for track_path in tqdm.tqdm(p.iterdir(), disable=True):
        tracks.append(track_path)

    for track in tqdm.tqdm(tracks):
        # seaparate
        input_file = str(Path(track, 'mixture.wav'))
        separated = separate(input_file, model_path)

        output_path = Path(output_dir, 'estimates', Path(input_file).parent.name)
        output_path.mkdir(exist_ok=True, parents=True)

        # save to wav
        for i in range(4):
            sf.write(str(output_path) + '/source' + str(i)+ '.wav', separated.T[:,i], SAMPLING_RATE)

        # evaluation
        estdir = output_path
        refdir = Path(test_dir, estdir.name)
        if refdir.exists():
            ref, sr = sf.read(str(Path(refdir, 'vocals' + '.wav')), always_2d=True)
            est, sr = sf.read(str(Path(estdir, 'source3' + '.wav')), always_2d=True)

            ref = ref[:,0][None, ...]
            est = est[None, ...]

            SDR, ISR, SIR, SAR = museval.evaluate(ref, est, win=sr, hop=sr)
            values = {
                'track':estdir.name,
                "SDR": median_nan(SDR[0]),
                "ISR": median_nan(ISR[0]),
                "SIR": median_nan(SIR[0]),
                "SAR": median_nan(SAR[0])
            }
            result.loc[result.shape[0]] = values

    values = {
        'track':'sum',
        "SDR": result['SDR'].median(),
        "ISR": result['ISR'].median(),
        "SIR": result['SIR'].median(),
        "SAR": result['SAR'].median()
    }
    result.loc[result.shape[0]] = values
    print(list((result.loc[result.shape[0] - 1])[1:]))
    result.to_csv(str(output_dir)+'result.csv',index=0)

if __name__ == '__main__':
    test_main()