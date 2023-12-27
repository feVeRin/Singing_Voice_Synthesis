import soundfile as sf
import torch
import torchaudio

from u_net import UNet
from utils import padding

N_PART = 4
N_FFT = 2047
SAMPLING_RATE = 22050

cuda_check = torch.cuda.is_available()
device = torch.device(f'cuda:1' if cuda_check else 'cpu')

def separate(input_wav, model_path):
    with torch.no_grad():
        sound, sr = torchaudio.load(input_wav)
        if sr > SAMPLING_RATE:
            sound = torchaudio.functional.resample(sound, orig_freq=44100, new_freq=SAMPLING_RATE) # sampling rate fixing
        sound = sound[[0], :].to(device)

        window = torch.hann_window(N_FFT, device=device)

        # Convert it to power spectrogram, and pad it to make the number of
        # time frames to a multiple of 64 to be fed into U-NET
        sound_stft = torch.stft(sound, N_FFT, window=window, return_complex=False)
        sound_spec = sound_stft.pow(2).sum(-1).sqrt()
        sound_spec, (left, right) = padding(sound_spec)

        # Load the model
        model = UNet(N_PART)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        right = sound_spec.size(2) - right
        mask = model(sound_spec).squeeze(0)[:, :, left:right]
        separated = mask.unsqueeze(3) * sound_stft
        
        #print(separated.type(torch.complex64)[:,:,:,0].shape)
        #separated = librosa.istft(separated.cpu().numpy(), n_fft=N_FFT, window='hann', length=sound.size(-1))
        # istft requires complex tensor // forced dtype transform
        separated = torch.istft(separated.type(torch.complex64)[:,:,:,0], N_FFT, window=window, length=sound.size(-1))
        separated = separated.cpu().numpy()
    
    return separated