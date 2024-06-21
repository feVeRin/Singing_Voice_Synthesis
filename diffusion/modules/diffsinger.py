import torch
import torch.nn as nn
from modules import Conv1d
from diffusion import Diffusion
from fastspeech import Duration_Positional_Encoding, FFT

class Encoder(nn.Moduele):
    def __init__(self, params):
        self.params = params

        ####
        self.token_emb = nn.Embedding(num_embeddings=self.params.Tokens, embedding_dim=self.params.Encoder.Size)
        self.midi_emb = nn.Embedding(num_embeddings=self.params.Notes, embedding_dim=self.params.Encoder.Size)
        self.dur_emb = Duration_Positional_Encoding(num_embeddings=self.params.Duration, embedding_dim=self.params.Encoder.Size) #pos_emb

        self.linear = Conv1d(in_channels= self.hp.Encoder.Size,out_channels= self.feature_size,
                             kernel_size= 1,bias= True,w_init_gain= 'linear')

        ffts = []
        for _ in range(self.params.Encoder.ConvFFT.Stack):
            ffts.append(FFT(channels=self.params.Encoder.Size, num_head=self.params.Encoder.ConvFFT.Head, 
                kernel_size=self.params.Encoder.ConvFFT.Kernel_Size, dropout_rate=self.params.Encoder.ConvFFT.Droput)) #FFT
        self.ffts = nn.ModuleList(ffts)

        torch.nn.init.xavier_uniform_(self.token_embedding.weight)
        torch.nn.init.xavier_uniform_(self.note_embedding.weight)

    def forward(self, txt_token, music_note, duration, length):
        x = self.token_emb(txt_token) + self.midi_emb(music_note) + self.dur_emb(duration)
        x = x.permute(0,2,1) #[B, C, T]

        for fft in self.ffts:
            x = fft(x, length)
        
        lin_proj = self.linear(x) #[B, Mel_Dim, T]

        return x, lin_proj

########
class DiffSinger(nn.Module):
    def __init__(self,  params):
        super.__init__()
        self.params = params

        self.encoder = Encoder(self.params)
        self.diffusion = Diffusion(self.params) ####

    def forward(self, txt_token, music_note, duration, length, features, steps):
        enc_out, lin_proj = self.encoder(txt_token=txt_token, music_note=music_note, duration=duration,length=length)
        enc_out = torch.cat([enc_out, lin_proj], dim=1) #[B, C+Mel_dim, T]

        if not features is None or steps is None or steps == self.hp.Diffusion.Max_Step:
            diffusion_predictions, noises, epsilons = self.diffusion(
                encodings= enc_out,
                features= features,
                )
        else:
            noises, epsilons = None, None
            diffusion_predictions = self.diffusion.DDIM(
                encodings= enc_out,
                ddim_steps= steps
                )
        
        return lin_proj, diffusion_predictions, noises, epsilons

