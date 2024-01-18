import subprocess

from separation import separate
import soundfile as sf
SAMPLING_RATE = 22050

# covnert wav
mp3_path = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/new_unet/notmusdb/data/mp3/eventhorizon.mp3'
wav_path = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/new_unet/notmusdb/data/wav/eventhorizon.wav'
subprocess.call(['ffmpeg', '-i', mp3_path, wav_path])

# Separation
sep_dir = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/new_unet/notmusdb/separate/'
wav_path = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/new_unet/notmusdb/data/wav/eventhorizon.wav'
model_path = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/new_unet/outputs/model-475.pth'

separated = separate(wav_path, model_path)
print(separated.T.shape)

# 0 : drum 1 : bass 2: other  3 : vocal
# Save to wav file
for i in range(4):
    sf.write(sep_dir+'source'+str(i)+'.wav', separated.T[:,i], SAMPLING_RATE)