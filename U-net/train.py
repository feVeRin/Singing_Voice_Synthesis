import argparse
import os

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import data
from u_net import UNet
from utils import padding

N_PART = 4
N_FFT = 2047
SAMPLING_RATE = 22050

batch_size = 64
epochs = 500
interval = 25
learning_rate = 2e-3

data_dir = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/data/convertMUSDB'
output_dir = '/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/new_unet/outputs'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

cuda_check = torch.cuda.is_available()
device = torch.device(f'cuda' if cuda_check else 'cpu')

def train(model, data_loader, optimizer, device, epoch, tb_writer):
    model.train()

    total_loss = 0
    for x, t in data_loader:
        batch_size = x.size(0)
        x, t = x.to(device), t.to(device)
        y = model(x) * x.unsqueeze(1)

        loss = F.l1_loss(y, t, reduction='sum') / batch_size
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tb_writer.add_scalar('train/loss', total_loss / len(data_loader), epoch)

def test(model, test_data, device, epoch, tb_writer):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        window = torch.hann_window(N_FFT, device=device)
        for sound in test_data:
            sound = sound.to(device)
            sound_stft = torch.stft(sound, N_FFT, window=window, return_complex=False)
            sound_spec = sound_stft.pow(2).sum(-1).sqrt()
            x, t = sound_spec[0], sound_spec[1:]

            x_padded, (left, right) = padding(x)
            right = x_padded.size(1) - right
            mask = model(x_padded.unsqueeze(0)).squeeze(0)[:, :, left:right]
            y = mask * x.unsqueeze(0)
            loss = F.l1_loss(y, t, reduction='sum')
            total_loss += loss.item()

    tb_writer.add_scalar('test/loss', total_loss / len(test_data), epoch)

def train_main():
    # Dataloader
    print('Data load..')
    train_data, test_data = data.read_data(data_dir, N_FFT, 512, SAMPLING_RATE)
    train_dataset = data.RandomCropDataset(train_data, 256)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)

    # Model
    print('Model setting..')
    model = UNet(N_PART)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Tensorboard
    print('Training..')
    tb_writer = SummaryWriter()

    for epoch in range(epochs):
        train(model, train_loader, optimizer, device, epoch, tb_writer)
        if epoch % interval == 0:
            # Save the model
            test(model, test_data, device, epoch, tb_writer)
            model.cpu()
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            path = os.path.join(output_dir, f'model-{epoch}.pth')
            torch.save(state_dict, path)
            model.to(device)

    tb_writer.close()
    print('Finish..')

if __name__ == '__main__':
    train_main()