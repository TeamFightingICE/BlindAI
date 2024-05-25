import torch
import torch.nn.functional as F
import torchaudio

from src.encoders.base import BaseEncoder


class MelSpecEncoder(BaseEncoder):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(MelSpecEncoder, self).__init__(sampling_rate, fps, frame_skip)
        self.window_size = int(self.sampling_rate * 0.025)
        self.hop_size = int(self.sampling_rate * 0.01)
        self.n_fft = int(self.sampling_rate * 0.025)
        self.n_mels = 80

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_mels=80,
            n_fft=self.n_fft,
            win_length=self.window_size,
            hop_length=self.hop_size,
            f_min=20,
            f_max=7600,
        )

        # Encoder
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)

    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        x = torch.log(self.mel_spectrogram(data) + 1e-5)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        if x.shape[-1] >= 2:
            x = self.pool(x)
        return x