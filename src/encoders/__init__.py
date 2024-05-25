from .base import BaseEncoder
from .fft import FFTEncoder
from .melspec import MelSpecEncoder
from .raw import RawEncoder
from .sample import SampleEncoder


def get_sound_encoder(encoder_name: str, n_frame: int) -> BaseEncoder:
    encoder = None
    if encoder_name == 'conv1d':
        encoder = RawEncoder(frame_skip=n_frame)
    elif encoder_name == 'fft':
        encoder = FFTEncoder(frame_skip=n_frame)
    elif encoder_name == 'mel':
        encoder = MelSpecEncoder(frame_skip=n_frame)
    else:
        encoder = SampleEncoder()
    return encoder