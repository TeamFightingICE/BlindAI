from enum import Enum


class EncoderEnum(str, Enum):
    CONV1D = "conv1d"
    FFT = "fft"
    MEL = "mel"
