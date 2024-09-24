import math

import numpy as np
import torch
from typing_extensions import Dict

from src.config import ROLL_OUT
from src.dataclasses.trajector_batch import TrajectorBatch
from src.dataclasses.trajector_batch_rnn import TrajectorBatchRNN
from src.utils import magic_combine


class TrajectoryDataset():
    """
    Fast dataset for producing training batches from trajectories.
    """
    def __init__(self, trajectories: Dict, batch_size: int, device: torch.DeviceObjType, sequence_len: int, recurrent=True):
        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.sequence_len = sequence_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - sequence_len + 1, 0, ROLL_OUT)
        self.cumsum_seq_len = np.cumsum(np.concatenate((np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = batch_size
        self.recurrent = recurrent

    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self

    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.sequence_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size)
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False)
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(start_idx, bins=self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.sequence_len - 1, num=self.sequence_len, dtype=np.int64)
            self.batch_count += 1
            if self.recurrent:
                return TrajectorBatchRNN(**{key: magic_combine(value[eps_idx, series_idx], 0, 2) for key, value
                                     in self.trajectories.items() if key in TrajectorBatchRNN.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)
            return TrajectorBatch(**{key: magic_combine(value[eps_idx, series_idx], 0, 2) for key, value
                                     in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)
    
    def __len__(self):
        return math.ceil(math.ceil(self.cumsum_seq_len[-1] / self.sequence_len) / self.batch_size)
