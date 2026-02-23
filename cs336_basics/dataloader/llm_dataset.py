from torch.utils.data import Dataset
import numpy.typing as npt


class LLMDataset(Dataset):
    def __init__(self, dataset: npt.NDArray):
        self.npy_data = dataset

    def __len__(self) -> int:
        return len(self.npy_data)

    def __getitem__(self, i: int) -> int:
        return self.npy_data[i]
