from PIL import Image
from torch.utils import data
from torchvision.transforms import ToTensor
import os
import torch


class PictureDataset(data.Dataset):
    def __init__(self, size, data_path='data/', extension='.jpg'):
        filenames = []
        for filename in os.listdir(data_path):
            if filename.endswith(extension):
                filenames.append(filename)

        self.filenames = filenames
        self.data_path=data_path
        self.size = size


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        # Generates one sample of data
        filename = self.filenames[index]
        image = Image.open(self.data_path + filename)
        image = image.resize((self.size, self.size), Image.LANCZOS)
        image_tensor = ToTensor()(image)

        return image_tensor
