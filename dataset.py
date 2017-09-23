import torch
import torch.utils.data as data

import cv2

from utils import load_img


def make_nz_tensor(split, value):
    input = torch.FloatTensor(split, 1, 1)

    value = split * value

    for i in range(split):
        if i % 2 == 0:
            input[i, 0, 0] = max(0, min(1, value))
        else:
            input[i, 0, 0] = 1 - max(0, min(1, value))
        value = value - 1.0
        
    return input


class ImageDataset(data.Dataset):

    def __init__(self, image, nz, tile_size):
        super().__init__()

        self.image = image
        self.nz = nz
        self.tile_size = tile_size

        _, self.height, self.width = self.image.size()
        assert self.height >= tile_size and self.width >= tile_size
        if self.height % tile_size != 0 or self.width % tile_size != 0:
            image = self.image.numpy().transpose(1, 2, 0)
            image = cv2.copyMakeBorder(image, 0, tile_size - (self.height % tile_size), 0, tile_size - (self.width % tile_size), cv2.BORDER_REFLECT)
            self.image = torch.from_numpy(image.transpose(2, 0, 1))

        self.max_size = max(self.width, self.height)
        
        self.x_num = self.width // tile_size
        self.y_num = self.height // tile_size


    def __getitem__(self, index):
        index_x = index % self.x_num
        index_y = index // self.x_num

        x = self.tile_size * index_x
        y = self.tile_size * index_y

        target = self.image[:,y:y+self.tile_size,x:x+self.tile_size]

        x_input = make_nz_tensor(self.nz, x / self.max_size)
        y_input = make_nz_tensor(self.nz, y / self.max_size)
        input = torch.cat((x_input, y_input))

        return input, target.clone()


    def __len__(self):
        return self.x_num * self.y_num
