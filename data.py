import os
from torch.utils.data import Dataset
from PIL import Image
import torch

# low_ssim = [3, 27, 31, 32, 39, 42, 43, 50, 51, 57, 58, 61, 62, 63, 64, 66, 67, 76, 77, 80, 81, 87, 92, 97, 101, 107, 109, 111, 119, 123, 125, 127, 128, 131, 133, 136, 142, 147, 151, 152, 153, 157, 159, 162, 166, 174, 179, 182, 183, 186, 188, 191, 196, 199, 231, 232, 241, 245, 246, 249, 250, 252, 255, 268, 269, 270, 272, 274, 287, 289, 296, 297, 299, 306, 310, 314, 317, 319, 320, 321, 329, 332, 339, 345, 346, 353, 356, 365, 368, 374, 381, 391, 394, 396, 407, 411, 413, 420, 426, 427, 430, 437, 448, 450, 452, 455, 456, 458, 472, 492, 495, 498, 499]
# low_ssim = [246, 128, 407, 186, 310, 250, 162, 329, 499, 131, 255, 430, 396, 174, 182, 320, 66, 245, 381, 76, 142, 427, 272, 67, 61, 125, 317, 306, 420, 458, 51, 77, 448, 345, 297, 413, 437, 232, 394, 368, 87, 339, 268, 252, 62, 314, 492, 57, 319, 123, 159, 196, 50, 299, 498, 64, 191, 411, 426, 97, 153, 270, 353, 321, 127, 58, 374, 332, 109, 119, 179, 456, 296, 107, 43, 39, 365, 27, 151, 157]
low_ssim = [3, 133, 391, 136, 269, 274, 147, 152, 31, 32, 287, 289, 166, 42, 183, 188, 63, 450, 452, 199, 455, 80, 81, 472, 346, 92, 356, 101, 231, 111, 495, 241, 249]
high_ssim = [14, 26, 40, 47, 54, 60, 88, 109, 126, 139, 169, 223, 225, 226, 340, 344, 410, 420, 428, 429, 434, 439, 445, 479]
class CifarDataset(Dataset):
    def __init__(self, label_file, image_folder, device, transform=None):
        self.device = device
        self.image_folder = image_folder
        self.transform = transform

        # 读取标签文件
        self.images = []
        self.labels = []
        with open(label_file, "r") as file:
            for line in file:
                line = line.strip().split()
                image_name, label = line[0], int(line[1])
                # if int(image_name.split('.')[0]) not in low_ssim:
                #     continue
                self.images.append(image_name)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name, label = self.images[index], self.labels[index]

        # 使用PIL库加载图像
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path)

        # 数据预处理
        if self.transform:
            image = self.transform(image)

        # 将图像和标签移到设备上
        image = image.to(self.device) 
        label = torch.tensor(label).to(self.device)

        # 返回图像和标签
        return image, label, image_name
