import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class CifarDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


def get_dataloader(isTrain=False):
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    cifar100_dataset = torchvision.datasets.CIFAR100(root="/Users/chaddy/datasets", train=isTrain, download=True, transform=transform_train)
    return CifarDataLoader(dataset=cifar100_dataset, batch_size=1)
