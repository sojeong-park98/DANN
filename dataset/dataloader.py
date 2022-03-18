import torch.utils.data as data
from PIL import Image
import os
import torch
from torchvision import datasets
from torchvision import transforms

def create_dataset(data_name='MNIST', batch_size=128, image_size=28):

    if data_name == 'MNIST':
        # 데이터 전처리를 위한 transformation 선언
        trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        # 소스 데이터 불러오기
        # 소스 데이터는 (MNIST) torchvision에서 패키지화하여 제공
        dataset_train = datasets.MNIST(
            root='./dataset/MNIST/',
            train=True,
            transform=trans,
            download=True
        )
        # torch.utils.data.DataLoader로 불러오는 방법 설정
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        dataset_test = datasets.MNIST(
            root='./dataset/MNIST/',
            train=False,
            transform=trans,
            download=True
        )
        dataloader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        return dataloader_train, dataloader_test

    if data_name == 'mnist_m':
        trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # 타겟 데이터 불러오기
        # 타겟 데이터는 (mnist_m) 직접 다운받아서 데이터로더 클래스를 작성하여 불러와야함
        dataset_train = custom_dataloader(
            data_root='./dataset/mnist_m/mnist_m_train',
            data_list='./dataset/mnist_m/mnist_m_train_labels.txt',
            transform=trans,
            prefix='png'
        )
        # torch.utils.data.DataLoader로 불러오는 방법 설정
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        dataset_test = custom_dataloader(
            data_root='./dataset/mnist_m/mnist_m_test',
            data_list='./dataset/mnist_m/mnist_m_test_labels.txt',
            transform=trans,
            prefix='png'
        )
        # torch.utils.data.DataLoader로 불러오는 방법 설정
        dataloader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        return dataloader_train, dataloader_test

class custom_dataloader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None, prefix='png'):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data.split('.')[0]+'.'+prefix)
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data