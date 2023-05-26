from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

custom_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 밝기 조정
    #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),  # 무작위로 이동, 확대/축소, 전단 변환
    # transforms.RandomRotation(15),  # 무작위로 이미지 회전
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023,  0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023,  0.1994, 0.2010))
])

def make_data_loader(args):
    
    # Get Dataset
    train_dataset = datasets.CIFAR10(args.data, train=True, transform=custom_transform, download=True)
    test_dataset = datasets.CIFAR10(args.data, train=False, transform=transform_test, download=True)

    # Get Dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader




def make_data(args):
    
    # Get Dataset
    train_dataset = datasets.CIFAR10(args.data, train=True, transform=custom_transform, download=True)
    test_dataset = datasets.CIFAR10(args.data, train=False, transform=transform_test, download=True)

    return train_dataset, test_dataset

def make_data_loader_train(args, train_dataset, test_dataset):
    
    # Get Dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader