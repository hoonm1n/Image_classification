from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

cinic_directory = '/home/kim/Term_DL/data/CINIC-10'



custom_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023,  0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023,  0.1994, 0.2010))
])

def make_data_loader(args):
    
    # Get Dataset
    cifar_dataset = datasets.CIFAR10(args.data, train=True, transform=custom_transform, download=True)
    cifar_test_dataset = datasets.CIFAR10(args.data, train=False, transform=transform_test, download=True)

    cinic_dataset = datasets.ImageFolder(root=cinic_directory + '/train', transform=custom_transform)
    cinic_test_dataset = datasets.ImageFolder(root=cinic_directory + '/test', transform=transform_test)


    # Concat Dataset
    combined_dataset = torch.utils.data.ConcatDataset([cifar_dataset, cinic_dataset])
    combined_test_dataset = torch.utils.data.ConcatDataset([cifar_test_dataset, cinic_test_dataset])


    # Get Dataloader
    train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(combined_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
