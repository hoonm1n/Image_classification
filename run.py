import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
#from model import BaseModel
from tqdm import tqdm
from PIL import Image

import torch.nn as nn
import torchvision

from efficientnet_pytorch import EfficientNet


class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None, fmt=':04d', extension='.jpg'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((32,32))
        data = self.transform(img)
        return data

def inference(args, data_loader, model):
    """ model inference """

    model.eval()
    preds = []
    
    
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for i, x in enumerate(pbar):
            
            image = x.to(args.device)
            
            y_hat = model(image)
            
            y_hat.argmax()

            _, predicted = torch.max(y_hat, 1)
            preds.extend(map(lambda t: t.item(), predicted))

    return preds



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--load-model', default='checkpoints/model18.pth', help="Model's state_dict")
    parser.add_argument('--batch-size', default=16, help='test loader batch size')
    parser.add_argument('--dataset', default='test_images/', help='image dataset directory')

    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # instantiate model
    # model = BaseModel()
    # model.load_state_dict(torch.load(args.load_model))
    # model.to(device)


    # model = torchvision.models.resnet18(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 클래스 수에 맞게 마지막 레이어를 변경합니다.
    # model.load_state_dict(torch.load(args.load_model))
    # model = model.to(device)


    # model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    # num_classes = 10
    # model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # 분류기 변경
    # model.load_state_dict(torch.load(args.load_model))
    # model = model.to(device)

    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=10)
    model.load_state_dict(torch.load(args.load_model))
    model = model.to(device)

    # load dataset in test image folder
    # you may need to edit transform
    test_data = ImageDataset(args.dataset, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    # write model inference
    preds = inference(args, test_loader, model)
    print(len(preds))
    label = [[i]*10 for i in range(10)]
    label = sum(label,[])
    acc = [pred == label[i] for i, pred in enumerate(preds)]
    print(acc.count(True)/100)
        
    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))