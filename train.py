import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from utils._utils import make_data_loader
# from model import BaseModel
#from model import PretrainModel
from model import ResNet50
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
#from torchvision.models import resnet101, ResNet101_Weights
import torchvision

command = "pip install efficientnet-pytorch"   

os.system(command)                                     

from efficientnet_pytorch import EfficientNet
from copy import deepcopy

writer = SummaryWriter()

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, model):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    total_num = 1
    best_acc = 0
    best_acc_model = None 
    counter = 0
    patience = 10
    
    for epoch in range(args.epochs):
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(data_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)          
            optimizer.zero_grad()

            output = model(image)

            label = label.squeeze()
            loss = criterion(output, label)
            writer.add_scalar("Loss/train", loss, total_num)
            loss.backward()
            optimizer.step()

            total_num += 1

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)
        scheduler.step()
        writer.flush()

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        writer.add_scalar("Acc/train", epoch_train_acc, total_num)

        if epoch_train_acc > best_acc:
            best_acc = epoch_train_acc
            best_acc_model = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        
        torch.save(best_acc_model, f'{args.save_path}/model27.pth')

        if counter == patience:
            print("early stopping!")
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    parser.add_argument('--load-model', default='checkpoints/model26.pth', help="Model's state_dict")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """
    

    # hyperparameters
    args.epochs = 500
    args.learning_rate = 0.001
    args.batch_size = 64

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, _ = make_data_loader(args)


    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=10)
    model = model.to(device)


    # Training The Model
    train(args, train_loader, model)
    writer.close()