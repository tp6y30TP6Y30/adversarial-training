import os
from os.path import join
from model import resnest
from dataloader import SelfSupervisedData, SupervisedData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Setup the training settings.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train-img-path', default = './dataset/data/p1_data/train_50/',
                        help = "The directory that contains training img files")

    parser.add_argument('--valid-img-path', default = './dataset/data/p1_data/val_50/',
                        help = "The directory that contains validation img files")

    parser.add_argument('--self-supervised', action = 'store_true',
                        help = "self-supervised training or not")

    parser.add_argument('--load-pretrain', default = 0, type = int,
                        help = "load which self-supervised pretrain model")

    parser.add_argument('--test', action = 'store_true',
                        help = "test mode or others (train, valid)")

    parser.add_argument('--save-path', default = './save_models/', type = str,
                        help = "The directory that restores the model")

    parser.add_argument('--pretrain-epochs', default = 20, type = int,
                        help = "The total pretraining epochs")

    parser.add_argument('--epochs', default = 20, type = int,
                        help = "The total training epochs")

    parser.add_argument('--batchsize', default = 40, type = int,
                        help = "The training batchsize")

    parser.add_argument('--lr', default = 1e-3, type = float,
                        help = "The training learning rate")

    args = parser.parse_args()
    return args

def train(self_supervised, dataset, model, device, optimizer, criterion, args, epoch, epochs):
    dataloader = DataLoader(dataset, batch_size = args.batchsize, shuffle = True, num_workers = 4, pin_memory = True)
    model.train()
    loss_data = 0
    for img, label in tqdm(dataloader, ncols = 90, desc = '[Train: {}] {:d}/{:d}'.format('self_supervised' if self_supervised else 'supervised', epoch, epochs)):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        loss_data += loss.item()
        optimizer.step()
    print('classify loss: ', loss_data / len(dataloader))

def valid(self_supervised, dataset, model, device, optimizer, criterion, args, epoch, epochs):
    dataloader = DataLoader(dataset, batch_size = args.batchsize, shuffle = False, num_workers = 4, pin_memory = True)
    model.eval()
    loss_data = 0
    hit = 0
    with torch.no_grad():
        for img, label in tqdm(dataloader, ncols = 90, desc = '[Valid: {}] {:d}/{:d}'.format('self_supervised' if self_supervised else 'supervised', epoch, epochs)):
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = criterion(pred, label)
            loss_data += loss.item()
            hit += sum(torch.argmax(pred, dim = 1) == label)
        print('classify loss: ', loss_data / len(dataloader))
        print('Accuracy: {:.2f}%'.format(hit.true_divide(len(dataloader) * args.batchsize) * 100))
        print()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_path, exist_ok = True)
    train_dataset = SelfSupervisedData(args.train_img_path) if args.self_supervised else SupervisedData(args.train_img_path)
    valid_dataset = SelfSupervisedData(args.valid_img_path) if args.self_supervised else SupervisedData(args.valid_img_path)
    num_classes = 4
    model = resnest.resnest50(num_classes = num_classes).to(device).float()
    optimizer = torch.optim.Adam(filter(lambda param : param.requires_grad, model.parameters()), lr = args.lr, betas = (0.5, 0.9))   
    criterion = nn.CrossEntropyLoss().to(device).float()
    if args.test:
        pass
    else:
        if args.self_supervised:
            for epoch in range(1, args.pretrain_epochs + 1):
                train(args.self_supervised, train_dataset, model, device, optimizer, criterion, args, epoch, args.pretrain_epochs)
                valid(args.self_supervised, valid_dataset, model, device, optimizer, criterion, args, epoch, args.pretrain_epochs)
                if epoch % 5 == 0: torch.save(model.state_dict(), args.save_path + '{}_ss.ckpt'.format(epoch))
        else:
            if args.load_pretrain: model.load_state_dict(torch.load(join(args.save_path, '{}_ss.ckpt'.format(str(args.load_pretrain)))))
            model.fc = nn.Linear(model.fc.in_features, 50).to(device).float()
            for epoch in range(1, args.epochs + 1):
                train(args.self_supervised, train_dataset, model, device, optimizer, criterion, args, epoch, args.epochs)
                valid(args.self_supervised, valid_dataset, model, device, optimizer, criterion, args, epoch, args.epochs)
                if epoch % 5 == 0: torch.save(model.state_dict(), args.save_path + '{}_s{}.ckpt'.format(epoch, '_pretrain' if args.load_pretrain else ''))

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()