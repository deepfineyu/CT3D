import argparse
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from unet import Decoder3d

import cv2
cv2.setNumThreads(0)

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, data_prefetcher
from torch.utils.data import DataLoader



# import torch.distributed as dist





dir_img = r"./trainset/data_x"
dir_embeddings = r"./trainset/embeddings"
dir_mask = r"./trainset/data_y"
dir_checkpoint = r'./checkpoints/'
net_n_channels = 1
net_n_classes = 1



def train_net(net,
              device,
              epochs=5,
              batch_size=2,
              lr=0.0001,
              save_cp=True):

    # dataset = BasicDataset()
    # n_train = len(dataset)
    # print(n_train)
    
    '''半精度'''
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    


    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net_n_classes > 1 else 'max', patience=2)
    
    if net_n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    # criterion = nn.MSELoss()

    for epoch in range(epochs):
        # train_sampler.set_epoch(epoch) #shuffle
        epoch_loss = 0
        net.train()

        # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)
        dataset = BasicDataset(dir_img, dir_embeddings, dir_mask)
        n_train = len(dataset)
        print(n_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
        count = 0
        prefetcher = data_prefetcher(train_loader)
        imgs, embeds, true_masks = prefetcher.next()
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            while imgs is not None:
            # for batch in train_loader:
                count+=1
                # imgs, embeds, true_masks = batch
                imgs_len = imgs.size()[0]
                # imgs = imgs.view(-1, 1, 512, 512)#.cuda()
                # embeds = embeds.view(-1, 256, 64, 64)#.cuda()
                # true_masks = true_masks.view(-1, 512, 512)#.cuda().long()
                # print(embeds.min(), embeds.max())
                # print(imgs.size(), embeds.size(), true_masks.size())
                assert imgs.shape[1] == net_n_channels, \
                    f'Network has been defined with {net_n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # masks_pred = net(embeds)
                # # print(imgs.size(), embeds.size(), true_masks.size(), masks_pred.size())
                # # print("=======", masks_pred.min(), masks_pred.max())

                # loss = criterion(masks_pred, true_masks)
                # epoch_loss += loss.item()



                
                '''半精度'''
                optimizer.zero_grad()
                with autocast():
                    masks_pred = net(embeds)
                    loss = criterion(masks_pred, true_masks)                               
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                    
                    
                    

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # optimizer.zero_grad()
                # loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # optimizer.step()

                pbar.update(imgs_len)
                global_step += 1
                if global_step % 32 == 0:
                    
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                # if global_step % 1024 == 0:
                #     writer.add_images('images', imgs[:4], global_step)
                #     writer.add_images('masks/true', true_masks[:4, None], global_step)
                #     writer.add_images('masks/pred', masks_pred[:4], global_step)
                    
                        
                if global_step % 256 == 0:
                    torch.save(net.state_dict(),
                            dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                imgs, embeds, true_masks = prefetcher.next()

        if epoch%1==0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                    dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
                

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=4000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default="",
                        help='Load model from a .pth file')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()


    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')

    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = Decoder3d().cuda()
    # print(list(net.samApter3d_0.parameters()))



    if args.load:
        f_model = torch.load(args.load, map_location=torch.device('cpu'))
        net.load_state_dict({k.replace('module.',''):v for k,v in f_model.items()})
        logging.info(f'Model loaded from {args.load}')

    # net.to(device=device)
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth') #, _use_new_zipfile_serialization=False)
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
