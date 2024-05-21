import torch
from PIL import Image, ImageFont
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import einops
import numpy as np
import random
from pytorch_lightning import seed_everything
from tqdm import tqdm
import argparse
import time
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current cuda device: ",torch.cuda.get_device_name(0))

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


input_dir = '/home/ubuntu/anytext_v1/AnyText/eval/training_data_v2'

def iterate_images_by_index(directory):
    index_groups = {}

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            index = filename.split('_')[-1].split('.')[0]
            naming_convention = filename.split('_')[0]  # Extract naming convention from filename
            if index in index_groups:
                if naming_convention in index_groups[index]:
                    index_groups[index][naming_convention].append(os.path.join(directory, filename))
                else:
                    index_groups[index][naming_convention] = [os.path.join(directory, filename)]
            else:
                index_groups[index] = {naming_convention: [os.path.join(directory, filename)]}

    for index, filenames in index_groups.items():
        if all(naming_convention in filenames for naming_convention in ["glyph", "pos", "rotated"]):
            yield filenames["glyph"], filenames["pos"], filenames["rotated"]
        else:
            print(f"Skipping index {index}: Missing naming conventions.")


import torchvision.transforms as T
import torch.nn as nn
loss_fn_vgg = nn.MSELoss()

pos_tensors = torch.zeros(1,1,512,512).to(device)
target_tensors = torch.zeros(1,1,1024,1024).to(device)
gly_tensors = torch.zeros(1,1,1024,1024).to(device)

for glyph_images, pos_images, target_images in iterate_images_by_index(input_dir):
    glyph_image = Image.open(glyph_images[0])
    #print("shape of glyph_image :- ", glyph_image.shape())
    pos_image = Image.open(pos_images[0])
    target_image = Image.open(target_images[0])
    gly_tensor = T.ToTensor()(glyph_image).to(device)
    #print("shape of gly_tensor :- ", gly_tensor.shape)
    #gly_tensor = gly_tensor.resize_(1,512,512)
    gly_tensor = gly_tensor.unsqueeze(0)
    #print("shape of gly_tensor :- ", gly_tensor.shape)
    gly_tensors = torch.cat((gly_tensors, gly_tensor), 0)
    pos_tensor = T.ToTensor()(pos_image).to(device)
    pos_tensor = pos_tensor.unsqueeze(0)
    pos_tensors = torch.cat((pos_tensors, pos_tensor), 0)
    target_tensor = T.ToTensor()(target_image).to(device)
    #target_tensor = target_tensor.resize_(1,512,512)
    target_tensor = target_tensor.unsqueeze(0)
    target_tensors = torch.cat((target_tensors, target_tensor), 0)

pos_tensors = pos_tensors[1:]
target_tensors = target_tensors[1:]
gly_tensors = gly_tensors[1:]
print("pos_tensors shape :- ", pos_tensors.shape)
print("target_tensors shape :- ", target_tensors.shape)
print("gly_tensors shape :- ", gly_tensors.shape)

#exit()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.glyph_block = nn.Sequential(
            nn.Conv2d(1,8,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(8,8,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(8,16,3,padding=1,stride=2),
            nn.SiLU(),
            nn.Conv2d(16,16,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(16,32,3,padding=1,stride=2),
            nn.SiLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(32,96,3,padding=1,stride=2),
            nn.SiLU(),
            nn.Conv2d(96,96,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(96,256,3,padding=1,stride=2),
            nn.SiLU(),
        )

        self.pos_block = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8,8,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(8,16,3,padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16,16,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(16,32,3,padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(32,96,3,padding=1,stride=2),
            nn.SiLU(),
        )

        self.fuse_block = zero_module(nn.Conv2d(256+96, 320, 3, padding=1))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(320, 160, 3, padding=1, stride=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(160, 80, 3, padding=1, stride=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(80, 40, 3, padding=1, stride=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 20, 3, padding=1, stride=2, output_padding=1),
            nn.SiLU(),
            nn.Conv2d(20,10,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(10,5,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(5, 1, 3, padding=1),
            #nn.SiLU()
        )

        #self.dropout = nn.Dropout(0.5)

    def forward(self, glyph_tensor, pos_tensor, target_shape):
        glyph_in = glyph_tensor
        pos_in = pos_tensor
        #print("shape of input of glyph block :-", glyph_in.shape)
        glyph_out = self.glyph_block(glyph_tensor)
        #print("shape of output of glyph block :-", glyph_out.shape)
        glyph_out_upsampled = glyph_out#.unsqueeze(1)
        #glyph_out_upsampled = torch.reshape(glyph_out_upsampled, target_shape)
        #print("shap eof input of pos block :- ", pos_tensor.shape)
        pos_out = self.pos_block(pos_tensor)
        #print("shape of output of pos block :-", pos_out.shape)
        pos_out_upsampled = pos_out#.unsqueeze(1)
        #pos_out_upsampled = torch.reshape(pos_out_upsampled, target_shape)
        #pos_out_upsampled = self.dropout(pos_out_upsampled)
        out = torch.cat([glyph_out_upsampled, pos_out_upsampled],
                        dim=1)
        #print("shape of output after concatenating output of glyph and pos block :-", out.shape)
        pred = self.fuse_block(out)
        #print("shape of output of fuse block :-", pred.shape)
        pred = self.decoder(pred)
        #print("shape of output of decoder :-", pred.shape)
        return pred, glyph_out_upsampled, glyph_in, pos_out_upsampled, pos_in

model = Model()
print("Model :- ", model)
# dict = torch.load('checkpoint.pth')
# model_dict = dict['state_dict']

# model.load_state_dict(model_dict)
#exit()
model.to(device)
import torch.nn as nn
loss_fn = nn.MSELoss()

import torch.optim as optim
# use lr scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

glyph_train = gly_tensors[:800]
pos_train = pos_tensors[:800]
glyph_val = gly_tensors[800:900]
pos_val = pos_tensors[800:900]
glyph_test = gly_tensors[900:]
pos_test = pos_tensors[900:]

target_train = target_tensors[:800]
target_val = target_tensors[800:900]
target_test = target_tensors[900:]


from torch.utils.data import Dataset, DataLoader

# change it
batch_size = 20

glyph_train_dataloader = DataLoader(dataset=glyph_train,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    shuffle=False)
pos_train_dataloader = DataLoader(dataset=pos_train,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    shuffle=False)
glyph_val_dataloader = DataLoader(dataset=glyph_val,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    shuffle=False)
pos_val_dataloader = DataLoader(dataset=pos_val,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    shuffle=False)
glyph_test_dataloader = DataLoader(dataset=glyph_test,
                                   batch_size=batch_size,
                                   num_workers=0,
                                   shuffle=False)
pos_test_dataloader = DataLoader(dataset=pos_test,
                                   batch_size=batch_size,
                                   num_workers=0,
                                   shuffle=False)

target_train_dataloader = DataLoader(dataset=target_train,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    shuffle=False)
target_val_dataloader = DataLoader(dataset=target_val,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    shuffle=False)
target_test_dataloader = DataLoader(dataset=target_test,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    shuffle=False)


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/anytext_runner_{}'.format(timestamp))

batches_per_epoch = (len(glyph_train)/batch_size)

valid_losses = []
avg_train_losses = []
avg_valid_losses = []

def train_one_epoch(epoch_index, tb_writer):
    train_losses = []
    running_loss = 0.
    #batch_no = 0.
    total_loss = 0.
    fuse_out_tensor = torch.zeros(1,1,1024,1024).to(device)
    gly_out_tensor = torch.zeros(1,256,64,64).to(device)
    pos_out_tensor = torch.zeros(1,96,64,64).to(device)

    pos_in_tensor = torch.zeros(1,1,512,512).to(device)
    target_in_tensor = torch.zeros(1,1,1024,1024).to(device)
    gly_in_tensor = torch.zeros(1,1,1024,1024).to(device)

    for i, data in enumerate(zip(glyph_train_dataloader,
                                 pos_train_dataloader,
                                 target_train_dataloader)):
        glyph, pos, target = data
        glyph = glyph.to(device)
        pos = pos.to(device)
        target = target.to(device)
        target_shape = torch.Size([batch_size, 1, 1024, 1024])
        optimizer.zero_grad()

        fuse_out, glyph_out, glyph_in, pos_out, pos_in = model(glyph, pos, target_shape) 
        loss = loss_fn(fuse_out, target)
        loss.backward()

        optimizer.step()
        print("loss :- ", loss.item())
        train_losses.append(loss.item())
        running_loss += loss.item()
        total_loss += loss.item()

        

        if(((i+1)% batch_size) == 0):
            last_loss = running_loss / batch_size # loss per batch
            print('  batch {} loss: {}'.format(i+1, last_loss))
            running_loss = 0.

    # print(" fuse_out[0] shape :- ", fuse_out[0].shape)
    # print(" fuse_out shape :- ", fuse_out.shape)
    # print(" glyph_out[0] shape :- ", glyph_out[0].shape)
    # print(" glyph_out shape :- ", glyph_out.shape)
    # print(" pos_out[0] shape :- ", pos_out[0].shape)
    # print(" pos_out shape :- ", pos_out.shape)

    fuse_out_tensor = torch.cat((fuse_out_tensor, fuse_out),0).to(device)
    fuse_out_tensor = fuse_out_tensor[1:]
    gly_out_tensor = torch.cat((gly_out_tensor, glyph_out),0).to(device)
    gly_out_tensor = gly_out_tensor[1:]
    pos_out_tensor = torch.cat((pos_out_tensor, pos_out),0).to(device)
    pos_out_tensor = pos_out_tensor[1:]

    # print("shape of pos_in :- ", pos_in.shape)
    # print("shape of target :- ", target.shape)
    # print("shape of glyph_in :- ", glyph_in.shape)
    pos_in_tensor = torch.cat((pos_in_tensor, pos_in),0).to(device)
    pos_in_tensor = pos_in_tensor[1:]
    target_in_tensor = torch.cat((target_in_tensor, target),0).to(device)
    target_in_tensor = target_in_tensor[1:]
    gly_in_tensor = torch.cat((gly_in_tensor, glyph_in),0).to(device)
    gly_in_tensor = gly_in_tensor[1:]
    return last_loss, total_loss, fuse_out_tensor, train_losses, gly_out_tensor, pos_out_tensor, pos_in_tensor, target_in_tensor, gly_in_tensor

epoch_number = 0


EPOCHS = 1000

best_vloss = float('inf')

import torchvision.transforms as T
transform = T.ToPILImage()
output_dir = '/home/ubuntu/anytext_v1/AnyText/eval/epoch_results_v2'
patience = 5 # Number of epochs to wait for improvement
early_stopping_counter = 0
best_valid_loss = float('inf')
best_train_loss = float('inf')


#from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt

#early_stopping = EarlyStopping(patience = patience, verbose=True)
losses = []
import random

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    model.train(True)
    if batch_size > 1:
        a = random.randint(1,(batch_size-1))
    else:
        a = 0
    print("random number :- ", a)
    last_loss, total_loss, tensor_output_visualize, train_losses, gly_out_tensor, pos_out_tensor, pos_in_tensor, target_in_tensor, gly_in_tensor = train_one_epoch(epoch_number, writer)
    #print(" tensor_output_visualize shape :- ", tensor_output_visualize.shape)
    img = tensor_output_visualize[a][0]
    output_file_path = os.path.join(output_dir, "output_image_"+str(epoch)+".jpg")
    img = transform(img)
    #print(" output_file_path :- ", output_file_path)
    img.save(output_file_path)

    img1 = gly_in_tensor[a][0]
    output_file_path1 = os.path.join(output_dir, "input_gly_image_"+str(epoch)+".jpg")
    img1 = transform(img1)
    #print(" output_file_path1 :- ", output_file_path1)
    img1.save(output_file_path1)

    img2 = pos_in_tensor[a][0]
    output_file_path2 = os.path.join(output_dir, "input_pos_image_"+str(epoch)+".jpg")
    img2 = transform(img2)
    #print(" output_file_path2 :- ", output_file_path2)
    img2.save(output_file_path2)

    img3 = target_in_tensor[a][0]
    output_file_path3 = os.path.join(output_dir, "GT_image_"+str(epoch)+".jpg")
    img3 = transform(img3)
    #print(" output_file_path2 :- ", output_file_path3)
    img3.save(output_file_path3)


    avg_loss = total_loss/(batch_size*batches_per_epoch*EPOCHS)
    #print("average loss :- ", avg_loss)
    losses.append(avg_loss)
    running_vloss = 0.0

    # set the model to evaluate disabling dropout
    model.eval()

    # disable gradient computation and reduce memory consumption
    with torch.no_grad():
        for i, vdata in enumerate(zip(glyph_val_dataloader,
                                      pos_val_dataloader,
                                      target_val_dataloader)):
            glyph, pos, target = vdata
            glyph = glyph.to(device)
            pos = pos.to(device)
            target = target.to(device)
            target_shape = torch.Size([batch_size, 1, 1024, 1024])
            val_fuse_out, val_glyph_out, val_glyph_in, val_pos_out, val_pos_in = model(glyph, pos, target_shape)
            vloss = loss_fn(val_fuse_out, target)
            valid_losses.append(vloss.item())
            running_vloss += vloss
    
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    print_msg = (f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
    print(print_msg)

    train_losses = []
    valid_losses = []
    avg_vloss = running_vloss / (i+1)
    # check if scheduler can use v_loss for steping ?
    #scheduler.step(avg_vloss)
    scheduler.step(train_loss)
    print('LOSS train {} valid {}'.format(train_loss, valid_loss))

    # save the checkpoint
    PATH = "checkpoint.pt"
    torch.save({
        'epoch': epoch,
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "loss": train_loss
    }, PATH)

    # for parameter in model.parameters():
    #     print("Model Prameters :- ", parameter.data)
        
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("early stopping triggered. No improvement for {} epochs.".format(patience))
            break

    epoch_number += 1

plt.plot(losses)
plt.savefig("losses_v1.png")


# checkpoint = {'model': Model(),
#               'state_dict': model.state_dict(),
#               'optimizer' : optimizer.state_dict()}

# torch.save(checkpoint, 'checkpoint.pth')
