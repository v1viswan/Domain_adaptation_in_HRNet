import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

import pickle
import glob
from torchvision import transforms
import PIL.Image as Image

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

import glob
from PIL import Image
import torchvision.transforms as transforms

class Image_loader():
    '''
    This Class is to have a dataloader for images from domain 2 which do not have classification labels.
    The class expects a folder path where all the domain 2 images are present and what should be the image shape
    for the images.
    '''
    def __init__(self, folder_path, img_shape):
        
        self.folder_path = folder_path
        self.img_shape = img_shape
        self.file_list = glob.glob(folder_path + "/*jpg")
        self.trans = transforms.ToTensor()
        self.order = np.random.choice([i for i in range(len(self.file_list))], len(self.file_list), replace=False)
        self.current_index = 0
    def get_images(self,batch_size, img_shape=None):
        if img_shape is None:
            img_shape = self.img_shape
        img_list = []
        for i in range(batch_size):
            img = Image.open(self.file_list[self.order[self.current_index]])
            self.current_index = (self.current_index+1)%len(self.file_list)
            
            img_list.append( self.trans(img.resize(img_shape)) )
        img_dataset = torch.stack(img_list)
        return img_dataset

######################### The Main Script starts here ###################################
args = parse_args()

logger, final_output_dir, tb_log_dir = create_logger(
    config, args.cfg, 'train')

logger.info(pprint.pformat(args))
logger.info(config)

writer_dict = {
    'writer': SummaryWriter(tb_log_dir),
    'train_global_steps': 0,
    'valid_global_steps': 0,
}

cudnn.benchmark = config.CUDNN.BENCHMARK
cudnn.deterministic = config.CUDNN.DETERMINISTIC
cudnn.enabled = config.CUDNN.ENABLED
# gpus = list(config.GPUS)
gpus = [0,1] # changed

# build model
model = eval('models.'+config.MODEL.NAME +
             '.get_seg_model')(config)

config.TRAIN.IMAGE_SIZE[0] = int(512)
config.TRAIN.IMAGE_SIZE[1] = int(256)

dump_input = torch.rand(
    (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
)
logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

# copy model file
this_dir = os.path.dirname('./') #changed
models_dst_dir = os.path.join(final_output_dir, 'models')
if os.path.exists(models_dst_dir):
    shutil.rmtree(models_dst_dir)
shutil.copytree(os.path.join(this_dir, './lib/models'), models_dst_dir)

# prepare data
crop_size = (int(config.TRAIN.IMAGE_SIZE[1]), int(config.TRAIN.IMAGE_SIZE[0]))
train_dataset = eval('datasets.'+config.DATASET.DATASET)(
#                     root = '../' + config.DATASET.ROOT, #changed
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.TRAIN_SET,
                    num_samples=None,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=config.TRAIN.MULTI_SCALE,
                    flip=config.TRAIN.FLIP,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TRAIN.BASE_SIZE,
                    crop_size=crop_size,
                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                    scale_factor=config.TRAIN.SCALE_FACTOR)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
    shuffle=config.TRAIN.SHUFFLE,
    num_workers=0, #config.WORKERS,
    pin_memory=True,
    drop_last=True)

if config.DATASET.EXTRA_TRAIN_SET:
    extra_train_dataset = eval('datasets.'+config.DATASET.DATASET)(
#                 root = '../' + config.DATASET.ROOT, #changed
                root=config.DATASET.ROOT,
                list_path=config.DATASET.EXTRA_TRAIN_SET,
                num_samples=None,
                num_classes=config.DATASET.NUM_CLASSES,
                multi_scale=config.TRAIN.MULTI_SCALE,
                flip=config.TRAIN.FLIP,
                ignore_label=config.TRAIN.IGNORE_LABEL,
                base_size=config.TRAIN.BASE_SIZE,
                crop_size=crop_size,
                downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                scale_factor=config.TRAIN.SCALE_FACTOR)

    extra_trainloader = torch.utils.data.DataLoader(
        extra_train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True)

test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
test_dataset = eval('datasets.'+config.DATASET.DATASET)(
#                     root = '../' + config.DATASET.ROOT, #changed
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.TEST_SET,
                    num_samples=config.TEST.NUM_SAMPLES,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=False,
                    flip=False,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TEST.BASE_SIZE,
                    crop_size=test_size,
                    downsample_rate=1)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
    shuffle=False,
    num_workers=0,#config.WORKERS,
    pin_memory=True)



from models import segnet_vj

domain_network = segnet_vj.segnet_domain_adapt(config)
domain_network.init_weights(config.MODEL.PRETRAINED)

optimizer = torch.optim.SGD([{'params':
                  filter(lambda p: p.requires_grad,
                         domain_network.parameters()),
                  'lr': config.CLASSIFIER_LR}],
                lr=config.CLASSIFIER_LR,
                momentum=config.TRAIN.MOMENTUM,
                weight_decay=config.TRAIN.WD,
                nesterov=config.TRAIN.NESTEROV,
                )
classifier_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                             weight=train_dataset.class_weights)


# find output dimension of our model
domain_network.eval()
domain_network.float()
domain_network.cuda()
with torch.no_grad():
    feat = domain_network.forward_feature(dump_input.cuda())
    pred = domain_network.forward_classifier(feat)
disc_input_size = feat.shape[1]

discriminator = segnet_vj.FCN_discriminator(in_channels=disc_input_size, out_classes=2)

# Define discriminator loss
disc_criterion = nn.CrossEntropyLoss()

disc_optimizer = torch.optim.SGD([{'params':
                filter(lambda p: p.requires_grad,
                     discriminator.parameters()),
                'lr': config.DISCRIMINATOR_LR}],
                lr=config.DISCRIMINATOR_LR,
                momentum=config.TRAIN.MOMENTUM,
                weight_decay=config.TRAIN.WD,
                nesterov=config.TRAIN.NESTEROV,
                )
full_network = segnet_vj.Full_Adaptation_Model(network = domain_network, loss_classifier=classifier_criterion,\
                            discriminator=discriminator, loss_discriminator=disc_criterion)
# full_network.cuda()

try:
    full_network.network.load_state_dict(torch.load(config.SAVED_MODELS.CLASSIFIER))
    full_network.discriminator.load_state_dict(torch.load(config.SAVED_MODELS.DISCRIMINATOR))
    print("Network loaded from saved models:", config.SAVED_MODELS.CLASSIFIER, config.SAVED_MODELS.DISCRIMINATOR)
except:
    print("Network loaded from scratch (except classification net)")

full_network_model = nn.DataParallel(full_network, device_ids=gpus).cuda()


folder_path = config.DOMAIN2_FOLDER
img_shape = (config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])

img_loader = Image_loader(folder_path = config.DOMAIN2_FOLDER, img_shape=img_shape)

# Now train the network
full_network_model.train()
for i_iter, batch in enumerate(trainloader, 0):
    disc_optimizer.zero_grad()
    optimizer.zero_grad()

    images, labels, _, _ = batch
    d2_images = img_loader.get_images(batch_size=len(images))
    label_discriminator = np.zeros([len(images)*2])
    label_discriminator[-len(d2_images):] = 1
    label_discriminator = torch.from_numpy(label_discriminator)
    
    loss = full_network_model(input_d1=images, label_d1=labels.cuda().long(), input_d2=d2_images,\
                                    label_discriminator=label_discriminator.cuda().long(), lamda=0.25)
    final_loss = torch.mean(loss)
    final_loss.backward()
    disc_optimizer.step()
    optimizer.step()
    if (i_iter%config.LOG_INTERVAL ==0):
        print("Iteration: %d, final_loss: %.4f"%(i_iter, final_loss.item()))
    if (i_iter >= config.RUNS):
        break
torch.save(full_network.network.state_dict(), config.SAVE_C_PATH)
torch.save(full_network.discriminator.state_dict(), config.SAVE_D_PATH)