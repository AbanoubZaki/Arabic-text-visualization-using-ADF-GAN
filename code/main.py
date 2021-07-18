from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import prepare_data, TextBertDataset
from eval.IS.inception_score import compute_IS
from eval.FID.fid_score import compute_FID

from DAMSM import BERT_RNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetD

import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)


UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--evaluation', type=int, help='evaluation', default= 0)
    args = parser.parse_args()
    return args


def sampling(text_encoder, netG, dataloader,device, validation= False):
    
    state_epoch = 0
    model_dir = '../models/%s/checkpoint_nets.pth' % (cfg.CONFIG_NAME)
        
    if(not validation and os.path.exists(model_dir)):
        checkpoint = torch.load(model_dir)
        netG.load_state_dict(checkpoint['netG_state'])
        state_epoch = checkpoint['epoch']
        netG.eval()
        print("loading last checkpoint at epoch: ",state_epoch)
        
    batch_size = cfg.TRAIN.BATCH_SIZE
    save_dir = '../images/%s/test' % (cfg.CONFIG_NAME)
    mkdir_p(save_dir)
    cnt = 0
    for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            cnt += batch_size
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
                noise = torch.randn(batch_size, 100)
                noise=noise.to(device)
                fake_imgs = netG(noise,sent_emb)
            for j in range(batch_size):
                s_tmp = '%s/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%3d.png' % (s_tmp,i)
                im.save(fullpath)
                
    return state_epoch

def validate(text_encoder, netG,device, writer, epoch):
    dataset = TextBertDataset(cfg.DATA_DIR, 'test',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))
    print(f'Starting generate validation images ...  at {epoch}')    
    sampling(text_encoder, netG, dataloader, device, validation= True)
    
    netG.train()
    
    
    print(f'Starting compute FID & IS ... at {epoch}')
    
    compute_FID(['../FIDS/%s_val.npz' % (cfg.CONFIG_NAME), 
        '../images/%s/test' % (cfg.CONFIG_NAME)], writer, epoch)
    
    compute_IS('../images/%s/test' % (cfg.CONFIG_NAME), writer, epoch)
  
  
def train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD,state_epoch,batch_size,device, writer):
    
  path = '../models/%s/checkpoint_nets.pth' % (cfg.CONFIG_NAME)
  
  if(os.path.exists(path)):
      checkpoint = torch.load(path)
      netG.load_state_dict(checkpoint['netG_state'])
      netD.load_state_dict(checkpoint['netD_state'])
      optimizerG.load_state_dict(checkpoint['optimizerG_state'])
      optimizerD.load_state_dict(checkpoint['optimizerD_state'])
      state_epoch = checkpoint['epoch']
      netG.train()
      netD.train()
      print("Loading last checkpoint at epoch: ",state_epoch)
  else:
      print("No checkpoint to load")
      
      

  for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
      D_loss = 0.0
      G_loss = 0.0
      for step, data in enumerate(dataloader, 0):
          
          imags, captions, cap_lens, class_ids, keys = prepare_data(data)
          hidden = text_encoder.init_hidden(batch_size)
          # words_embs: batch_size x nef x seq_len
          # sent_emb: batch_size x nef
          words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
          words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

          imgs=imags[0].to(device)
          real_features = netD(imgs)
          
          output = netD.COND_DNET(real_features,sent_emb)
          errD_real = torch.nn.ReLU()(1.0 - output).mean()

          output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
          errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

          # synthesize fake images
          noise = torch.randn(batch_size, 100)
          noise=noise.to(device)
          fake = netG(noise,sent_emb)  
          
          # G does not need update with D
          fake_features = netD(fake.detach()) 

          errD_fake = netD.COND_DNET(fake_features,sent_emb)
          errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()          

          errD = errD_real + (errD_fake + errD_mismatch)/2.0
          optimizerD.zero_grad()
          optimizerG.zero_grad()
          errD.backward()
          optimizerD.step()

          #MA-GP
          interpolated = (imgs.data).requires_grad_()
          sent_inter = (sent_emb.data).requires_grad_()
          features = netD(interpolated)
          out = netD.COND_DNET(features,sent_inter)
          grads = torch.autograd.grad(outputs=out,
                                  inputs=(interpolated,sent_inter),
                                  grad_outputs=torch.ones(out.size()).cuda(),
                                  retain_graph=True,
                                  create_graph=True,
                                  only_inputs=True)
          grad0 = grads[0].view(grads[0].size(0), -1)
          grad1 = grads[1].view(grads[1].size(0), -1)
          grad = torch.cat((grad0,grad1),dim=1)                        
          grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
          d_loss_gp = torch.mean((grad_l2norm) ** 6)
          d_loss = 2.0 * d_loss_gp
          optimizerD.zero_grad()
          optimizerG.zero_grad()
          d_loss.backward()
          optimizerD.step()
          
          # update G
          features = netD(fake)
          output = netD.COND_DNET(features,sent_emb)
          errG = - output.mean()
          optimizerG.zero_grad()
          optimizerD.zero_grad()
          errG.backward()
          optimizerG.step()

          D_loss += errD.item() + d_loss.item()
          G_loss += errG.item()

          print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f total_Loss_D: %.3f total_Loss_G %.3f'
              % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errD.item(), errG.item(), D_loss, G_loss))

      vutils.save_image(fake.data,
                      '../images/%s/fakes/fake_samples_epoch_%03d.png' % (cfg.CONFIG_NAME, epoch),
                      normalize=True)

      # if epoch%10==0:
      torch.save({
          'epoch': epoch,
          'netG_state': netG.state_dict(),
          'optimizerG_state': optimizerG.state_dict(),
          'netD_state': netD.state_dict(),
          'optimizerD_state': optimizerD.state_dict()
          }, path)
          
      writer.add_scalar('D_Loss/train', D_loss, epoch)
      writer.add_scalar('G_Loss/train', G_loss, epoch)
      
      if epoch%50 == 0:
          return epoch

  return cfg.TRAIN.MAX_EPOCH




if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    
    cfg.B_VALIDATION = bool(args.evaluation)

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextBertDataset(cfg.DATA_DIR, 'test',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:     
        dataset = TextBertDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)

    text_encoder = BERT_RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    

    state_epoch=0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))  


    if cfg.B_VALIDATION:
        state_epoch = sampling(text_encoder, netG, dataloader,device)  # generate images for the whole valid dataset
        print('state_epoch:  %d'%(state_epoch))
    else:
        writer = SummaryWriter(f"tensorboards/{cfg.CONFIG_NAME}/ADGAN_train")
        epoch = train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD, state_epoch,batch_size,device, writer)
        validate(text_encoder, netG, device, writer, epoch)


        