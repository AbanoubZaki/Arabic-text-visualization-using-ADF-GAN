from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import prepare_data, TextBertDataset

from DAMSM import BERT_RNN_ENCODER

from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor

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

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import warnings
warnings.filterwarnings("ignore")

from matplotlib.pyplot import imshow
import streamlit as st
# %matplotlib inline


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


import multiprocessing
multiprocessing.set_start_method('spawn', True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPDATE_INTERVAL = 200

import tensorflow as tf
# This should not be hashed by Streamlit when using st.cache.
TL_GAN_HASH_FUNCS = {
    tf.compat.v1.Session : id
}

def parse_args():
    parser = argparse.ArgumentParser(description='GUI')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--evaluation', type=int, help='evaluation', default= 0)
    args = parser.parse_args()
    
    return args
    
    
def vectorize_caption(arabert_prep, tokenizer, wordtoix, caption, copies= 1):
    # create caption vector

    caption = arabert_prep.preprocess(caption)
    print(caption)
    tokens = tokenizer.tokenize(caption.lower())
    print(tokens)
    
    cap_v = []
    for t in tokens:
        t = t.strip().encode('cp1256', 'ignore').decode('cp1256')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i,:] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)

    return captions.astype(int), cap_lens.astype(int)
    
    
    
def generate(caption, tokenizer, arabert_prep, text_encoder, netG, wordtoix, copies= 1):
    # load word vector
    captions, cap_lens  = vectorize_caption(arabert_prep, tokenizer, wordtoix, caption, copies)
    n_words = len(wordtoix)

    # only one to generate
    batch_size = captions.shape[0]

    captions = Variable(torch.from_numpy(captions)).to(device)
    cap_lens = Variable(torch.from_numpy(cap_lens)).to(device)    

    #######################################################
    # (1) Extract text embeddings
    #######################################################
    hidden = text_encoder.init_hidden(batch_size)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        

    #######################################################
    # (2) Generate fake images
    #######################################################
    with torch.no_grad():
      noise = torch.randn(batch_size, 100)
      noise = noise.to(device)
      fake_imgs = netG(noise, sent_emb)

    # G attention
    cap_lens_np = cap_lens.cpu().data.numpy()

    imgs = []
    # only look at first one
    #j = 0
    for k in range(len(fake_imgs)):
        im = fake_imgs[k].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        imgs.append(im)
    # imshow(imgs[-1])
    return imgs

@st.cache(allow_output_mutation=True, hash_funcs=TL_GAN_HASH_FUNCS)
def load_model():
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    
    print('Using config:')
    pprint.pprint(cfg)

    # if not cfg.TRAIN.FLAG:
    #     args.manualSeed = 100
    # elif args.manualSeed is None:
    #     args.manualSeed = 100

        
    # print("seed now is : ",args.manualSeed)
    # random.seed(args.manualSeed)
    # np.random.seed(args.manualSeed)
    # torch.manual_seed(args.manualSeed)
    # if cfg.CUDA:
    #     torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    
    filepath = os.path.join(cfg.DATA_DIR, 'arabic_captions.pickle')
    x = pickle.load(open(filepath, 'rb'))
    wordtoix = x[3]

    word_len = len(wordtoix)


    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # load generator
    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    path = '../models/%s/checkpoint_nets_1250.pth' % (cfg.CONFIG_NAME)
  
    if(os.path.exists(path)):
      checkpoint = torch.load(path)
      netG.load_state_dict(checkpoint['netG_state'])
      state_epoch = checkpoint['epoch']
      netG.eval()
      print("Loading last checkpoint at epoch: ",state_epoch)
    else:
      print("No checkpoint to load")
    
    #load text encoder
    text_encoder = BERT_RNN_ENCODER(word_len, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    
    
    #load tokenizer & ArabertPreprocessor.
    tokenizer = AutoTokenizer.from_pretrained(cfg.GAN.BERT_NAME)
    arabert_prep = ArabertPreprocessor(cfg.GAN.BERT_NAME)
    
    return tokenizer, arabert_prep, text_encoder, netG, wordtoix


tokenizer, arabert_prep, text_encoder, netG, wordtoix = load_model()

def main():
    st.title("Arabic text visualization Demo")
    st.write('\n\n')
    selected_caption = st.selectbox( 'Select The Caption', 
    ('هذا طائر بني ذو صدر رمادي ومنقار مدبب.',
     'هذا الطائر له ريش أسمر على صدره بنمط مخطط أبيض وأسود وأصفر على رأسه وظهره',
      'الطائر ذو المنقار الرمادي ، والصدر والبطن ، والحلق الأبيض ، والعين السوداء ، والرأس يتناسب مع جسمه',
      'هذا الطائر الصغير له صدر رمادي فاتح ، أصفر على تاجه وخلفه وأشرطة جناح بيضاء.',
      'طائر صغير أبيض اللون ذو أجنحة صفراء ومنقار قصير مدبب بين رأس أسود مع شريط أبيض فوق عينه.',
      'يحتوي هذا الطائر الصغير على ريش أصفر لامع في جميع أنحاء جسمه ماعدا أجنحته وذيله الأسود.'))

    caption = st.text_input("Enter The Caption", selected_caption)
    n_copies = st.slider('Number of Generated Images', min_value=1, max_value=12, value=6, step=1)

    if st.button('Generate Image'):
        generated_images = generate(caption, tokenizer, arabert_prep, text_encoder, netG, wordtoix, copies= n_copies)
        st.image(generated_images, caption = [f'Generated Image {i + 1}' for i in range(len(generated_images))], width= None)
        # for img in generated_images:
        #   st.image(img, caption = 'Generated Image.', width=None)


if __name__ == "__main__":
    main()