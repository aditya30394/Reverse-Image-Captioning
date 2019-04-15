from __future__ import print_function

import os
import time
import argparse
import skipthoughts

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
import torchvision.utils as vutils

from nets.generator import Generator

import Tkinter

from Tkinter import *
from PIL import Image, ImageTk
import numpy
import random

import skipthoughts

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Size of the image')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Size of the latent variable')
    parser.add_argument('--final_model', type=str, default='final_model',
                        help='Save INFO into logger after every x iterations')
    parser.add_argument('--save_img', type=str, default='.',
                        help='Save predicted images')
    parser.add_argument('--text_embed_dim', type=int, default=4800,
                        help='Size of the embeddding for the captions')
    parser.add_argument('--text_reduced_dim', type=int, default=1024,
                        help='Reduced dimension of the caption encoding')
    parser.add_argument('--text', type=str, help='Input text to be converted into image')

    args = parser.parse_args()
    return args


config = parse_args()

print('------------------------SKIP THOUGHT LOADING-----------------------')
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
print('------------------------SKIP THOUGHT LOADING FINISHED-----------------------')

print('------------------------GENERATOR LOADING-----------------------')

gen = Generator(batch_size=config.batch_size,
                    img_size=config.img_size,
                    z_dim=config.z_dim,
                    text_embed_dim=config.text_embed_dim,
                    reduced_text_dim=config.text_reduced_dim)

gen.cuda()
# Loading the trained model
G_path = os.path.join(config.final_model, '{}-G.ckpt'.format('final_model'))

gen.load_state_dict(torch.load(G_path))
gen.eval()
print('------------------------GENERATOR LOADING FINISHED-----------------------')

output_dir = './'    


top = Tk()
top.title('Reverse Image Captioning')
top.geometry('500x500')

top_row = Frame(top).grid(row=0)

left = Frame(top_row).grid(row=0, column=0)
L1 = Label(left, text="Enter the image description:").grid(row=0)
E1 = Entry(left)
E1.grid(row=1)

right = Frame(top_row).grid(row=0, column=1)

canvas = Canvas(right, width=300,height=300, bd=0,bg='white')
canvas.grid(row=0, column=1)

def GenerateImage():
    z = torch.randn(config.batch_size, config.z_dim)
    z = z.cuda()
    text_input = E1.get()
    if (len(text_input) >= 10):
        text_input = [text_input]
        print(text_input)
        text_embedding = encoder.encode(text_input)
        print(text_embedding)
        text_embedding = torch.from_numpy(text_embedding)
        text_embedding = text_embedding.cuda()
        print(text_embedding.shape)
        output_img = gen(text_embedding, z)
        save_name = 'output.png'
        
        # fake.data is still [-1, 1]
        vutils.save_image(output_img.data, save_name, normalize=True)
            
        load = Image.open(save_name)
        w, h = load.size
        load = load.resize((256, 256))
        imgfile = ImageTk.PhotoImage(load)
        
        canvas.image = imgfile  # <--- keep reference of your image
        canvas.create_image(2,2,anchor='nw',image=imgfile)
    
    E1.delete(0, END)


submit_button = Button(top, text ='Generate Image', command = GenerateImage)
submit_button.grid(row=2, column=0)

submit_button = Button(top, text ='Exit', command = top.quit)
submit_button.grid(row=2, column=1)

top.mainloop()
