import os
import argparse
import time
import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

import numpy as np
from nets.discriminator import Discriminator
from nets.generator import Generator
import itertools
from utils import Utils
import torchvision.utils as vutils

class GAN_CLS(object):
    def __init__(self, args, data_loader, SUPERVISED=True):
        """
        Arguments :
        ----------
        args : Arguments defined in Argument Parser
        data_loader = An instance of class DataLoader for loading our dataset in batches
        SUPERVISED :

        """
        config = args
        self.data_loader = data_loader
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size

        self.log_step = config.log_step
        self.sample_step = config.sample_step

        self.log_dir = args.log_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.final_model = args.final_model

        self.dataset = args.dataset
        #self.model_name = args.model_name

        self.img_size = args.img_size
        self.z_dim = args.z_dim
        self.text_embed_dim = args.text_embed_dim
        self.text_reduced_dim = args.text_reduced_dim
        self.learning_rate = args.learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.l1_coeff = args.l1_coeff
        self.resume_epoch = args.resume_epoch
        self.SUPERVISED = SUPERVISED

        # Logger setting
        self.logger = logging.getLogger('__name__')
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        self.file_handler = logging.FileHandler(self.log_dir+'/file.log')
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

        self.build_model()

    def build_model(self):
        """ A function of defining following instances :

        -----  Generator
        -----  Discriminator
        -----  Optimizer for Generator
        -----  Optimizer for Discriminator
        -----  Defining Loss functions

        """

        # ---------------------------------------------------------------------
        #						1. Network Initialization
        # ---------------------------------------------------------------------
        self.gen = Generator(batch_size=self.batch_size,
            img_size=self.img_size,
            z_dim=self.z_dim,
            text_embed_dim=self.text_embed_dim,
            reduced_text_dim=self.text_reduced_dim)

        self.disc = Discriminator(batch_size=self.batch_size,
                                  img_size=self.img_size,
                                  text_embed_dim=self.text_embed_dim,
                                  text_reduced_dim=self.text_reduced_dim)

        self.gen_optim = optim.Adam(self.gen.parameters(),
                                    lr=self.learning_rate,
                                    betas=(self.beta1, self.beta2))

        self.disc_optim = optim.Adam(self.disc.parameters(),
                                     lr=self.learning_rate,
                                     betas=(self.beta1, self.beta2))

        self.cls_gan_optim = optim.Adam(itertools.chain(self.gen.parameters(),
                                                        self.disc.parameters()),
                                        lr=self.learning_rate,
                                        betas=(self.beta1, self.beta2))

        print ('-------------  Generator Model Info  ---------------')
        self.print_network(self.gen, 'G')
        print ('------------------------------------------------')

        print ('-------------  Discriminator Model Info  ---------------')
        self.print_network(self.disc, 'D')
        print ('------------------------------------------------')

        self.gen.cuda()
        self.disc.cuda()
        self.criterion = nn.BCELoss().cuda()
        self.l1loss = nn.L1Loss().cuda()
        self.l2loss = nn.MSELoss().cuda()
        # self.CE_loss = nn.CrossEntropyLoss().cuda()
        # self.MSE_loss = nn.MSELoss().cuda()
        self.gen.train()
        self.disc.train()

    def print_network(self, model, name):
        """ A function for printing total number of model parameters """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        print(model)
        print(name)
        print("Total number of parameters: {}".format(num_params))

    def load_checkpoints(self, resume_epoch):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_epoch))
        G_path = os.path.join(self.checkpoint_dir, '{}-G.ckpt'.format(resume_epoch))
        D_path = os.path.join(self.checkpoint_dir, '{}-D.ckpt'.format(resume_epoch))
        self.gen.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.disc.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def save_img_results(self, data_img, fake, epoch, image_dir):
        num = 64
        fake = fake[0:num]
        # data_img is changed to [0,1]
        if data_img is not None:
            data_img = data_img[0:num]
            vutils.save_image(data_img, '%s/real_samples.png' % image_dir, normalize=True)
            # fake.data is still [-1, 1]
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' %(image_dir, epoch), normalize=True)
        else:
            vutils.save_image(
                fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
                (image_dir, epoch), normalize=True)

    def train_model(self):
        fixed_noise = Variable(torch.randn(64, self.z_dim)).cuda()
        data_loader = self.data_loader

        start_epoch = 0
        if self.resume_epoch:
            start_epoch = self.resume_epoch
            self.load_checkpoints(self.resume_epoch)

        print ('---------------  Model Training Started  ---------------')
        start_time = time.time()
        log = ""
        for epoch in range(start_epoch, self.num_epochs):
            start_t = time.time()
            for idx, batch in enumerate(data_loader):
                true_imgs = batch['true_imgs']
                true_embed = batch['true_embed']
                false_imgs = batch['false_imgs']

                real_labels = torch.ones(true_imgs.size(0))
                fake_labels = torch.zeros(true_imgs.size(0))
                
                smooth_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                true_imgs = Variable(true_imgs.float()).cuda()
                true_embed = Variable(true_embed.float()).cuda()
                false_imgs = Variable(false_imgs.float()).cuda()

                real_labels = Variable(real_labels).cuda()
                smooth_real_labels = Variable(smooth_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # ---------------------------------------------------------------
                #                   2. Training the discriminator
                # ---------------------------------------------------------------
                self.disc.zero_grad()
                true_out, true_logit = self.disc(true_imgs, true_embed)
                false_out, false_logit = self.disc(false_imgs, true_embed)
                disc_loss = self.criterion(true_out, smooth_real_labels) + self.criterion(false_out, fake_labels)

                noise = Variable(torch.randn(true_imgs.size(0), self.z_dim)).cuda()
                fake_imgs = self.gen(true_embed, noise)
                false_out, _ = self.disc(fake_imgs, true_embed)
                disc_loss = disc_loss + self.criterion(false_out, fake_labels)

                disc_loss.backward()
                self.disc_optim.step()


                # ---------------------------------------------------------------
                # 					  3. Training the generator
                # ---------------------------------------------------------------
                self.gen.zero_grad()
                
                z = Variable(torch.randn(true_imgs.size(0), self.z_dim)).cuda()
                fake_imgs = self.gen(true_embed, z)
                fake_out, fake_logit = self.disc(fake_imgs, true_embed)
                true_out, true_logit = self.disc(true_imgs, true_embed)

                activation_fake = torch.mean(fake_logit, 0)
                activation_real = torch.mean(true_logit, 0)

                gen_loss = self.criterion(fake_out, real_labels)
                gen_loss = gen_loss + self.l1_coeff * self.l1loss(fake_imgs, true_imgs) + self.l2loss(activation_fake, activation_real)

                gen_loss.backward()
                self.gen_optim.step()

                # self.cls_gan_optim.step()

                # Logging
                loss = {}
                loss['G_loss'] = gen_loss.item()
                loss['D_loss'] = disc_loss.item()

                # ---------------------------------------------------------------
                # 					4. Logging INFO into log_dir
                # ---------------------------------------------------------------
                if idx % self.log_step == 0:
                    end_time = time.time() - start_time
                    end_time = datetime.timedelta(seconds=end_time)
                    log = "Elapsed [{}], Epoch [{}/{}], Idx [{}/{}]".format(end_time, epoch,
                                                                         self.num_epochs, idx, len(data_loader))
                    for net, loss_value in loss.items():
                        log += ", {}: {:.4f}".format(net, loss_value)
                    print (log)
                    self.logger.info(log)

                """
                log = "Epoch [{}/{}], Idx [{}/{}]".format(epoch, self.num_epochs, idx, len(data_loader))
                for net, loss_value in loss.items():
                    log += ", {}: {:.4f}".format(net, loss_value)
                
                self.logger.info(log)
                """    

                # ---------------------------------------------------------------
                # 					5. Saving generated images
                # ---------------------------------------------------------------
                if (idx + 1) % self.sample_step == 0:
                    fake_imgs = self.gen(true_embed, fixed_noise)
                    concat_imgs = torch.cat((true_imgs, fake_imgs), 2)  # ??????????
                    save_path = os.path.join(self.sample_dir, '{}-images.png'.format(idx + 1))
                    concat_imgs = (concat_imgs + 1) / 2
                    # out.clamp_(0, 1)
                    #save_image(concat_imgs.data.cpu(), save_path, nrow=1, padding=0)
                    self.save_img_results(true_imgs, fake_imgs, epoch, self.sample_dir)

                    print ('Saved real and fake images into {}...'.format(self.sample_dir))

                # ---------------------------------------------------------------
                # 				6. Saving the checkpoints & final model
                # ---------------------------------------------------------------
            
            end_t = time.time()    
            G_path = os.path.join(self.checkpoint_dir, '{}-G.ckpt'.format(epoch))
            D_path = os.path.join(self.checkpoint_dir, '{}-D.ckpt'.format(epoch))
            torch.save(self.gen.state_dict(), G_path)
            torch.save(self.disc.state_dict(), D_path)
            print(log)
            print('Total Time: {:.2f} sec and Saved model checkpoints into {}...'.format((end_t - start_t), self.checkpoint_dir))        

        print ('---------------  Model Training Completed  ---------------')
        # Saving final model into final_model directory
        G_path = os.path.join(self.final_model, '{}-G.pth'.format('final'))
        D_path = os.path.join(self.final_model, '{}-D.pth'.format('final'))
        torch.save(self.gen.state_dict(), G_path)
        torch.save(self.disc.state_dict(), D_path)
        print('Saved final model into {}...'.format(self.final_model))

    