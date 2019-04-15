import os
import torch
import numpy as np
from PIL import Image
import pickle
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
# Each batch will have 3 things : true image, its captions(5), and false image(real image but image
# corresponding to an incorrect caption).
# Discriminator is trained in such a way that true_img + caption corresponds to a real example and
# false_img + caption corresponds to a fake example.


class Text2ImageDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        
        self.load_flower_dataset()

    def load_flower_dataset(self):
        # It will return two things : a list of image file names, a dictionary of 5 captions per image
        # with image file name as the key of the dictionary and 5 values(captions) for each key.

        print ("------------------  Loading images  ------------------")
        filepath = os.path.join(self.data_dir, 'file_caption_map.pickle')
        fileObject = open(filepath,'rb')  
        filenames = pickle.load(fileObject)
        self.img_files = np.array(filenames.keys())

        print('Load filenames from: %s (%d)' % (filepath, len(self.img_files)))

        print ("------------------  Loading captions  ----------------")
        
        self.img_captions = filenames                 
        
        print ("---------------  Loading Skip-thought Model  ---------------")
        embedding_filename = '/file_caption_embedding.pickle'

        with open(self.data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            self.encoded_captions =  embeddings   
        
        print ("-------------  Encoding of image captions DONE  -------------")

    def read_image(self, image_file_name):
        image = Image.open(os.path.join(self.data_dir, 'images/' + image_file_name))
        # check its shape and reshape it to (64, 64, 3)
        image = image.resize((64, 64))
        return image

    def get_false_img(self, index):
        false_img_id = np.random.randint(len(self.img_files))
        if false_img_id != index:
            return self.img_files[false_img_id]

        return self.get_false_img(index)

    def __len__(self):

        return len(self.img_files)

    def __getitem__(self, index):

        sample = {}
        sample['true_imgs'] = self.image_transform(self.read_image(self.img_files[index]))
        sample['false_imgs'] = self.image_transform(self.read_image(self.get_false_img(index)))
        embeddings = self.encoded_captions[self.img_files[index]]
        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        sample['true_embed'] = torch.FloatTensor(embedding)

        return sample
