#%%
import glob, os
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import numpy as np

#If you downloaded this dataset from kaggle https://www.kaggle.com/datasets/yatata1/crack-dataset status as of July 3, 2023
#At first you should fix some things, function below will do it for you
def fix_dataset(convert=False):
    #1. Change folder name at path ./Concrete/Negative/Mask to Masks
    mask_path = './Concrete/Negative/Mask'
    if(os.path.isdir(mask_path)):
        os.rename(mask_path, mask_path + 's')
        print('Path modified')

    #We are ignoring *.png files because of their different size, it could lead to reach out of memory on gpu
    image_paths_negative = glob.glob(".\\Concrete\\Negative\\Images\\*.jpg",recursive=False)
    mask_paths_negative = glob.glob(".\\Concrete\\Negative\\Masks\\*.jpg",recursive=False)
    image_paths_positive = glob.glob(".\\Concrete\\Positive\\Images\\*.jpg",recursive=False)
    mask_paths_positive = glob.glob(".\\Concrete\\Positive\\Masks\\*.jpg",recursive=False)

    #2. Check if every image has corrensponding mask
    if(image_paths_negative != mask_paths_negative):
        print('negative', 'img:', len(image_paths_negative), ' mask:',len(mask_paths_negative))
        solve_not_equal_paths(image_paths_negative, mask_paths_negative, 'Negative')

    if(image_paths_positive != mask_paths_positive):
        print('positive', 'img:', len(image_paths_positive), ' mask:',len(mask_paths_positive))
        solve_not_equal_paths(image_paths_positive, mask_paths_positive, 'Positive')

    #3. Convert 3-channel masks to single channel
    if convert:     
        mask_paths = mask_paths_negative + mask_paths_positive
        for i in range(len(dataset)):
            print(f'Checking: {i+1}/{len(dataset)}', dataset[i][1], end='\r')
            if dataset[i][1].mode != 'L':
                print(f'Converting and saving "{os.path.basename(os.path.normpath(mask_paths[i]))}" to single channel')
                single_ch_mask = ImageOps.grayscale(dataset[i][1])
                single_ch_mask.save(mask_paths[i])
  
 
def solve_not_equal_paths(images, masks, folder):
    if(len(images) > len(masks)):
        for img_path in images:
            if(not os.path.exists("./Concrete/" + folder + "/Masks/" + os.path.basename(img_path))):
               print('to remove', img_path)
               os.remove(img_path)

    if(len(images) < len(masks)):
        for mask_path in masks:
            if(not os.path.exists("./Concrete/" + folder + "/Images/" + os.path.basename(mask_path))):
               print('to remove', mask_path)
               os.remove(mask_path)
    

class ConcreteCracksDataset(Dataset, ):
    def __init__(self, n_negative=0, n_positive=0, transform=None, skip=0):
        images_paths_negative = glob.glob(".\\Concrete\\Negative\\Images\\*.jpg",recursive=False)
        masks_paths_negative = glob.glob(".\\Concrete\\Negative\\Masks\\*.jpg",recursive=False)
        images_paths_positive = glob.glob(".\\Concrete\\Positive\\Images\\*.jpg",recursive=False)
        masks_paths_positive = glob.glob(".\\Concrete\\Positive\\Masks\\*.jpg",recursive=False)

        if n_negative == 0:
            self.n_negative = len(images_paths_negative)
            self.skip = 0
        elif n_negative + skip > len(images_paths_negative) and skip != 0:
            self.n_negative = n_negative
            self.skip = 0
            print('Skip value to big, will be 0 N')
        else:
            self.n_negative = n_negative
            self.skip = skip

        if n_positive == 0:
            self.n_positive = len(images_paths_positive)
            self.skip = 0
        elif n_positive + skip > len(images_paths_positive) and skip != 0:
            self.n_positive = n_positive
            self.skip = 0
            print('Skip value to big, will be 0 P')
        else:
            self.n_positive = n_positive
            self.skip = skip
            
        self.transform = transform   
        self.images_paths = images_paths_negative[skip:self.n_negative+skip] + images_paths_positive[skip:self.n_positive+skip]
        self.masks_paths = masks_paths_negative[skip:self.n_negative+skip] + masks_paths_positive[skip:self.n_positive+skip]

    def __getitem__(self, index):       
        image = Image.open(self.images_paths[index])
        mask = Image.open(self.masks_paths[index])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    
    def __len__(self):
        return len(self.images_paths)
#%% 
if __name__ == '__main__':
    dataset = ConcreteCracksDataset(10000,10000, skip=0)
    fix_dataset(convert=True)