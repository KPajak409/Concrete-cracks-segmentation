#%%
import glob, os, math
import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import numpy as np



#If you downloaded this dataset from kaggle https://www.kaggle.com/datasets/yatata1/crack-dataset status as of July 3, 2023
#At first you should fix some things, function below will do it for you
#3. Remove 3-channel masks
def fix_dataset():
    #1. Change folder name at path ./Concrete/Negative/Mask to Masks
    mask_path = './Concrete/Negative/Mask'
    if(os.path.isdir(mask_path)):
        os.rename(mask_path, mask_path + 's')
        print('Path modified')

    #We are ignoring *.png files because of their different size, it could lead to reach out of memory on gpu
    image_paths_negative = glob.glob("./Concrete/Negative/Images/*.jpg",recursive=False)
    mask_paths_negative = glob.glob("./Concrete/Negative/Masks/*.jpg",recursive=False)
    image_paths_positive = glob.glob("./Concrete/Positive/Images/*.jpg",recursive=False)
    mask_paths_positive = glob.glob("./Concrete/Positive/Masks/*.jpg",recursive=False)

    #2. Check if every image has corrensponding mask
    if(image_paths_negative != mask_paths_negative):
        print('negative', 'img:', len(image_paths_negative), ' mask:',len(mask_paths_negative))
        solve_not_equal_paths(image_paths_negative, mask_paths_negative, 'Negative')

    if(image_paths_positive != mask_paths_positive):
        print('positive', 'img:', len(image_paths_positive), ' mask:',len(mask_paths_positive))
        solve_not_equal_paths(image_paths_positive, mask_paths_positive, 'Positive')
  
 
def solve_not_equal_paths(images, masks, folder):
    if(len(images) > len(masks)):
        for img_path in images:
            if(not os.path.exists("./Concrete/" + folder + "/Masks/" + os.path.basename(img_path))):
               print('to remove', img_path)
               #os.remove(img_path)

    if(len(images) < len(masks)):
        for mask_path in masks:
            if(not os.path.exists("./Concrete/" + folder + "/Images/" + os.path.basename(mask_path))):
               print('to remove', mask_path)
               #os.remove(mask_path)
    

class ConcreteScarsDataset(Dataset, ):
    def __init__(self, transform=None, n_negative=300, n_positive=700):
        images_paths_negative = glob.glob("./Concrete/Negative/Images/*.jpg",recursive=False)
        masks_paths_negative = glob.glob("./Concrete/Negative/Masks/*.jpg",recursive=False)
        images_paths_positive = glob.glob("./Concrete/Positive/Images/*.jpg",recursive=False)
        masks_paths_positive = glob.glob("./Concrete/Positive/Masks/*.jpg",recursive=False)
        self.transform = transform   
        self.images_paths = images_paths_negative[:n_negative] + images_paths_positive[:n_positive]
        self.masks_paths = masks_paths_negative[:n_negative] + masks_paths_positive[:n_positive]

    def __getitem__(self, index):       
        image = Image.open(self.images_paths[index])
        mask = Image.open(self.masks_paths[index])
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            #In the masks folder there is 3-channel mask, so for now transform is necessary
            if(mask.shape[0] != 1):
                transform = torchvision.transforms.Grayscale()
                mask = transform(mask)
        return image, mask
    
    def __len__(self):
        return len(self.images_paths)

#%% 
if __name__ == '__main__':
    fix_dataset()
    composed = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = ConcreteScarsDataset(transform=composed)
    dataloader = DataLoader(dataset=dataset, batch_size=10, num_workers=1)

    dataiter = iter(dataloader)
    image, mask = next(dataiter)
    print(image.shape, mask.shape)
#%%