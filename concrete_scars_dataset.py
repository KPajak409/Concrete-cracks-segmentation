#%%
import glob, os
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import convert_image_dtype
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#If you downloaded this dataset from kaggle https://www.kaggle.com/datasets/yatata1/crack-dataset status as of July 3, 2023
#At first you should fix some things, function below will do it for you
def fix_dataset():
    #1. Change folder name at path ./Concrete/Negative/Mask to Masks
    mask_path = './Concrete/Negative/Mask'
    if(os.path.isdir(mask_path)):
        os.rename(mask_path, mask_path + 's')
        print('Path modified')

    #We are ignoring *.png files because of their different size
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
    def __init__(self, transform=None):
        self.transform = transform
        images_paths_negative = glob.glob("./Concrete/Negative/Images/*.jpg",recursive=False)
        masks_paths_negative = glob.glob("./Concrete/Negative/Masks/*.jpg",recursive=False)
        images_paths_positive = glob.glob("./Concrete/Positive/Images/*.jpg",recursive=False)
        masks_paths_positive = glob.glob("./Concrete/Positive/Masks/*.jpg",recursive=False)
        self.images_paths = images_paths_negative[:900] + images_paths_positive[:1500]
        self.masks_paths = masks_paths_negative[:900] + masks_paths_positive[:1500]
        self.n_samples = len(self.images_paths) + len(self.masks_paths)


    def __getitem__(self, index):       
        image = Image.open(self.images_paths[index])
        mask = Image.open(self.masks_paths[index])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    
    def __len__(self):
        return self.n_samples

#%% 
if __name__ == '__main__':
    fix_dataset()
    composed = transforms.Compose([transforms.ToTensor()])
    dataset = ConcreteScarsDataset(transform=composed)
    dataloader = DataLoader(dataset=dataset, batch_size=32, num_workers=1)

    dataiter = iter(dataloader)
    image, mask = next(dataiter)
    print(image.shape, mask.shape)
#%%