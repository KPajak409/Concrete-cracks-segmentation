#%%
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import convert_image_dtype
import matplotlib.pyplot as plt

class ConcreteCracksModel(nn.Module):
    def __init__(self):
        super(ConcreteCracksModel, self).__init__()              
        self.conv1 = nn.Conv2d(3,3, 5, padding=2)
        self.enc1 = encoder(3, 64)
        self.enc2 = encoder(64, 128)
        self.enc3 = encoder(128, 256)

        self.dec1 = decoder(256,128)
        self.dec2 = decoder(128,64)
        self.dec3 = decoder(64,1)     
                    
    def forward(self, x, display_shape=False):

        x, indices1 = self.enc1(x)
        x, indices2 = self.enc2(x)
        x, indices3 = self.enc3(x)

        x = self.dec1(x, indices3)
        x = self.dec2(x, indices2)
        x = self.dec3(x, indices1)
        return x
        
    def info(self):
        print(self)
        print("Params: %i" % sum([param.nelement() for param in model.parameters()]))


class encoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(encoder, self).__init__()       
        self.conv = nn.Conv2d(ch_in,ch_out, (3,3), padding=1)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x, indices = self.pool(x)
        x = self.activation(x)
     
        return x, indices


class decoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(decoder, self).__init__()
        
        self.tconv = nn.ConvTranspose2d(ch_in,ch_out, (3,3), padding=1)
        self.unpool = nn.MaxUnpool2d(2)
        self.activation = nn.ReLU()
        
    def forward(self, x, indices):
        x = self.unpool(x, indices)
        x = self.activation(x)
        x = self.tconv(x)
        x = self.activation(x)
     
        return x

if __name__ == '__main__':
    input = output = 1
    model = ConcreteCracksModel()
    model.info()

    # test 
    # image set (batch, input[grey], size_x, size_y)
    image = torchvision.io.read_image("Concrete\Positive\Images\_002.png",)
    image = convert_image_dtype(image, dtype=torch.float32)
    mask = torchvision.io.read_image("Concrete\Positive\Masks\_002.png")
    mask = torchvision.transforms.Grayscale(1)(mask)
    mask = convert_image_dtype(mask, dtype=torch.float32)

    print("image = ", image.shape)
    y = model(image)
    print("y = ", y.shape)
    # plt.imshow(x.transpose(0,2)/255)
    # plt.show()
    with torch.no_grad():
        plt.imshow(y.transpose(0,2)/255)
    plt.show()