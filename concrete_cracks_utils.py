#%%
import torch, torchvision
from concrete_cracks_dataset import ConcreteCracksDataset
from torch.utils.data import DataLoader
import torchmetrics.classification as tm
from Unet import Unet
from tqdm import tqdm
from torchvision.transforms import ToTensor

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def train_fn(loader, model, device, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        # forward
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def log_images(loader, model, device, epoch, experiment):
    model.eval()
    for i, (image, mask) in enumerate(loader):
        with torch.no_grad():
            image = image.to(device)
            predict = model(image)
        #converting dimensions to HWC
        img = image[0].permute(1,2,0).cpu()
        msk = mask[0].permute(1,2,0)
        predict = predict[0].permute(1,2,0).cpu()
        #log images to comet
        step = (epoch+1)*len(loader)
        experiment.log_image(img, f'image{i+1}', step=step)
        experiment.log_image(msk, f'mask{i+1}', step=step)
        experiment.log_image(predict, f'predict{i+1}', step=step) #*255 makes it more visible in comet
    model.train()



def check_accuracy(loader, model, device):
    model.eval()
    f1_score = tm.BinaryF1Score().to(device)
    f1_scores = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            y = (y > 0.5).float()
            f1_scores.append(f1_score(preds, y).item())
    model.train()
    return sum(f1_scores)/len(loader)

#%%
if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device('cpu')
    
    model = Unet().to(DEVICE)
    model.load_state_dict(torch.load('./model_versions/unet50epochs'))

    train_dataset = ConcreteCracksDataset(500, 500, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=5,
        shuffle=True,
        num_workers=1
    )

    val_dataset = ConcreteCracksDataset(0, 10, skip=0, transform=ToTensor())
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=1
    )

    print(check_accuracy(val_loader,model, DEVICE))



# %%
