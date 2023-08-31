# Load models.
import torch
#from deeplabv3plus import CustomDataset
import deeplabv3plus
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
#from unet_model import UNet
from backboned_unet.unet import Unet

import torchvision



augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class CustomDataset(Dataset):
    def __init__(self, x_path, y_path, mask_idx, transform = True, pure_eval = False):
        self.x_paths = sorted([str(fn) for fn in Path(x_path).rglob('*.png')])
        self.y_paths = sorted([str(fn) for fn in Path(y_path).rglob(f'*_{mask_idx}.png')])
        self.transform = transform
        self.pure_eval = pure_eval
        
    def __getitem__(self, idx):
        
        img = Image.open(self.x_paths[idx]).convert('RGB')
        if not self.pure_eval:
            msk = Image.open(self.y_paths[idx]).convert('L')
        if self.transform:
            img = augs(img)
            if not self.pure_eval:
                msk = torch.from_numpy(np.array(msk)).float()
                msk[msk > 127] = 255.
                msk[msk != 255] = 0.
                msk[msk == 255] = 1.
        if not self.pure_eval:
            return (img, msk)
        else :
            return img
    def __len__(self):
        return len(self.x_paths)


models = []
BONE_IDX = 7
PURE_EVAL = True

for i in range(8):
    cfg = yaml.load(open('cfg.yaml', "r"), Loader=yaml.Loader)
    #model = deeplabv3plus.DeepLabV3Plus(cfg)
    model = Unet(backbone_name='resnet101', classes=1)
    model.load_state_dict(torch.load(f'model-fold-{i}-TRAPEZIUM.pth'))
    print(f'Evaluate model {i}..')
    #eval
    if not PURE_EVAL: 
        dataset = CustomDataset('./wrist_dataset/AP_DRR_DATA/AP_DRR', './wrist_dataset/AP_mask', BONE_IDX, True)
        print(dataset.__len__())
        _, test = train_test_split(np.arange(dataset.__len__()), train_size= 0.8, test_size = 0.2)
        test_sampler = D.SubsetRandomSampler(test)
        testloader = DataLoader(dataset, batch_size=4, sampler = test_sampler)
        model.eval()
        with torch.no_grad():
            DL = 0.0
            for j, (img, msk) in enumerate(testloader):
                msk = msk.view(-1)
                out = model(img)
                out = out.view(-1)
                out[out > .5] = 1.
                out[out != 1.] = 0.
                DL += (2*(msk*out).sum()+1e-8)/(msk.sum() + out.sum() + 1e-8)
            print(f'{i} fold Average DL (Along Batch) : {DL / (j+1)}')
            
        # Save visualization plots
        print(f'Visualization From {i}th model..')
        fig, ax = plt.subplots(nrows=4, ncols=3)
        for j, (img, msk) in enumerate(testloader):
            
            o = model(img).numpy()
            o[o > .5] = 1
            o[o != 1] = 0
            o = o.astype(np.uint8).squeeze(1)
            print(o.shape)
            
            for k in range(img.shape[0]):
                ax[k,0].set_title(f'{k}th image in batch')
                ax[k,1].set_title(f'seg pred')
                ax[k,2].set_title(f'real mask')
                print(img[k].shape)
                ax[k,0].imshow(img[k].numpy().transpose((1,2,0)), cmap = 'gray')
                MSK = msk[k].numpy()
                #print(np.unique(msk[k]))

                OUT = o[k]
                print(np.unique(OUT))
                #OUT[OUT >= 0.5] = 1
                #OUT[OUT != 1] = 0
                #print(OUT.shape, MSK.shape)
            
                ax[k,1].imshow(OUT, cmap = 'PuBu')
                ax[k,2].imshow(MSK, cmap = 'OrRd')
                #ax[k,2].imshow(OUT, cmap = 'PuBu')
        plt.savefig(f'./vis_{i}_{BONE_IDX}.png')
    
