import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader
#from torchsummary import summary
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from backboned_unet.unet import Unet







class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=False, 
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x, need_fp=False):
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            return out, out_fp

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

#PyTorch
'''
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
'''

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()
    
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()




import torchvision



augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

crop_augs = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(128),
    torchvision.transforms.Resize(512)
])


class CustomDataset(Dataset):
    def __init__(self, x_path, y_path, mask_idx, transform = True):
        self.x_paths = sorted([str(fn) for fn in Path(x_path).rglob('*.png')])
        self.y_paths = sorted([str(fn) for fn in Path(y_path).rglob(f'*_{mask_idx}.png')])
        self.transform = transform
        '''
        global x_path_str
        x_path_str = []
        '''
        #print(len(self.x_paths), len(self.y_paths))
        xx = [x.split('/')[-1][:-4] for x in self.x_paths]
        yy = [y.split('/')[-1][0:8] for y in self.y_paths]
        
        for x in xx:
            if x not in yy :
                print(x)
        
    def __getitem__(self, idx):
        '''
        global x_path_str
        if len(x_path_str) == 4 :
            x_path_str = []
        x_path_str.append(self.x_paths[idx])
        #y_path_str = self.y_paths[idx]
        '''
        img = Image.open(self.x_paths[idx]).convert('RGB')
        msk = Image.open(self.y_paths[idx]).convert('L')
        if self.transform:
            #img = crop_augs(img)
            #msk = crop_augs(msk)
            img = augs(img)
            msk = torch.from_numpy(np.array(msk)).float()
            msk[msk > 127] = 255.
            msk[msk != 255] = 0.
            msk[msk == 255] = 1.
        return (img, msk)
    def __len__(self):
        return len(self.x_paths)
        
        

if __name__ == '__main__':
    print('test.')
    cfg = yaml.load(open('cfg.yaml', "r"), Loader=yaml.Loader)

    dataset = CustomDataset(x_path='./wrist_dataset/AP_DRR_DATA/AP_DRR', y_path='./wrist_dataset/AP_mask/', mask_idx= 8, transform=augs)
    print(dataset.__len__())
    train_idx, test_idx = train_test_split(list(range(dataset.__len__())), test_size = 0.2, shuffle = True)


    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=cfg['batch_size'], sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=1, sampler=test_subsampler)

    '''
    for t, (img, msk) in enumerate(trainloader):
        img_arr, msk_arr = img.numpy(), msk.numpy()
        print(img_arr.shape, msk_arr.shape)
        fig, ax = plt.subplots(nrows = 4, ncols = 2)
        for i, (img, msk) in enumerate(zip(img_arr, msk_arr)):
            ax[i][0].set_title(x_path_str[i])
            ax[i][0].imshow(img, cmap = 'gray')
            ax[i][1].imshow(msk, cmap = 'gray')
        plt.savefig(f'./fig{t}.png')

    for t, (img, msk) in enumerate(testloader):
        img_arr, msk_arr = img.numpy(), msk.numpy()
        print(img_arr.shape, msk_arr.shape)
        fig, ax = plt.subplots(nrows = 4, ncols = 2)
        for i, (img, msk) in enumerate(zip(img_arr, msk_arr)):
            ax[i][0].set_title(x_path_str[i])
            ax[i][0].imshow(img, cmap = 'gray')
            ax[i][1].imshow(msk, cmap = 'gray')
        plt.savefig(f'./testfig{t}.png')
    '''

    criterion = MixedLoss(10.0, 2.0)
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = lambda pred, targ : 0.5*nn.BCEWithLogitsLoss()(pred, targ) + 0.5*FocalLoss().forward(pred, targ) 
    #model = DeepLabV3Plus(cfg)
    #model = UNet(n_channels=3, n_classes=1)
    model = Unet(backbone_name='resnet101', classes=1)
    '''
    optimizer = torch.optim.SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                        'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    '''

    #optimizer = torch.optim.SGD(params = model.backbone.parameters(), lr=cfg['lr'], momentum = 0.9, weight_decay=1e-4)
    optimizer = torch.optim.SGD(params = model.parameters(), lr=cfg['lr'], momentum = 0.9, weight_decay=1e-4)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    '''
    #model.cuda()

    for epoch in range(1, cfg['epochs']):
        dsc = 0.0
        current_loss = 0.0
        model.cuda()
        model.train()
        for i, (input, target) in enumerate(trainloader):
            input = input.cuda()
            target = target.cuda()
            target = target[:, None, :, :]
            #print(np.unique(target.detach().cpu().numpy()))
            out = model(input)
            #print(np.unique(out.detach().cpu().numpy()))
            loss = criterion(target, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            out_flattened = out.view(-1)
            out_flattened[out_flattened>=.5] = 1.
            out_flattened[out_flattened!=1.] = 0.
            targ_flattened = target.view(-1)
            intersection = (out_flattened*targ_flattened).sum()
            dsc += (2*intersection + 1e-8) / (out_flattened.sum() + targ_flattened.sum() + 1e-8)
        
        print(f'Epoch {epoch} Loss : {current_loss / (i+1)}, Dice Loss : {1.0 - (dsc / (i+1))}')
        model.eval()
        dsc_val = 0.0
        for j, (input, target) in enumerate(testloader):
            #input = input
            #target = target
            model.cpu()
            out = model(input)
            out_flattened = out.view(-1)
            out_flattened[out_flattened>=.5] = 1.
            out_flattened[out_flattened!=1.] = 0.
            targ_flattened = target.view(-1)
            intersection = (out_flattened*targ_flattened).sum()
            dsc_val += (2*intersection + 1e-8) / (out_flattened.sum() + targ_flattened.sum() + 1e-8)
        print(f'Epoch {epoch} Dice Score : {(dsc_val / (j+1))}')

    print('Saving model..')    
    save_path = f'./model_trained.pth'
    torch.save(model.state_dict(), save_path)
    '''


    indices = list(range(dataset.__len__()))

    k_folds = 10
    results = {}

    #for epoch in range(1, cfg['epochs']):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(indices)):
        
    # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=cfg['batch_size'], sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=cfg['batch_size'], sampler=test_subsampler)
        
        # Init the neural network
        
        criterion = MixedLoss(10.0, 2.0)
        #criterion = nn.BCEWithLogitsLoss()
        #criterion = lambda pred, targ : 0.3*nn.BCEWithLogitsLoss()(pred, targ) + 0.7*FocalLoss().forward(pred, targ) 
        #model = DeepLabV3Plus(cfg)
        #model = UNet(n_channels=3, n_classes=1)
        model = Unet(backbone_name='resnet101', classes=1)
        
        '''
        optimizer = torch.optim.SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                        'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
        '''
        #optimizer = torch.optim.SGD(params = model.backbone.parameters(), lr=cfg['lr'], momentum = 0.9, weight_decay=1e-4)
        optimizer = torch.optim.SGD(params = model.parameters(), lr=cfg['lr'], momentum = 0.9, weight_decay=1e-4)
        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        
        model.train()
        
        # Run the training loop for defined number of epochs
        for epoch in range(1, cfg['epochs']):
            
            # Print epoch
            #print(f'Starting epoch {epoch}')

            # Set current loss value
            current_loss = 0.0
            dsc = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
            
                # Get inputs
                inputs, targets = data
                inputs = inputs.cuda()
                targets = targets.cuda()
                targets = targets[:,None,:,:]
            
                # Zero the gradients
                optimizer.zero_grad()
            
                # Perform forward pass
                outputs = model(inputs)
            
                # Compute loss
                loss = criterion(outputs, targets)
            
                # Perform backward pass
                loss.backward()
            
                # Perform optimization
                optimizer.step()
            
                # Print statistics
                current_loss += loss.item()
                
                #print(outputs.shape, targets.shape)
                
                # Compute Dice Score
                out_flattened = outputs.view(-1)
                out_flattened[out_flattened>=.5] = 1.
                out_flattened[out_flattened!=1.] = 0.
                targ_flattened = targets.view(-1)
                #print(out_flattened.shape, targ_flattened.shape)
                
                #print(np.unique(out_flattened.detach().cpu().numpy()))
                #print(np.unique(targ_flattened.detach().cpu().numpy()))
                
                intersection = (out_flattened*targ_flattened).sum()
                dsc += (2*intersection + 1e-8) / (out_flattened.sum() + targ_flattened.sum() + 1e-8)
                
            
            print(f'Epoch {epoch} Loss : {current_loss / (i+1)}, Dice Loss : {1.0 - (dsc / (i+1))}')
            
                
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')
        
        # Saving the model
        save_path = f'./model-fold-{fold}-TRAPEZIUM_REAL.pth'
        torch.save(model.state_dict(), save_path)

        model.eval()

        # Evaluation for this fold
        val_dsc = 0.0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):

                # Get inputs
                inputs, targets = data
                inputs = inputs.cuda()
                targets = targets.cuda()

                print(inputs.shape, targets.shape)

                # Generate outputs
                outputs = model(inputs)

                # Set total and correct
                
                
                out_flattened = outputs.view(-1)
                out_flattened[out_flattened>=.5] = 1.
                out_flattened[out_flattened!=1.] = 0.
                targ_flattened = targets.view(-1)
                #print(out_flattened.shape, targ_flattened.shape)
                intersection = (out_flattened*targ_flattened).sum()
                val_dsc += (2*intersection + 1e-8) / (out_flattened.sum() + targ_flattened.sum() + 1e-8)
                print('val_dsc : ', val_dsc)
                
                
            # Print accuracy
            print(f'Dice score for fold {fold} : {val_dsc / (i+1)}')
            print('--------------------------------')
            results[fold] = 100.0 * (val_dsc / (i+1))
        
        
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')



    '''

    for i in range(1):
        input = torch.randn(2, 3, 512, 512).cuda(0).float()
        targ = torch.randn(2, 1, 512, 512).cuda(0).float()
        out = model(input)
        
        #opt.zero_grad()
        #loss = criterion(targ, out)
        #loss.backward()
        #opt.step()
        
    print(out.shape)
    print(out)
    '''    
        
        
        

