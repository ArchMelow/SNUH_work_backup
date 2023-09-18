# code written by Jaejin Lee (Sogang Univ. 20181671), with few references

import model.backbone.resnet as resnet
from model.backbone.xception import xception

import random
from copy import deepcopy
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
import matplotlib.pyplot as plt
import os
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import DataParallel
from apex.parallel import DistributedDataParallel as DDP
import apex.amp as amp


parser = argparse.ArgumentParser(
        description=None)
parser.add_argument(
    '--local-rank', type=int,
    help='local rank',
    default=0
)
parser.add_argument(
    '--path', type=str,
    help='root path')
parser.add_argument(
    '--epochs', type=int,
    help='epochs'
)
parser.add_argument(
    '--mask_idx', type=int,
    help='mask_idx'
)
parser.add_argument(
    '--x_path', type=str,
    help='x_path'
)
parser.add_argument(
    '--y_path', type=str,
    help='y_path'
)
parser.add_argument(
    '--label_name', type=str,
    help='label name'
)
parser.add_argument(
    '--x_u_path', type = str,
    help='unlabeled x path',
    default = None
)
parser.add_argument(
    '--lambda_weight', type=int,
    help = 'weight on unsupervised loss',
    default = 1
)

# Will change this to experiment.
parser.add_argument(
    '--confident_threshold', type = float,
    help = 'confidence threshold',
    default= 0.85
)

parser.add_argument(
    '--model_name', type = str,
    help = 'model name'
)


args = parser.parse_args()


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
import torchvision.transforms.functional as TF


base_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

# horizontal flip (p = 0.5)

def weak_augs(image, mask = None):
    dice = random.random()
    
    if dice > 0.5 :
        image = TF.hflip(image)
        if mask :
            mask = TF.hflip(mask)
    
    if mask :
        return image, mask
    else :
        return image

strong_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandAugment(num_ops = 2, magnitude = 9)    
])


class CustomDataset(Dataset):
    def __init__(self, x_path, y_path, mask_idx, transform = True, unlabeled = False, unlabeled_use_idx = 480, transform_type = 'weak'):
        self.transform = transform
        self.unlabeled = unlabeled
        self.transform_type = transform_type
        
        if not self.unlabeled:
            self.x_paths = sorted([str(fn) for fn in Path(x_path).rglob('*.png')])[:120]
            self.y_paths = sorted([str(fn) for fn in Path(y_path).rglob(f'*_{mask_idx}.png')])[:120]
        else :
            self.x_paths = sorted([str(fn) for fn in Path(x_path).rglob('*.png')])[:unlabeled_use_idx] # experiment with different label usage ratios.
        
        #print(len(self.x_paths), len(self.y_paths))
        
        '''
        xx = [x.split('/')[-1][:-4] for x in self.x_paths]
        yy = [y.split('/')[-1][0:8] for y in self.y_paths]
        
        for x in xx:
            if x not in yy :
                print(x)
        '''
      
        
    def __getitem__(self, idx):
        img = Image.open(self.x_paths[idx]).convert('RGB')
        
        if not self.unlabeled:
            msk = Image.open(self.y_paths[idx]).convert('L')
        
        if self.transform: # if it is True
            
            if self.unlabeled :
                if self.transform_type == 'weak':
                    img = base_augs(torchvision.transforms.Resize(128)(weak_augs(image = img)))
                    
                elif self.transform_type == 'strong':
                    img = base_augs(torchvision.transforms.Resize(128)(strong_augs(img)))
                    #print(np.unique(np.array(img)))
            
            else :     
                img, msk = weak_augs(image = img, mask = msk)
                img = torchvision.transforms.Resize(128)(base_augs(img))
                msk = torch.from_numpy(np.array(torchvision.transforms.Resize(128)(msk))).float()
                msk[msk > 127] = 255.
                msk[msk != 255] = 0.
                msk[msk == 255] = 1.
                
                
        if not self.unlabeled:
            return (img, msk)
        else :
            return img
        
    def __len__(self):
        return len(self.x_paths)
        
        

if __name__ == '__main__':
    print('semi test.')
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        print(os.environ['WORLD_SIZE'])
    
    if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

    torch.backends.cudnn.benchmark = True
    
    cfg = yaml.load(open('cfg.yaml', "r"), Loader=yaml.Loader)

    labeled_dataset = CustomDataset(x_path=args.x_path, y_path=args.y_path, mask_idx= args.mask_idx, transform=True)
    #labeled_dataset = CustomDataset(x_path='./wrist_dataset/AP_DRR_DATA/AP_DRR', y_path='./wrist_dataset/AP_mask/', mask_idx= 0, transform=True)
    
    #testset, trainset = train_test_split(labeled_dataset, train_size=0, test_size = 1/31)
    trainset = labeled_dataset # use all
    print(len(trainset))
    
    #print(labeled_dataset.__len__())

    #unlabeled_dataset = CustomDataset(x_path=args.x_u_path, y_path = None, mask_idx= None, transform = augs)
    
    # unlabeled dataset
    weak_unlabeled_dataset = CustomDataset(x_path='../FastGAN-pytorch/train_results/test1/eval_40000', y_path = None, mask_idx= None, transform = True, unlabeled=
                                      True, transform_type = 'weak')

    strong_unlabeled_dataset = CustomDataset(x_path='../FastGAN-pytorch/train_results/test1/eval_40000', y_path = None, mask_idx= None, transform = True, unlabeled=
                                      True, transform_type = 'strong')

    unlabeled_indices = list(range(weak_unlabeled_dataset.__len__())) # same for weak and strong augmented samples
    labeled_indices = list(range(trainset.__len__()))
    
    # check if ratio is int typed.
    assert weak_unlabeled_dataset.__len__() / trainset.__len__() - int(weak_unlabeled_dataset.__len__() / trainset.__len__()) == 0
    assert strong_unlabeled_dataset.__len__() / trainset.__len__() - int(strong_unlabeled_dataset.__len__() / trainset.__len__()) == 0
    
    # check if #strong samples = #weak samples 
    assert int(weak_unlabeled_dataset.__len__() / trainset.__len__()) == int(strong_unlabeled_dataset.__len__() / trainset.__len__())
    
    l_u_ratio = int(weak_unlabeled_dataset.__len__() / trainset.__len__())
    
    print(weak_unlabeled_dataset.__len__(), trainset.__len__())
    print(l_u_ratio)
    # 90: 1800 (1 : 20) ratio = 20


    #assert 1!=1

    k_folds = 12
    results = {}
    
    u_kfold = KFold(n_splits=k_folds, shuffle=True)
    l_kfold = KFold(n_splits=k_folds, shuffle=True)
    # Start print
    print('--------------------------------')

    unlabeled_iter = u_kfold.split(unlabeled_indices)

    # K-fold Cross Validation model evaluation
    for fold, (labeled_train_ids, labeled_val_ids) in enumerate(l_kfold.split(labeled_indices)):
        
        # next split of unlabeled dataset (to see all the unlabeled dataset splits.)
        unlabeled_idx = next(unlabeled_iter)
        unlabeled_ids = list(set(unlabeled_idx[0]) - set(unlabeled_idx[1]))
    # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        #print(labeled_train_ids.shape, labeled_val_ids.shape)
        #print(len(unlabeled_ids))
        
        # Sample elements randomly from a given list of ids, no replacement.
        labeled_train_subsampler = torch.utils.data.SubsetRandomSampler(labeled_train_ids)
        labeled_val_subsampler = torch.utils.data.SubsetRandomSampler(labeled_val_ids)
        
        unlabeled_subsampler = torch.utils.data.SubsetRandomSampler(unlabeled_ids)
        
        # labeled data loaders
        train_loader = torch.utils.data.DataLoader(
                        labeled_dataset, 
                        batch_size=cfg['batch_size'], sampler=labeled_train_subsampler)
        val_loader = torch.utils.data.DataLoader(
                        labeled_dataset,
                        batch_size=1, sampler=labeled_val_subsampler)
        
        # unlabeled data loaders
        weak_u_loader = torch.utils.data.DataLoader(
                        weak_unlabeled_dataset, 
                        batch_size=cfg['batch_size'] * l_u_ratio, sampler=unlabeled_subsampler)
        
        strong_u_loader = torch.utils.data.DataLoader(
                        strong_unlabeled_dataset, 
                        batch_size=cfg['batch_size'] * l_u_ratio, sampler=unlabeled_subsampler)
        
        # check unlabeled batches.
        '''
        for i,u in enumerate(u_loader):
            print(f'{i} th batch : shape of {u.shape}')
        '''  
        
        
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
        optimizer = torch.optim.SGD(params = model.parameters(), lr=cfg['lr'], nesterov= True, momentum = 0.9, weight_decay=1e-4)
        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
       
        model.cuda()
        
        model.train()
        
        #model, optimizer = amp.initialize(model, optimizer, opt_level = 'O1')
        
        # distribution learning (with apex.DDP)
        
        if args.distributed:
            model = DDP(model)
        
        # Run the training loop for defined number of epochs
        #for epoch in range(1, cfg['epochs']):
        for epoch in range(1, args.epochs):   # change for experiment
            
            # newly define unlabeled loader iterators. (reset the iteration ptr)
            
            weak_u_loader_iter = iter(deepcopy(weak_u_loader))
            strong_u_loader_iter = iter(deepcopy(strong_u_loader))
            
                        
            current_loss = 0.0
            dsc = 0.0

            # Iterate over the DataLoader for training data
            for i, labeled_data in enumerate(train_loader, 0):
                
                # Get inputs
                labeled_inputs, targets = labeled_data
                
                # log the sizes of the batches distributed to each GPU.
        
                #print(f"IMAGE SIZE: {labeled_inputs.shape}, MSK SIZE : {targets.shape}")        
                
                
                # bring next weakly augmented unlabeled data and strongly augmented unlabeled data
                weak_unlabeled_inputs = next(weak_u_loader_iter).cuda()
                strong_unlabeled_inputs = next(strong_u_loader_iter).cuda()
                
                # check one sample for image/mask consistency after augmentation
                '''
                fig, ax = plt.subplots(nrows=4, ncols=2)
                
                for idx, (inp, targ) in enumerate(zip(labeled_inputs, targets)):
                    ax[idx][0].imshow(inp.cpu().numpy().transpose(1,2,0), cmap = 'gray')
                    ax[idx][1].imshow(targ, cmap = 'gray')
                plt.savefig('../saved.png')
                
                assert 1!=1
                '''
                
                labeled_inputs = labeled_inputs.cuda()
                targets = targets.cuda()
                targets = targets[:,None,:,:]
        
                # Perform forward pass (final activation is not Sigmoid, so apply Sigmoid)
                pred = nn.Sigmoid()(model(labeled_inputs)).clone()
            
                #print(np.unique(pred.detach().cpu().numpy()))
                #assert 1!=1
            
                # Compute SUPERVISED loss.
                sup_loss = criterion(pred, targets)
            
                pseudolabel = nn.Sigmoid()(model(weak_unlabeled_inputs))
                #print(np.unique(pseudolabel.detach().cpu().numpy()))

                # thresholding (pseudo_label)
                pseudolabel = pseudolabel.clone()
                pseudolabel[pseudolabel >= args.confident_threshold] = 255
                pseudolabel[pseudolabel < args.confident_threshold] = 0
                pseudolabel[pseudolabel == 255] = 1
                #print(np.unique(pseudolabel.detach().cpu().numpy()))
                
                # Compute UNSUPERVISED loss.
                unsup_loss = criterion(pseudolabel, nn.Sigmoid()(model(strong_unlabeled_inputs)))
                
                # Compute TOTAL loss.
                loss = sup_loss + args.lambda_weight * unsup_loss
                
                # Zero the gradients
                optimizer.zero_grad()
            
                # Perform backward pass
                loss.backward()
            
                # Perform optimization
                optimizer.step()
            
                # statistics
                current_loss += loss.item()
                
                # outputs as the prediction after backprop
                outputs = nn.Sigmoid()(model(labeled_inputs)).clone()
                
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
                
            
            print(f'Epoch {epoch} Loss on GPU {args.local_rank}: {current_loss / (i+1)}, Dice Loss : {1.0 - (dsc / (i+1))}')
            
                
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')
        
        
        # Saving the model (only if rank is 0)
        if args.local_rank == 0:
            save_path = f'./{args.model_name}-model-fold-{fold}-{args.label_name}.pth'
            #save_path = f'./model-fold-{fold}-ULNA_SEM.pth'
            torch.save(model.state_dict(), save_path)
        
        
        model.eval()


        # erase this after testing
        #DIR_STR = 'sem_test' + str(fold)
        
        DIR_STR = None
        
        if args.local_rank == 0:
            DIR_STR = args.path + str(fold) + f'-{args.local_rank}'
        #DIR_STR = args.path + str(fold) + '-single'
        else:
            continue # do not save figure.
        
        # make folder to save seg results.
        if not os.path.exists(DIR_STR):
            os.mkdir(DIR_STR)
        
        # Evaluation for this fold
        val_dsc = 0.0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(val_loader, 0):

                fig, ax = plt.subplots(nrows = 1, ncols = 3)
                
                # Get inputs
                inputs, targets = data
                inputs = inputs.cuda()
                targets = targets.cuda()

                print(inputs.shape, targets.shape)    

                ax[0].imshow(inputs[0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
                ax[0].set_label('Test Data')
                ax[1].imshow(targets[0].detach().cpu().numpy(), cmap='PuBu')
                ax[1].set_label('Ground Truth')
                


                # Generate outputs
                outputs = model(inputs)
                outputs[outputs >= .5] = 1.
                outputs[outputs != 1.] = 0.
                #print('out shape : ', outputs.shape)
                ax[2].imshow(outputs[0][0].detach().cpu().numpy(), cmap = 'PuBu')
                ax[2].set_label('Prediction Mask')
                
                # save figure only if the device's rank == 0
                if args.local_rank == 0:
                    plt.savefig(DIR_STR + f'/{i}.jpg')
                

                plt.close()
                
                # Set total and correct
                
                
                
                out_flattened = outputs.view(-1)
                #out_flattened[out_flattened>=.5] = 1.
                #out_flattened[out_flattened!=1.] = 0.
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
    average_result = sum / len(results.items())
    
    result_filename = f'val_res_{args.label_name}_from_gpu_{args.local_rank}.txt'
    with open(result_filename, 'w') as fp:
        fp.write(f'Validation average: {average_result}')


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
        
        
        

