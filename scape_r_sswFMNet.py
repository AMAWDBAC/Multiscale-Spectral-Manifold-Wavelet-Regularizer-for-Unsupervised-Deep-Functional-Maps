import os
import sys
from itertools import permutations
import os.path as osp
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.io as sio

sys.path.append(osp.join(os.getcwd(),'src'))
import diffusion_net

from faust_dataset import MatchingDataset

from src.diffusion_net.utils import nn_search

# erro

def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err
# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
# n_class = 8

# model 
# input_features = args.input_features # one of ['xyz', 'hks']
input_features = 'WKS'
k_eig = 128
Nf=6
n_fmap=100
n_cfmap=30

# training settings
train = not args.evaluate
n_epoch = 5
lr = 1e-3
decay_every = 50
decay_rate = 0.5

# Important paths
base_path = osp.dirname(__file__)
dataset_path = osp.join(base_path, 'data','faust_5k')
pretrain_path = osp.join(dataset_path, "pretrained_models/t_hk1104_faust_{}_4x128.pth".format(input_features))
# model_save_path = os.path.join(dataset_path, 'saved_models','t_hk1104_faust_{}_4x128.pth'.format(input_features))


# Load the train dataset
if train:
    train_dataset = MatchingDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    now = datetime.now()
    folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    model_save_dir=osp.join(dataset_path,'save_models',folder_str)
    diffusion_net.utils.ensure_dir_exists(model_save_dir)

# === Create the model

C_in={'xyz':3, 'hks':16, 'WKS':128}[input_features] # dimension of input features


model = diffusion_net.layers.SSWFMNet(C_in=C_in,C_out=256,n_fmap=n_fmap)

model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_iter_loss=[]

def train_epoch(epoch):

    global iter
    global min_erro
    if epoch > 1 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_num = 0
    for data in tqdm(train_loader):

        # Get data
        verts_x,descs_x,massvec_x,evals_x,evecs_x,gradX_x,gradY_x,L_x,\
            verts_y,descs_y,massvec_y,evals_y,evecs_y,gradX_y,gradY_y,L_y=data
        
        
        # updata gs_x, gs_y
        gs_x=diffusion_net.utils.Meyer(evals_x[n_fmap-1],Nf=Nf)(evals_x[:n_fmap]).float()
        gs_y=diffusion_net.utils.Meyer(evals_y[n_fmap-1],Nf=Nf)(evals_y[:n_fmap]).float()
        
        
        # Move to device
        verts_x=verts_x.to(device)
        descs_x=descs_x.to(device)
        massvec_x=massvec_x.to(device)
        evecs_x=evecs_x.to(device)
        evals_x=evals_x.to(device)
        gs_x=gs_x.to(device)
        gradX_x=gradX_x.to(device)
        gradY_x=gradY_x.to(device) #[N,N]
        L_x=L_x.to(device)

        verts_y=verts_y.to(device)
        descs_y=descs_y.to(device)
        massvec_y=massvec_y.to(device)
        evecs_y=evecs_y.to(device)
        evals_y=evals_y.to(device)
        gs_y=gs_y.to(device)
        gradX_y=gradX_y.to(device)
        gradY_y=gradY_y.to(device)
        L_y=L_y.to(device)

    
        
        
        # Apply the model
        loss= model(descs_x,massvec_x,evals_x,gs_x,evecs_x,gradX_x,gradY_x,\
                descs_y,massvec_y,evals_y,gs_y,evecs_y,gradX_y,gradY_y)

        # Evaluate loss
        loss.requires_grad_(True)
        loss.backward()
        
        # track accuracy
        total_loss+=loss.item()
        total_num += 1
        iter+=1
        train_iter_loss.append(loss.item())

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

        if total_num%100==0:
            print('Iterations: {:02d}, train loss: {:.4f}'.format(total_num, total_loss / total_num))
            total_loss=0.0
            total_num=0

        if iter%4000==0:
            avg_erro=test()
            print(avg_erro)
            model_save_path = osp.join(model_save_dir, 'ckpt_ep{best}.pth')
            if avg_erro < min_erro:
                    torch.save(model.state_dict(), model_save_path)
                    min_erro = avg_erro 
def test():
    test_dataset = MatchingDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True)
    test_loader = DataLoader(test_dataset, batch_size=None)

    file = osp.join(dataset_path, 'files_test.txt')
    with open(file, 'r') as f:
        names = [line.rstrip() for line in f]

    combinations = list(permutations(range(len(names)), 2))

    model.eval()
    with torch.no_grad():
        count=0
        erro=0
        for data in tqdm(test_loader):

            verts_x,descs_x,massvec_x,evals_x,evecs_x,gradX_x,gradY_x,L_x,\
                verts_y,descs_y,massvec_y,evals_y,evecs_y,gradX_y,gradY_y,L_y,=data


            # updata gs_x, gs_y
            gs_x=diffusion_net.utils.Meyer(evals_x[n_fmap-1],Nf=Nf)(evals_x[:n_fmap]).float()
            gs_y=diffusion_net.utils.Meyer(evals_y[n_fmap-1],Nf=Nf)(evals_y[:n_fmap]).float()


            # Move to device
            verts_x=verts_x.to(device)
            descs_x=descs_x.to(device)
            massvec_x=massvec_x.to(device)
            evecs_x=evecs_x.to(device)
            evals_x=evals_x.to(device)
            gs_x=gs_x.to(device)
            gradX_x=gradX_x.to(device)
            gradY_x=gradY_x.to(device) #[N,N]
            L_x=L_x.to(device)

            verts_y=verts_y.to(device)
            descs_y=descs_y.to(device)
            massvec_y=massvec_y.to(device)
            evecs_y=evecs_y.to(device)
            evals_y=evals_y.to(device)
            gs_y=gs_y.to(device)
            gradX_y=gradX_y.to(device)
            gradY_y=gradY_y.to(device)
            L_y=L_y.to(device)

            # Apply the model
            p12= model.model_test(descs_x,massvec_x,evals_x,gs_x,evecs_x,gradX_x,gradY_x,\
                    descs_y,massvec_y,evals_y,gs_y,evecs_y,gradX_y,gradY_y)


            p12=p12.cpu()
            p12=np.array(p12)

            idx1,idx2=combinations[count]
            data_x=sio.loadmat("path/dist/"+names[idx2]+'.mat')
            dist_x=data_x["dist"]

            corr_x_file="path/corres/"+names[idx2]+'.vts'
            arrays_corr_x = []
            with open(corr_x_file, 'r') as file:
               for line in file:
                 numbers = int(line.strip())
                 arrays_corr_x.append(numbers)
            arrays_corr_x = [x  for x in arrays_corr_x]
            arrays_corr_x = np.array( arrays_corr_x)

            corr_y_file="path/corres/"+names[idx1]+'.vts'
            arrays_corr_y = []
            with open(corr_y_file, 'r') as file:
               for line in file:
                 numbers = int(line.strip())
                 arrays_corr_y.append(numbers)
            arrays_corr_y = [x  for x in arrays_corr_y]
            arrays_corr_y = np.array( arrays_corr_y)
            erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)
            count+=1
            erro=erro+erro_i

        avg_erro=erro/count
        return avg_erro
if train:
    print("Training...")
    iter=0
    min_erro=100
    # min_loss=1e10
    for epoch in range(n_epoch):
        # torch.cuda.empty_cache()
        # start_time = time.time()
        train_epoch(epoch)


