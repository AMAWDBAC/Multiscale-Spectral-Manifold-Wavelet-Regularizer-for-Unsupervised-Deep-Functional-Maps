import torch.nn.functional as F
import sys
import os
import random

import scipy
import scipy.sparse.linalg as sla
from scipy.linalg import pinv
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)
from diffusion_net.utils import nn_search
import numpy as np
import torch
import torch.nn as nn

from .utils import toNP
from .geometry import to_basis, from_basis

class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes 
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values 
    """

    def __init__(self, C_inout, method='spectral'):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)
        

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            # Diffuse
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex 
            x_diffuse = from_basis(x_diffuse_spec, evecs)
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.
    
    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots 
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if(self.with_gradient_rotations):
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
            vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
        else:
            vectorsBreal = self.A(vectors[...,0])
            vectorsBimag = self.A(vectors[...,1])

        dots = vectorsA[...,0] * vectorsBreal + vectorsA[...,1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True, 
                 diffusion_method='spectral',
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        
        self.MLP_C = 2*self.C_width
      
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width
        
        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)


    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
                x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out


class DiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices', mlp_hidden_dims=None, dropout=True, 
                       with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral'):   
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(C_width = C_width,
                                      mlp_hidden_dims = mlp_hidden_dims,
                                      dropout = dropout,
                                      diffusion_method = diffusion_method,
                                      with_gradient_features = with_gradient_features, 
                                      with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)
        
        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices': 
            x_out = x
        
        elif self.outputs_at == 'edges': 
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        
        elif self.outputs_at == 'faces': 
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        
        elif self.outputs_at == 'global_mean': 
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant. 
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out



from .utils import dist_mat

def sinkhorn(d, sigma=0.1, num_sink=10):
    d = d / d.mean()
    log_p = -d / (2*sigma**2)

    for it in range(num_sink):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
    p = torch.exp(log_p)
    # log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    # self.p_adj = torch.exp(log_p).transpose(0, 1)

    return p


def feat_correspondences(emb_x, emb_y):
    d = dist_mat(emb_x, emb_y, False)
    return sinkhorn(d)


def MWP(massvec_x, evecs_x, gs_x, evecs_y, gs_y, P):
    # input:
    #   massvec_x/y: [M/N,]
    #   evecs_x/y: [M/N,Kx/Ky]
    #   gs_x/y: [Nf,Kx/Ky]

    # compute MWP functional map
    C = evecs_x.transpose(-2, -1)@(massvec_x.unsqueeze(-1)*(P@evecs_y))
    C_new = torch.zeros_like(C)

    gs_x = gs_x.unsqueeze(-1)  # [Nf,Kx]->[Nf,Kx,1]
    gs_y = gs_y.unsqueeze(-1)  # [Nf,Ky]->[Nf,Ky,1]

    # MWP filters
    Nf = gs_x.size(0)
    for s in range(Nf):
        C_new += gs_x[s]*C*gs_y[s].transpose(-2, -1)

    return C_new


# version without batch
class ResidualBlock(torch.nn.Module):
    """Implement one residual block as presented in FMNet paper."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, out_dim)
        self.bn1 = torch.nn.BatchNorm1d(out_dim)
        self.fc2 = torch.nn.Linear(out_dim, out_dim)
        self.bn2 = torch.nn.BatchNorm1d(out_dim)

        if in_dim != out_dim:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.BatchNorm1d(out_dim)  # non implemented in original FMNet paper, suggested in resnet paper
            )
        else:
            self.projection = None

    def forward(self, x):
        x_res = F.relu(self.bn1(self.fc1(x)))
        x_res = self.bn2(self.fc2(x_res))
        if self.projection:
            x = self.projection(x)
        x_res += x
        return F.relu(x_res)


class RefineNet(torch.nn.Module):
    """Implement the refine net of FMNet. Take as input hand-crafted descriptors.
       Output learned descriptors well suited to the task of correspondence"""

    def __init__(self, n_residual_blocks=7, in_dim=352):
        super().__init__()
        model = []
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_dim, in_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """One pass in refine net.

        Arguments:
            x {torch.Tensor} -- input hand-crafted descriptor. Shape: batch-size x num-vertices x num-features

        Returns:
            torch.Tensor -- learned descriptor. Shape: batch-size x num-vertices x num-features
        """
        return self.model(x)


    
    def sinkhorn(self, d, sigma=0.1, num_sink=10):
        d = d / d.mean()
        log_p = -d / (2*sigma**2)
        
        for it in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        self.p = torch.exp(log_p)
        # log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        # self.p_adj = torch.exp(log_p).transpose(0, 1)
    
    def feat_correspondences(self, emb_x, emb_y):
        d = dist_mat(emb_x, emb_y, False)
        self.sinkhorn(d)

    def MWP(self, gs_x, gs_y):
        # input:
        #   massvec_x/y: [M/N,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]

        # compute MWP functional map
        # C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        C_new=torch.zeros_like(self.C)

        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]

        # MWP filters
        Nf=gs_x.size(0)
        for s in range(Nf):
            C_new+=gs_x[s]*self.C*gs_y[s].transpose(-2,-1)
        
        self.C=C_new


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2)
        return loss





# spatial and spectral  with muliscale wavelets as regularizer
class SSWFMNet(nn.Module):
    def __init__(self, C_in,  C_out, n_fmap):
        super().__init__()
        self.n_fmap=n_fmap
        self.feat_extrac=DiffusionNet(C_in=C_in, C_out=C_out)
        self.frob_loss = FrobeniusLoss()
        self.lambda_param=1e-2


        
        
    def forward(self,descs_x,massvec_x,evals_x,gs_x,evecs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,gs_y,evecs_y,gradX_y,gradY_y):#
        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)      
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)
        feat_x = feat_x / feat_x.norm(dim=1, keepdim=True)
        feat_y = feat_y / feat_y.norm(dim=1, keepdim=True)

        evecs_x=evecs_x[:,:self.n_fmap]
        evecs_y=evecs_y[:,:self.n_fmap]
        
    
        evecs_trans_x=(evecs_x*massvec_x.unsqueeze(-1)).t()
   
        evecs_trans_y=(evecs_y*massvec_y.unsqueeze(-1)).t()


        C21=self.waveletCReg(feat_y, feat_x, evecs_trans_y, evecs_trans_x, gs_y, gs_x)
       

        Pxy=self.feat_correspondences(feat_x,feat_y)
    
        loss=self.frob_loss(evecs_x, Pxy@evecs_y@C21.transpose(-2,-1))       
        return loss

    def model_test(self,descs_x,massvec_x,evals_x,gs_x,evecs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,gs_y,evecs_y,gradX_y,gradY_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)
        feat_x = feat_x / feat_x.norm(dim=1, keepdim=True)
        feat_y = feat_y / feat_y.norm(dim=1, keepdim=True)
      
        evecs_x=evecs_x[:,:self.n_fmap]
        evecs_y=evecs_y[:,:self.n_fmap]
    
        evecs_trans_x=(evecs_x*massvec_x.unsqueeze(-1)).t()
        evecs_trans_y=(evecs_y*massvec_y.unsqueeze(-1)).t()


        Pxy=self.feat_correspondences(feat_x,feat_y)#iso

        Cyx = evecs_trans_x @ (Pxy @ evecs_y)#iso

        p12 = nn_search(evecs_y@(Cyx.t()), evecs_x)#iso   

        return p12
    
    def MWP(self, gs_x, gs_y):
        # input:
        #   massvec_x/y: [M/N,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]

        # compute MWP functional map
        # C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        C_new=torch.zeros_like(self.C)

        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]

        # MWP filters
        Nf=gs_x.size(0)
        for s in range(Nf):
            C_new+=gs_x[s]*self.C*gs_y[s].transpose(-2,-1)
        
        self.C=C_new
        

    def sinkhorn(self, d, sigma=0.1, num_sink=10):
        d = d / d.mean()
        log_p = -d / (2*sigma**2)
        
        for it in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        # self.p = torch.exp(log_p)
        p = torch.exp(log_p)
        return p
        # log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        # self.p_adj = torch.exp(log_p).transpose(0, 1)
    
    def feat_correspondences(self, emb_x, emb_y):
        d = dist_mat(emb_x, emb_y, False)
        return self.sinkhorn(d)
        

    
    def waveletCReg(self, feat_x, feat_y, evecs_trans_x, evecs_trans_y, gs_x, gs_y):
        # feat_x/y; [nx/ny,p]
        # gs_x/y; [s,kx/ky]
        # evecs_trans_x/y: [kx/ky,nx/ny]sinkhorn
        # output: Cxy

        #计算傅里叶(谱)系数
        A = torch.matmul(evecs_trans_x, feat_x)
        B = torch.matmul(evecs_trans_y, feat_y)
        scaling_factor = max(torch.max(gs_x), torch.max(gs_y))
        gs_x, gs_y = gs_x / scaling_factor, gs_y / scaling_factor
        #构造W矩阵
        Ds=0

        Nf = gs_x.size(0)
        for s in range(Nf):


            D=(gs_x[s].unsqueeze(0)-gs_y[s].unsqueeze(1))**2 
            Ds+=D
            

        #计算C
        A_A_T=torch.matmul(A,A.t())
        A_B_T=torch.matmul(A,B.t())
        
        C=[]
        for i in range(gs_y.size(1)):
            D_i=torch.diag(Ds[i])
            # C_i= torch.matmul(torch.inverse(A_A_T +lambda_param * D_i), A_B_T[:, i])
            C_i=torch.linalg.lstsq(A_A_T+self.lambda_param*D_i,A_B_T[:,i]).solution
            C.append(C_i.unsqueeze(0))

        C=torch.cat(C,dim=0)

        return C

  