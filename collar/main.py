
import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, BaseFinetuning, LearningRateMonitor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import numpy as np
import argparse
import sys
import warnings
sys.path.append('./')

from model.COLLAR import COLLAR
from data.TS import TSDataModule
from model.metrics import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
b=32
init_scale=0.99
m=3330
n=1
IK=torch.eye(32).to(device) 


Q1=np.random.standard_normal((3330,370))
Q2,_=np.linalg.qr(Q1)
Q=Q2.astype(np.float32)
Q=torch.from_numpy(Q).to(device)


def inv_transform(scaler: StandardScaler, Y: torch.FloatTensor):
    '''pytorch version and takes (n_features, n_samples)
    '''
    dev = Y.device
    scale = torch.from_numpy(scaler.scale_).to(dev).unsqueeze(-1)
    mean = torch.from_numpy(scaler.mean_).to(dev).unsqueeze(-1)
    Y = Y * scale + mean
    
    return Y


def basisAttn(basis: torch.Tensor):
    '''(#entities, #features) -> (#entities, #entities)
    '''
    Nt, E = basis.shape
    attn = basis @ basis.T
    attn = attn / math.sqrt(E)
    attn = F.softmax(attn, dim=-1)
    return attn

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


class dynamics(pl.LightningModule):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data= torch.mm(U, V.t()) * init_scale
        
  
    def forward(self, x):
        x = self.dynamics(x)

        return x


class COLLAR(pl.LightningModule):
    def __init__(
        self,
        num_series=370,
        enc_channels=[64, 16],
        rnn_hidden=32,
        rnn_layers=3,
        dropout_X=0.2,
        rg=64,
        reg=0.5,
        lr=0.0001,
        lr_decay=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()  
        num_layers = len(enc_channels)
        assert num_layers > 1
        layers = []
        for i in range(num_layers):
            in_feat = enc_channels[i-1] if i > 0 else num_series
            out_feat = enc_channels[i]
            layers.append(nn.Linear(in_feat, out_feat))
            if i != num_layers-1:  # act except last layer
                layers.append(nn.ELU())
        self.encoder = nn.Sequential(*layers)
        self.encoder1=nn.Sequential(nn.Linear(m*n, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, b)
                                     )
        self.dynamics = dynamics(b, init_scale)                              
        # temporal model
        # input of shape (L, N=N_latent, H_in=1)
        self.Xseq = nn.GRU(
            input_size=1,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout_X
        )
        self.Xrg = rg
        self.distribution_mu = nn.Linear(rnn_hidden, 1)
        
        dec_channels = enc_channels[::-1]
        layers = []
        for i in range(num_layers):
            in_feat = dec_channels[i]
            out_feat = dec_channels[i+1] if i != num_layers-1 else num_series
            layers.append(nn.Linear(in_feat, out_feat))
            if i != num_layers-1:  # act except last layer
                layers.append(nn.ELU())
        self.decoder = nn.Sequential(*layers)
        self.decoder1=nn.Sequential(nn.Linear(b, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, m*n)
                                     ) 

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "hparams" in name:
                raise AttributeError()
            return getattr(self.hparams, name)

    @property
    def num_latent(self):
        return self.hparams.enc_channels[-1]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        (_, inpY), (_, tgtY), _, wnd_start = batch  # use normed data
        pred_steps = tgtY.shape[-1]
        assert inpY.shape[-1] == self.Xrg + (pred_steps-1)  # Yseq cond_steps
        wnd_len = self.Xrg + pred_steps  # inpY.dim_time + 1
        num_series = inpY.shape[0]
        
        Y = torch.cat((inpY, tgtY[:, [-1]]), -1)  
        
        X = self.encoder(Y.T)  # L, N_latent
        rnn_out, _ = self.Xseq(X[:-1, :, None])  # L, N_latent, H_hidden
        X_h = self.distribution_mu(rnn_out).squeeze(-1)  # L, N_latent
        X_h = torch.cat((X[:self.Xrg, ...], X_h[-pred_steps:, ...]), 0)
        Y_h = self.decoder(X_h)  # L, N_input
        Y_h = Y_h.T  # N_input, L


        Y1=torch.mm(Y.T,Q.T)
        X2=self.encoder1(Y1)
        X_I=self.dynamics(X2[:-1,:])
        X_I=torch.cat((X2[:1,:],X_I),0)
        Y_I=self.decoder1(X_I) 
        Y_I=torch.mm(Y_I,Q)
        Y_I=Y_I.T
        A=self.dynamics.dynamics.weight.to(device)    

        l1loss = nn.L1Loss()
        recov_loss = l1loss(Y_h, Y)
        koopman_loss=l1loss(Y_h,Y_I)   
        l2loss = nn.MSELoss()
        identity_loss=(torch.norm(torch.mm(A,A.T)-IK))**2
        pre_loss = l2loss(X_h[-pred_steps:, ...], X[-pred_steps:, ...])
        
        loss = recov_loss+(5e-2)*koopman_loss+(5e-1)*identity_loss+pre_loss
        if self.current_epoch > 10 and loss.item() > 10:
            self.print(f"abnormal loss at batch {batch_idx} (starting from {wnd_start})")
        self.log("loss/train", loss, 
                 on_step=True, on_epoch=False, batch_size=1)
        self.log("loss/train_recov", recov_loss,
                 on_step=True, on_epoch=False, batch_size=1)
        self.log("loss/train_pre", pre_loss,
                 on_step=True, on_epoch=False, batch_size=1)
        self.log("loss/train_koopman", koopman_loss,
                 on_step=True, on_epoch=False, batch_size=1)
        self.log("loss/train_identity", identity_loss,
                 on_step=True, on_epoch=False, batch_size=1)            
          
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.hparams.lr)

        min_lr = self.lr * (self.lr_decay ** 3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', self.lr_decay, patience=3, cooldown=2, min_lr=min_lr)

        return {
            'optimizer': optimizer, 'lr_scheduler': {
                "scheduler": lr_scheduler,
                "monitor": "loss/val", }
        }

    def forward(self, inpY, wnd_start, pred_steps):
       
        assert inpY.ndim == 2
        assert inpY.shape[-1] == self.Xrg

       
        X = self.encoder(inpY.T)  # L, N_latent
        X_hs = []  # no teacher forcing
        rnn_out, hidden = self.Xseq(X[:, :, None])  # L, N_latent
        X_h = self.distribution_mu(rnn_out[[-1], ...])  # 1, N_latent, 1
        X_hs.append(X_h)
        for i in range(pred_steps-1):
            rnn_out, hidden = self.Xseq(X_h, hidden)
            X_h = self.distribution_mu(rnn_out[[-1], ...])  # 1, N_latent
        X_h = torch.cat(X_hs, 0).squeeze(-1)  # L, N_latent
        Y_h = self.decoder(X_h)  # L, N_input
        Y_h = Y_h.T  # N_input, L

       
        Y1=torch.mm(inpY.T,Q.T)
        X2=self.encoder1(Y1)
        X_Is = []
        X_I=self.dynamics(X2)
        X_I=X_I[[-1], ...]
        X_Is.append(X_I)
        for i in range(pred_steps-1):
            X_I=self.dynamics(X_I)
            X_Is.append(X_I)
        X_I=torch.cat(X_Is, 0) 
        Y_I = self.decoder1(X_h)  
        Y_I = Y_I.T  # N_input

        return Y_h, X_h,Y_I,X_I
     

    def validation_step(self, batch, batch_idx):
        (_, inpY_n), (tgtY, tgtY_n), scaler, wnd_start = batch
        pred_steps = tgtY.shape[-1]
        assert inpY_n.shape[-1] == self.Xrg + (pred_steps-1)  # Yseq cond_steps
        wnd_len = self.Xrg + pred_steps  # inpY.dim_time + 1
        num_series = inpY_n.shape[0]

        Y = torch.cat((inpY_n, tgtY_n[:, [-1]]), -1)  
        # get prediction
        Y_h1, X_h,Y_I,X_I = self(Y[..., :self.Xrg], wnd_start, pred_steps)

        
        X0 = self.encoder(Y[..., :self.Xrg].T)
        Y_h0 = self.decoder(X0).T
        Y_h = torch.cat((Y_h0, Y_h1), -1)
        X1 = self.encoder(Y[..., self.Xrg:].T)
        Y11=torch.mm(Y.T,Q.T)
        Y11=Y11.T
        X22=self.encoder1(Y11[..., :self.Xrg].T)
        Y22 = self.decoder1(X22).T
        Y_I = torch.cat((Y22, Y_I), -1)
        Y_I=torch.mm(Y_I.T,Q)
        Y_I=Y_I.T
        A=self.dynamics.dynamics.weight.to(device)  

        # for metrics        
        outY = inv_transform(scaler, Y_h1)
        assert outY.isnan().any() == False

        l1loss = nn.L1Loss()
        recov_loss = l1loss(Y_h, Y)
        koopman_loss=l1loss(Y_h,Y_I)  
        l2loss = nn.MSELoss()
        pre_loss = l2loss(X_h, X1)     
        identity_loss=(torch.norm(torch.mm(A,A.T)-IK))**2
        loss = recov_loss+(5e-2)*koopman_loss+(5e-1)*identity_loss+pre_loss
      
        assert loss < 1e3
        self.log("loss/val", loss,
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("loss/val_recov", recov_loss,
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("loss/val_pre", pre_loss,
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("loss/val_koopman", koopman_loss,
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("loss/val_identity", identity_loss,
                 on_step=False, on_epoch=True, batch_size=1)              
        """@nni.report_intermediate_result(...)"""

        return loss, outY,tgtY

    def validation_epoch_end(self, outputs) -> None:
        outYs = []
        tgtYs = []
        for loss, outY,tgtY in outputs:
            outYs.append(outY)
            tgtYs.append(tgtY)
        outY = torch.cat(outYs, -1)
        tgtY = torch.cat(tgtYs, -1)
        wape, mape, smape = metrics3(outY, tgtY)
        self.log("metrics/wape", wape, batch_size=1)
        self.log("metrics/mape", mape, batch_size=1)
        self.log("metrics/smape", smape, batch_size=1)
      

    def configure_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            monitor='loss/val', mode='min', save_last=True)
        self.earlystop_callback = EarlyStopping(
            monitor='loss/val', patience=20, mode='min')
        self.loglr_cb = LearningRateMonitor("epoch")

        return [self.loglr_cb, self.earlystop_callback, self.checkpoint_callback]  #

    def on_train_epoch_end(self):
        if self.trainer.should_stop:
            hp_metric = self.checkpoint_callback.best_model_score
            self.log('hp_metric', hp_metric)
            final_result = hp_metric.item()
            """@nni.report_final_result(final_result)"""


def main_COLLAR(*args):
    rg = 24*7
    dm = TSDataModule("electricity", rg, 24, val_wnd=7, hbsize=2*rg, hbstep=24,
                      mode="M", bs=1024)
    
    enc_channels = [96, 64, 32]
    rnn_hidden = 64 * 2
    rnn_layers = 4
    dropout_X = 0
    reg = 0.5
    lr = 0.001
    model = COLLAR(num_series=370,
                 enc_channels=enc_channels,
                 rnn_hidden=rnn_hidden,
                 rnn_layers=rnn_layers,
                 dropout_X=dropout_X,
                 rg=rg,
                 reg=reg,
                 lr=lr,
                 )
    trainer = pl.Trainer(gpus=1, log_every_n_steps=37,  # fast_dev_run=5,
                         max_epochs=750, min_epochs=300)  #
    trainer.fit(model, datamodule=dm)


warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*")

if __name__ == "__main__":
    pl.seed_everything(2021)
    main_COLLAR()