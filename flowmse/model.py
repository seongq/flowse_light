import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from flowmse import sampling
from flowmse.odes import ODERegistry
from flowmse.backbones import BackboneRegistry
from flowmse.util.inference import evaluate_model
from flowmse.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt
# from flowmse.odes import OTFLOW
import random
from torch_pesq import PesqLoss



class VFModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (0 by default)")
        parser.add_argument("--T_rev",type=float, default=1.0, help="The maximum time")
        parser.add_argument("--sr", type=int, default=16000, help="The sample rate of the audio files.")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        parser.add_argument("--mode_condition", type=str, required=True, choices=("orifirstflow_orisecondflow_orafirstflow_orasecondflow_kdfirstCFM_kdsecondCFM_nogradkd_CTFSE_noisy_mean","orifirstflow_orisecondflow_orafirstflow_orasecondflow_kdfirstCFM_kdsecondCFM_nogradkd_CTFSE_MSE_noisy_mean", "pesq_ori_ora_kd_noisy_mean_local_align_no_grad", "pesq_ora_kd_noisy_mean_no_grad","pesq_ori_ora_kd_noisy_mean_no_grad","flowse_noisy_mean", "ori_kd_noisy_mean_no_grad", "flowse","ora_kd_zero_mean_no_grad","ora_kd_noisy_mean_no_grad", "ori_ora_kd_noisy_mean_no_grad", "ori_ora_kd_zero_mean","ori_ora_kd_zero_mean_no_grad","ori_ora_kd_noisy_mean","ori_ora_kd","ori_ora","ori_kd","ora_kd","ori","ori_ora_kd_nograd","ori_kd_nograd","ora_kd_nograd"))
        return parser

    def __init__(
        self, backbone, ode, lr=1e-4, ema_decay=0.999, t_eps=0.03, T_rev = 1.0,  loss_abs_exponent=0.5, 
        num_eval_files=10, loss_type='mse', data_module_cls=None, sr=16000, N_enh=10, mode_condition="ori_ora_kd_zero_mean", **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)        
        ode_cls = ODERegistry.get_by_name(ode)
        self.mode_condition = mode_condition
        
        self.ode = ode_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.ode.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        # self.mode = mode
        self.sr = sr
        # Initialize PESQ loss if pesq_weight > 0.0
        if "pesq" in self.mode_condition:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        else:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        if "pesq" in self.mode_condition:
            self.ema.update(self.dnn.parameters())
        else:
            self.ema.update(self.parameters())
    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                if "pesq" in self.mode_condition:
                    self.ema.store(self.dnn.parameters())        # store current params in EMA
                    self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
                else:
                    self.ema.store(self.parameters())        # store current params in EMA
                    self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if "pesq" in self.mode_condition:
                    if self.ema.collected_params is not None:
                        self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
                else:
                    if self.ema.collected_params is not None:
                        self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _mse_loss(self, x, x_hat):    
        err = x-x_hat
        losses = torch.square(err.abs())

        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    
    def _loss(self, vectorfield, condVF):    
        if self.loss_type == 'mse':
            err = vectorfield-condVF
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            err = vectorfield-condVF
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        import random
        x0, y = batch
        rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps        
        t = torch.min(rdm, torch.tensor(self.T_rev))
        # mean, std = self.ode.marginal_prob(x0, t, y)
        # z = torch.randn_like(x0)  #
        # sigmas = std[:, None, None, None]
        # xt = mean + sigmas * z
        # der_std = self.ode.der_std(t)
        # der_mean = self.ode.der_mean(x0,t,y)
        # condVF = der_std * z + der_mean   #target
        # VECTORFIELD_origin = self(xt,t,y,y)
        # loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
    # "ori_ora_kd","ori_ora","ori_kd","ora_kd","ori","ori_ora_kd_nograd","ori_kd_nograd","ora_kd_nograd"
        if self.mode_condition == "ori_ora_kd_zero_mean":
            mean, std = self.ode.marginal_prob(x0, t, torch.zeros_like(y))
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,torch.zeros_like(y))
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN)
            loss = loss_original_flow+loss_oracle_flow+loss_kd
            
        elif self.mode_condition == "ori_ora_kd_zero_mean_no_grad":
            mean, std = self.ode.marginal_prob(x0, t, torch.zeros_like(y))
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,torch.zeros_like(y))
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach()
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd)
            loss = loss_original_flow+loss_oracle_flow+loss_kd
            
            
        elif self.mode_condition == "ora_kd_zero_mean_no_grad":
            mean, std = self.ode.marginal_prob(x0, t, torch.zeros_like(y))
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,torch.zeros_like(y))
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach()
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd)
            
            
            loss = loss_oracle_flow+loss_kd
        elif self.mode_condition == "ora_kd_noisy_mean_no_grad":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach()
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd)
            loss = loss_oracle_flow+loss_kd
            
        elif self.mode_condition == "ori_ora_kd_noisy_mean_no_grad":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach()
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd)
            loss = loss_original_flow+loss_oracle_flow+loss_kd
            
        elif self.mode_condition == "ori_kd_noisy_mean_no_grad":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach()
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd)
            loss = loss_original_flow+loss_kd
            
        elif self.mode_condition == "flowse_noisy_mean":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)            
            loss = loss_original_flow
            
        elif self.mode_condition == "ori_ora_kd_noisy_mean":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN)
            loss = loss_original_flow+loss_oracle_flow+loss_kd
        elif self.mode_condition == "pesq_ori_ora_kd_noisy_mean_no_grad":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach()
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd)
            x_hat_original = xt -t[:, None, None, None]*VECTORFIELD_origin
            x_hat_oracle = xt -t[:, None, None, None]*VECTORFIELD_CLEAN
            # print(x_hat_original.size())
            
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_original_td = self.to_audio(x_hat_original.squeeze(), target_len)
            x_hat_oracle_td = self.to_audio(x_hat_oracle.squeeze(), target_len)
            x_td = self.to_audio(x0.squeeze(), target_len)
            
            
            loss_pesq_oracle= self.pesq_loss(x_td, x_hat_oracle_td)
            loss_pesq_oracle = torch.mean(loss_pesq_oracle)
            loss_pesq_original= self.pesq_loss(x_td, x_hat_original_td)
            loss_pesq_original = torch.mean(loss_pesq_original)
            
            
            
            loss = loss_original_flow+loss_oracle_flow+loss_kd + loss_pesq_oracle+loss_pesq_original
        elif self.mode_condition == "pesq_ora_kd_noisy_mean_no_grad":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach()
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd)
            x_hat_oracle = xt -t[:, None, None, None]*VECTORFIELD_CLEAN
            # print(x_hat_original.size())
            
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_oracle_td = self.to_audio(x_hat_oracle.squeeze(), target_len)
            x_td = self.to_audio(x0.squeeze(), target_len)
            
            
            loss_pesq_oracle= self.pesq_loss(x_td, x_hat_oracle_td)
            loss_pesq_oracle = torch.mean(loss_pesq_oracle)
           
            
            
            
            loss = loss_oracle_flow+loss_kd + loss_pesq_oracle
            
            
        elif self.mode_condition == "pesq_ori_ora_kd_noisy_mean_local_align_no_grad":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF)
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach()
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd)
            x_hat_original = xt -t[:, None, None, None]*VECTORFIELD_origin
            x_hat_oracle = xt -t[:, None, None, None]*VECTORFIELD_CLEAN
            # print(x_hat_original.size())
            
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_original_td = self.to_audio(x_hat_original.squeeze(), target_len)
            x_hat_oracle_td = self.to_audio(x_hat_oracle.squeeze(), target_len)
            x_td = self.to_audio(x0.squeeze(), target_len)
            
            
            t_plus_delta = torch.rand_like(t) * (self.T_rev - t) + t  # t와 T_rev 사이에서 다시 샘플링
            delta_t = t_plus_delta - t
            # print(t)
            # print(delta_t)
            vectorfield_oracle_local = self(xt+delta_t[:,None,None,None] * VECTORFIELD_CLEAN,t_plus_delta, y, x0)
            vectorfield_orginal_local = self(xt+delta_t[:,None,None,None] * VECTORFIELD_origin,t_plus_delta, y, y)
            
            loss_pesq_oracle= self.pesq_loss(x_td, x_hat_oracle_td)
            loss_pesq_oracle = torch.mean(loss_pesq_oracle)
            loss_pesq_original= self.pesq_loss(x_td, x_hat_original_td)
            loss_pesq_original = torch.mean(loss_pesq_original)
            loss_localalign_oracle = self._loss(VECTORFIELD_CLEAN,vectorfield_oracle_local)
            loss_localalign_origin = self._loss(VECTORFIELD_origin,vectorfield_orginal_local)
            
            
            
            loss = loss_original_flow+loss_oracle_flow+loss_kd + loss_pesq_oracle+loss_pesq_original+loss_localalign_oracle+loss_localalign_origin
        
        elif self.mode_condition == "orifirstflow_orisecondflow_orafirstflow_orasecondflow_kdfirstCFM_kdsecondCFM_nogradkd_CTFSE_noisy_mean":
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF = der_std * z + der_mean   #target
            VECTORFIELD_CLEAN =  self(xt,t,y,x0)
            VECTORFIELD_origin = self(xt,t,y,y)
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF) #orifirstflow
            loss_oracle_flow = self._loss(VECTORFIELD_CLEAN,condVF) #orafirstflow
            VECTORFIELD_CLEAN_nograd = VECTORFIELD_CLEAN.detach() #nogradkd
            loss_kd = self._loss(VECTORFIELD_origin,VECTORFIELD_CLEAN_nograd) #kdfirstCFM
            
            
            
            x1, z = self.ode.prior_sampling(y.shape,y)
            D_theta = x1 - self(x1,torch.ones(y.shape[0], device=y.device),y,y)
            
            
            rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps        
            t = torch.min(rdm, torch.tensor(self.T_rev))
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,y)
            condVF_second = der_std * z + der_mean   #target
            
            CONDITION=(D_theta+y)/2
            VECTORFIELD_CLEAN_second =  self(xt,t,CONDITION,x0)
            VECTORFIELD_origin_second = self(xt,t,CONDITION,D_theta)
            loss_original_flow_second = self._loss(VECTORFIELD_origin_second,condVF_second) #orifirstflow
            loss_oracle_flow_second = self._loss(VECTORFIELD_CLEAN_second,condVF_second) #orafirstflow
            VECTORFIELD_CLEAN_second_nograd = VECTORFIELD_CLEAN_second.detach() #nogradkd
            loss_kd_second = self._loss(VECTORFIELD_origin_second,VECTORFIELD_CLEAN_second_nograd) #kdfirstCFM
            
            # MSE = self._loss(D_theta, x0)
            
            
            loss=loss_original_flow+loss_oracle_flow+loss_kd+loss_original_flow_second+loss_oracle_flow_second+loss_kd_second
           
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y, c):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y, c], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
    