import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from utils.type import ArrayLike


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.embedding import NumEmbedding
from model.BERT import BERT, GMTMaskedLanguageModel
from utils.data import GMTMLMData


from utils.data import GMTMLMData, QuantileDiscretize
from utils.criterion import GMTCrossEntropy, GMTMSE, GMTWAS
# from utils.regularizer import L2EmbedPenalty, L2EmbedPenaltyStop
from utils.metrics import MLMAccuracy
from utils.utils import CheckPoint, make_save_dir
from utils.scheduler import WarmupCosineLR
import tqdm

import os
import pathlib
import json
import warnings
import functools


class GMTMLModel(nn.Module):
    def __init__(self,
                 encoding_info: dict[str, dict[str, int]], 
                 max_len: int,
                 max_position: int,
                 embedding_dim: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 8, 
                 dropout: float = 0.1) -> None:
        super(GMTMLModel, self).__init__()
        
        self.encoding_info = encoding_info
        self.max_len = max_len
        self.max_position = max_position
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        
        
        # self.embedding = MulticolNumEmbedding(max_len = max_len,
        #                             max_position = max_position,
        #                             embedding_dim = embedding_dim,
        #                             mask_idx = 0)      
        
        self.embedding = NumEmbedding(max_len = max_len,
                                    max_position = max_position,
                                    embedding_dim = embedding_dim,
                                    mask_idx = 0)      
        
        self.bert = BERT(self.embedding,
                         n_layers = n_layers,
                         n_heads = n_heads,
                         dropout = dropout)
        
        self.model = GMTMaskedLanguageModel(self.bert, encoding_info)
        
    def forward(self, bin_ids, subbin_ids):
        out = self.model(bin_ids, subbin_ids)
        return out
    

class GMTModel(nn.Module):
    def __init__(
        self,
        x: ArrayLike,
        encoding_info: Dict[str, int]|Dict[str, Dict[str, int]] = None,
        device: torch.device = torch.device('cpu'),
        valid_x = None
    ) -> None:

        super(GMTModel, self).__init__()
        
        self.x = x
        self.encoding_info = encoding_info
        K = encoding_info['K']
        L = encoding_info['L']
        encoding_info.pop('K', None); encoding_info.pop('L', None)
        if len(encoding_info) == 0:
            encoding_info = None
        
        self.discretizer = QuantileDiscretize(K = K, L = L, encoding_info = encoding_info)
        self.discretizer.fit(x)
        self.bin_ids = self.discretizer.discretize(x)
        
        self.valid_x = valid_x
        if valid_x is not None:
            self.valid_bin_ids = self.discretizer.discretize(valid_x)
        
        self.device = device
        self.model = None
        self.optimizer = None

        # Get current file path
        # save_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.dirname(pathlib.Path(__file__).parent)
        
        # Set model save directory
        save_dir = make_save_dir(save_dir, 'pretraining')
        self.save_dir = save_dir
        
        # Set logger
        self.logger = SummaryWriter(os.path.join(save_dir, 'log'))
        
        # save_path_args = os.path.join(save_dir, 'hparams.json')
        # with open(save_path_args, 'wt') as f:
        #     json.dump(vars(args), f, indent = 4)

        
    def set_mlm(self, 
                embedding_dim: int = 256,
                n_layers: int = 2,
                n_heads: int = 4,
                dropout: float = 0.1) -> None:
        
        max_len = max([max(v['K'], v['L']) for v in self.discretizer.encoding_info.values()])
        max_position = len(self.discretizer.encoding_info)
        self.model = GMTMLModel(encoding_info = self.discretizer.encoding_info,
                                max_len = max_len,
                                max_position = max_position,
                                embedding_dim = embedding_dim,
                                n_layers = n_layers,
                                n_heads = n_heads,
                                dropout = dropout).to(self.device)
    
    def set_optimizer(self,
                      lr: float = 1e-3,
                      weight_decay: float = 0.01,
                      betas: Tuple[float, float] = (0.9, 0.999)
                      ) -> None:
        self.optimizer = functools.partial(torch.optim.AdamW, 
                                           lr = lr,
                                           weight_decay = weight_decay,
                                           betas = betas)    
    
    def train(self, 
              epochs: int = 1000,
              batch_size: int = 512,
              criterion: str = 'mse',
              lamb: float = 0.0,
              mask_token_prob: float = 0.1,
              random_token_prob: float = 0.1,
              unchange_token_prob: float = 0.1,
              ) -> None:
        
        train_dataset = GMTMLMData(self.bin_ids['bin_ids'], self.bin_ids['subbin_ids'], 
                                   self.discretizer.encoding_info, 
                                   mask_token_prob = mask_token_prob,
                                   random_token_prob = random_token_prob,
                                   unchange_token_prob = unchange_token_prob,
                                   device = self.device)
        trainloader = DataLoader(train_dataset, 
                                 batch_size = batch_size,
                                 shuffle = True, 
                                 drop_last = True)

        if self.valid_x is not None:
            valid_dataset = GMTMLMData(self.valid_bin_ids['bin_ids'], self.valid_bin_ids['subbin_ids'], 
                                       self.discretizer.encoding_info, 
                                       mask_token_prob = mask_token_prob,
                                       random_token_prob = random_token_prob,
                                       unchange_token_prob = unchange_token_prob,
                                       device = self.device)

            validloader = DataLoader(valid_dataset, 
                         batch_size = batch_size, 
                         shuffle = True,
                         drop_last = False)
            
        
        if self.model is None:
            warnings.warn(
                """
                Model has not been configured. Proceeding with default parameter settings.
                To set the model manually, use the 'set_mlm' method.
                Default model:
                embedding_dim = 256,
                n_layers = 2,
                n_heads = 4,
                dropout = 0.1
                """
            )
            self.set_mlm()
        
        
        CE = GMTCrossEntropy(self.discretizer.encoding_info, ignore_index = -100)
        ntl_criterion = {'mse': GMTMSE, 'was': GMTWAS}
        NTL = ntl_criterion[criterion](self.discretizer.encoding_info, ignore_index = -100)
        
        # Define metric
        accuracy = MLMAccuracy(ignore_index = -100)
        checkpoint = CheckPoint(self.save_dir, max = True)
        
        if self.optimizer is None:
            warnings.warn(
                """
                Optimizer has not been configured. Proceeding with default parameter settings.
                To set the optimizer manually, use the 'set_optimizer' method.
                Default optimizer:
                learning rate = 1e-3,
                weight_decay = 0.01,
                betas = (0.9, 0.999)
                """
            )
            self.set_optimizer()
        self.optimizer = self.optimizer(params = self.model.parameters())
        total_steps = epochs * len(trainloader)
        scheduler = WarmupCosineLR(self.optimizer, 
                                   warmup_epochs = total_steps * 0.1,
                                   max_epochs = total_steps, 
                                   warmup_start_lr = 1e-5)
        
        k = 0
        for epoch in tqdm.tqdm(range(epochs), desc = 'Pretraining...'):
            risk = 0.
            acc_cum = 0.
            valid_risk = 0.
            valid_acc_cum = 0.
            for i, (train_bin_ids, train_subbin_ids, train_labels) in enumerate(trainloader):
                self.model.train(); self.optimizer.zero_grad()
                preds = self.model(train_bin_ids, train_subbin_ids)
                
                ce = CE(preds, train_labels)
                ntl = NTL(preds, train_labels)
                
                loss = (1. - lamb) * ce + lamb * ntl
                
                acc = accuracy(preds, train_labels)
                acc_cum += acc
                
                loss.backward(); self.optimizer.step(); scheduler.step()
                risk += loss.item()
                
                self.logger.add_scalar('Loss/train', loss.item(), k)
                self.logger.add_scalar('Accuracy/train', acc, k)
                k += 1
            
            if self.valid_x is not None:    
                with torch.no_grad():
                    self.model.eval()
                    for j, (valid_bin_ids, valid_subbin_ids, valid_labels) in enumerate(validloader):
                        valid_preds = self.model(valid_bin_ids, valid_subbin_ids)
                        
                        valid_ce = CE(valid_preds, valid_labels)
                        valid_ntl = NTL(valid_preds, valid_labels)
                        valid_loss = (1 - lamb) * valid_ce + lamb * valid_ntl
                        
                        valid_risk += valid_loss.item()
                        valid_acc = accuracy(valid_preds, valid_labels)
                        valid_acc_cum += valid_acc
                        
                        self.logger.add_scalar('Loss/valid', valid_loss.item(), epoch)
                        self.logger.add_scalar('Accuracy/valid', valid_acc, epoch)
                    
                    checkpoint(valid_acc_cum / (j + 1), self.model, epoch)
                    print('EPOCH: {epoch}, Training Loss: {loss}, Accuracy: {acc}'.format(epoch = epoch, 
                                                                                    loss = risk / (i + 1), 
                                                                                    acc = acc_cum / (i + 1)))
                    print('Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}'.format(valid_loss = valid_risk / (j + 1),
                                                                                                valid_acc = valid_acc_cum / (j + 1)))
            else:
                checkpoint(acc_cum / (i + 1), self.model, epoch)
                print('EPOCH: {epoch}, Training Loss: {loss}, Accuracy: {acc}'.format(epoch = epoch, 
                                                                                    loss = risk / (i + 1), 
                                                                                    acc = acc_cum / (i + 1)))    
                    
                    
