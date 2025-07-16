import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List
from utils.type import ArrayLike
from torch.utils.data import Dataset


class DiscretizeBase:
    """
    Base class for the discretization classes, ['QunatileDiscretize'].
    """
    
    def __init__(self, 
                 K: int = 10,
                 L: int = 10,
                 encoding_info: Dict[str, Dict[str, int]] = None
                 ) -> None:
       
       # Default number of bins and subbins 
       self.K = K
       self.L = L
       
       # Specific binning information
       self.encoding_info = encoding_info
       
    def _fit(self,
             x: ArrayLike,
             K: int,
             L: int
             ) -> List[float]:
        raise NotImplementedError()
    
    def fit(self,
            x: ArrayLike
            ) -> None:
        
        if isinstance(x, pd.DataFrame):
            self.columns = list(x.columns)
            x = x.values
        else:
            self.columns = [j for j in range(x.shape[1])]
        
        # Set the default number of bins and subbins
        encoding_info = {k: {'K': self.K, 'L': self.L} for k in self.columns}
        
        if self.encoding_info is not None:
            vars = list(self.encoding_info.keys())
            if all([k in encoding_info.keys() for k in vars]) is not True:
                raise ValueError(
                    f"Column(s) specified in 'encoding_info' are not found in the input data: {vars}"
                )
            for v in vars:
                encoding_info[v] = self.encoding_info[v]
        
        # Getting cut-off values for binning
        bins = {}
        for j, k in enumerate(encoding_info.keys()):
            bins[k] = self._fit(x[:, j], K = encoding_info[k]['K'], L = encoding_info[k]['L'])
        self.encoding_info = encoding_info
        self.bins = bins
        
    def _discretize(self, 
                    x: ArrayLike,
                    bins: List[float]|ArrayLike,
                    K: int,
                    L: int,
                    ) -> Tuple[ArrayLike, ArrayLike]:

        ids = np.digitize(x, bins = bins, right = False)
        bin_ids = np.floor(ids / L) 
        subbin_ids = ids - L * bin_ids
        if bin_ids.max() >= K:
            raise ValueError(
                    "Invalid bin cut-offs: The maximum cut-off value in bins must exceed the maximum value in x."
                )
        return bin_ids.astype(int) + 1, subbin_ids.astype(int) + 1
       
    def discretize(self, 
                   x: ArrayLike):
        
        if isinstance(x, pd.DataFrame):
            x = x.values
        
        
        if len(self.encoding_info) != x.shape[1]:
            raise ValueError(
                "The number of columns in the data to be discretized does not match the number of columns in the fitted data."
            )
        
        bin_ids_list = list(); subbin_ids_list = list()
        for j, k in enumerate(self.encoding_info.keys()):
            bin_ids, subbin_ids = self._discretize(x = x[:, j], 
                                                   bins = self.bins[k], 
                                                   K = self.encoding_info[k]['K'],
                                                   L = self.encoding_info[k]['L'])
            bin_ids_list.append(bin_ids)
            subbin_ids_list.append(subbin_ids)

        # Bin index starts with 1
        return {'bin_ids': np.stack(bin_ids_list, axis = 1),
                'subbin_ids': np.stack(subbin_ids_list, axis = 1)}



class QuantileDiscretize(DiscretizeBase):
    def __init__(self, 
                 K: int = 10,
                 L: int = 10,
                 encoding_info: Dict[str, Dict[str, int]] = None
                 ) -> None:
        
        super(QuantileDiscretize, self).__init__(
            K = K,
            L = L,
            encoding_info = encoding_info
        )
        
    def _fit(self, 
             x: ArrayLike,
             K: int,
             L: int,
             ) -> ArrayLike:
        bins = np.quantile(x, np.linspace(0, 1, K * L + 1))
        bins[-1] = np.Inf
        return bins[1:]
    
    
class GMTMLMData(Dataset):
    def __init__(self,
                 bin_ids: ArrayLike,
                 subbin_ids: ArrayLike,
                 encoding_info: Dict[str, int],
                 mask_token_id: int = 0,
                 mask_token_prob: float = 0.2,
                 ignore_index: int = -100,
                 random_token_prob: float = 0.1,
                 unchange_token_prob: float = 0.1,
                 device: torch.device = torch.device('cpu')
                 ):
        
        super(GMTMLMData, self).__init__()
        
        if isinstance(bin_ids, pd.DataFrame):
            bin_ids = bin_ids.values
            
        if isinstance(subbin_ids, pd.DataFrame):
            subbin_ids = subbin_ids.values
            
        self.bin_ids = bin_ids
        self.subbin_ids = subbin_ids
        self.encoding_info = encoding_info
        self.mask_token_id = mask_token_id
        self.mask_token_prob = mask_token_prob
        self.ignore_index = ignore_index
        self.random_token_prob = random_token_prob
        self.unchange_token_prob = unchange_token_prob
        self.device = device
        self.L = torch.tensor([self.encoding_info[k]['L'] for k in self.encoding_info.keys()])
        self.K = torch.tensor([self.encoding_info[k]['K'] for k in self.encoding_info.keys()])
        
            
    def random_word(self, bin_ids, probs, max_idx):
        mask_ids = probs < self.mask_token_prob
        labels = bin_ids.clone()
        random_ids = probs < (self.mask_token_prob * self.random_token_prob)
        unchange_ids = probs > (self.mask_token_prob - self.mask_token_prob * self.unchange_token_prob)
        unchange_ids = unchange_ids * mask_ids
        mask_ids = ~(random_ids | unchange_ids) * mask_ids
        bin_ids[random_ids] = (torch.rand(len(bin_ids)) * max_idx).floor().long()[random_ids] + 1
        bin_ids.masked_fill_(mask_ids, self.mask_token_id)
        return bin_ids, labels
        
        
    def __getitem__(self, idx):
        bin_ids = self.bin_ids[idx]
        subbin_ids = self.subbin_ids[idx]
        
        bin_ids = torch.tensor(bin_ids)
        subbin_ids = torch.tensor(subbin_ids)
        
        probs1 = torch.rand(len(bin_ids))
        probs2 = torch.rand(len(subbin_ids))
        bin_ids, bin_labels = self.random_word(bin_ids, probs1, self.K)
        subbin_ids, subbin_labels = self.random_word(subbin_ids, probs2, self.L)
        
        labels = self.L * (bin_labels - 1) + subbin_labels - 1
        labels[~((probs1 < self.mask_token_prob) | (probs2 < self.mask_token_prob))] = self.ignore_index
        
        return bin_ids.to(self.device), subbin_ids.to(self.device), labels.to(self.device)
    
    def __len__(self):
        return len(self.bin_ids)


