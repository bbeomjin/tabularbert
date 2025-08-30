import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List
from torch.utils.data import Dataset
from .type import ArrayLike


def get_n_categories(x: ArrayLike) -> Dict[str, int]:
    if isinstance(x, pd.DataFrame):
        columns = list(x.columns)
        n_categories = list(x.apply(lambda x: len(x.unique())))
        return {k: v for k, v in zip(columns, n_categories)}
    else:
        columns = [j for j in range(x.shape[1])]
        n_categories = [len(np.unique(x[:, j])) for j in range(x.shape[1])]
        return {k: v for k, v in zip(columns, n_categories)}



def divide_numerical_and_categorical(
            x: ArrayLike,
            encoding_info: Dict[str, int|str]
            ) -> Dict[str, Dict[str, ArrayLike]]:
    
    if isinstance(x, pd.DataFrame):
        columns = list(x.columns)
    else:
        columns = list(range(x.shape[1]))
    
    num_encoding_info = {k: v for k, v in encoding_info.items() if v != 'category'}
    cat_encoding_info = {k: v for k, v in encoding_info.items() if v == 'category'}
        
    cat_vars = list(cat_encoding_info.keys())
    num_vars = [k for k in columns if k not in cat_vars]
    
    if isinstance(x, pd.DataFrame):
        x_cat = x[cat_vars].copy()
        x_num = x[num_vars].copy()
    else:
        x_num = x[:, num_vars]
        x_cat = x[:, cat_vars]
        
        if np.issubdtype(x_num.dtype, np.number) is not True:
            x_num = x_num.astype(np.float64)
    
    return {"numerical": {"x": x_num,
                         "encoding_info": num_encoding_info}, 
            "categorical": {"x": x_cat,
                            "encoding_info": cat_encoding_info}}    



class DiscretizeBase:
    """
    Base class for the discretization classes, ['QunatileDiscretize', 'UniformDiscretize'].
    """
    
    def __init__(self, 
                 num_bins: int = 10,
                 encoding_info: Dict[str, Dict[str, int]] = None
                 ) -> None:
       
       # Default number of bins and subbins 
       self.num_bins = num_bins
       
       # Specific binning information
       self.encoding_info = encoding_info
       
    def _fit(self,
             x: ArrayLike,
             num_bins: int,
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
        
        # Set the default number of bins
        encoding_info = {k: self.num_bins for k in self.columns}
        
        if self.encoding_info is not None:
            vars = list(self.encoding_info.keys())
            if all([k in encoding_info.keys() for k in vars]) is not True:
                raise ValueError(
                    f"Column(s) specified in 'encoding_info' are not found in the input data: {vars}"
                )
            for v in vars:
                encoding_info[v] = self.encoding_info[v]
                
        self.encoding_info = encoding_info
        
        # Getting cut-off values for binning
        bins = {}
        for j, (k, v) in enumerate(encoding_info.items()):
            if isinstance(v, int) or isinstance(v, float):
                xx = x[:, j]
                if np.issubdtype(xx.dtype, np.number) is not True:
                    xx = xx.astype(np.float64)
                bins[k] = self._fit(xx, num_bins=v)
            else:
                bins[k] = None        
        
        self.bins = bins
        
    def _discretize(self, 
                    x: ArrayLike,
                    bins: List[float] | ArrayLike,
                    ) -> ArrayLike:

        ids = np.digitize(x, bins = bins, right = False)
        # Bin index starts with 1
        return ids.astype(int) + 1
    
    def _cat2int(self,
                 x: ArrayLike
                 ) -> ArrayLike:
        codes = {k: v + 1 for v, k in enumerate(set(x))}
        return np.array([codes[k] for k in x])
    
    def discretize(self, 
                   x: ArrayLike):
        
        if isinstance(x, pd.DataFrame):
            x = x.values
        
        if len(self.encoding_info) != x.shape[1]:
            raise ValueError(
                "The number of columns in the data to be discretized does not match the number of columns in the fitted data."
            )
        
        bin_ids_list = list()
        for j, (k, v) in enumerate(self.encoding_info.items()):
            if isinstance(v, int) or isinstance(v, float):
                xx = x[:, j]
                if np.issubdtype(xx.dtype, np.number) is not True:
                    xx = xx.astype(np.float64)
                bin_ids = self._discretize(x=xx, 
                                           bins=self.bins[k])
            else:
                bin_ids = self._cat2int(x=x[:, j])
                
            bin_ids_list.append(bin_ids)

        return np.stack(bin_ids_list, axis = 1)



class QuantileDiscretize(DiscretizeBase):
    def __init__(self, 
                 num_bins: int = 10,
                 encoding_info: Dict[str, Dict[str, int]] = None
                 ) -> None:
        
        super(QuantileDiscretize, self).__init__(
            num_bins = num_bins,
            encoding_info = encoding_info
        )
        
    def _fit(self, 
             x: ArrayLike,
             num_bins: int,
             ) -> ArrayLike:
        bins = np.quantile(x, np.linspace(0, 1, num_bins + 1))
        bins[-1] = np.inf
        return bins[1:]



class UniformDiscretize(DiscretizeBase):
    def __init__(self, 
                 num_bins: int = 10,
                 encoding_info: Dict[str, Dict[str, int]] = None
                 ) -> None:
        
        super(UniformDiscretize, self).__init__(
            num_bins = num_bins,
            encoding_info = encoding_info
        )
        
    def _fit(self, 
             x: ArrayLike,
             num_bins: int,
             ) -> ArrayLike:
        bins = np.linspace(np.min(x), np.max(x), num_bins + 1)
        bins[-1] = np.inf
        return bins[1:]    

    
    
class SSLDataset(Dataset):
    """
    Dataset class for TabularBERT masked language modeling pretraining.
    
    This dataset handles the masking strategy for self-supervised learning on
    tabular data that has been discretized into bins. It applies three types of
    token transformations: masking, random replacement, and keeping unchanged.
    
    Args:
        x (ArrayLike): Original tabular data
        bin_ids (ArrayLike): Discretized tabular data as bin indices
        encoding_info (Dict[str, int]): Mapping of feature names to number of bins
        mask_token_id (int): Token ID used for masking. Default: 0
        mask_token_prob (float): Probability of masking tokens. Default: 0.15
        random_token_prob (float): Probability of random token replacement. Default: 0.1
        unchanged_token_prob (float): Probability of keeping tokens unchanged. Default: 0.1
        ignore_index (int): Index to ignore in loss calculation. Default: -100
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (masked_tokens, labels)
            - masked_tokens: Input tokens with masking applied
            - labels: Original tokens for loss calculation
    """
    
    def __init__(self,
                 x: ArrayLike,
                 bin_ids: ArrayLike,
                 encoding_info: Dict[str, int],
                 mask_token_id: int=0,
                 mask_token_prob: float=0.15,
                 random_token_prob: float=0.1,
                 unchanged_token_prob: float=0.1,
                 ignore_index: int=-100
                 ) -> None:
        
        super(SSLDataset, self).__init__()
        
        # Convert pandas DataFrame to numpy if needed
        if isinstance(x, pd.DataFrame):
            x = x.values
        
        if isinstance(bin_ids, pd.DataFrame):
            bin_ids = bin_ids.values
            
        # Store data and parameters
        self.x = x
        self.bin_ids = bin_ids
        self.encoding_info = encoding_info
        self.mask_token_id = mask_token_id
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.unchanged_token_prob = unchanged_token_prob
        self.ignore_index = ignore_index
        
        # Create the number of bins tensor for each feature
        self.num_bins = torch.tensor([
            self.encoding_info[k] for k in self.encoding_info.keys()
        ])
        
    def _apply_masking(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking strategy to input tokens.
        
        Args:
            tokens (torch.Tensor): Original token sequence
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (masked_tokens, labels)
        """
        # Clone tokens for labels and masking
        labels = (tokens - 1).clone()
        masked_tokens = tokens.clone()
        
        # Generate random probabilities for each token
        probs = torch.rand(tokens.shape)
        
        # Determine which tokens to process (mask_token_prob of all tokens)
        mask_candidates = probs < self.mask_token_prob
        
        # Within mask candidates, determine the specific action:
        # - random_token_prob: replace with random token
        # - unchanged_token_prob: keep original token
        # - remaining: replace with [MASK] token
        
        random_mask = probs < (self.mask_token_prob * self.random_token_prob)
        unchanged_mask = (probs > (self.mask_token_prob - self.mask_token_prob * self.unchanged_token_prob)) & mask_candidates
        mask = mask_candidates & ~(random_mask | unchanged_mask)
        
        # Apply random token replacement
        masked_tokens[random_mask] = (torch.rand(len(tokens)) * self.num_bins + 1).type(masked_tokens.dtype)[random_mask]
        
        # Apply mask token
        masked_tokens[mask] = self.mask_token_id
        
        # Set labels for non-masked tokens to ignore_index
        labels[~mask_candidates] = self.ignore_index
        
        return masked_tokens, labels
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with masking applied.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (masked_tokens, labels)
        """
        # Get tokens for this sample
        y = torch.tensor(self.x[idx], dtype=torch.float)
        tokens = torch.tensor(self.bin_ids[idx], dtype=torch.long)
        
        # Apply masking strategy
        masked_tokens, labels = self._apply_masking(tokens)
        y[labels == self.ignore_index] = torch.nan
        
        return masked_tokens, labels, y
    
    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.bin_ids)



class FinetuneDataset(Dataset):
    def __init__(self, 
                 bin_ids: ArrayLike,
                 y: ArrayLike
                 ) -> None:
        super(FinetuneDataset, self).__init__()
        
        self.bin_ids = bin_ids
        self.y = y
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bin_idx = torch.tensor(self.bin_ids[idx], dtype=torch.long)
        dtype = torch.long if np.issubdtype(self.y[idx].dtype, np.integer) else torch.float
        label = torch.tensor(self.y[idx], dtype=dtype)
        return bin_idx, label
    
    def __len__(self) -> int:
        return len(self.bin_ids)



if __name__ == '__main__':
    x = np.random.rand(20, 10)
    discretizer = QuantileDiscretize(num_bins=100, encoding_info={0: 10})
    discretizer.fit(x)
    bin_ids = discretizer.discretize(x)
    dataset = SSLDataset(x=x,
                     bin_ids=bin_ids,
                     encoding_info=discretizer.encoding_info,
                     mask_token_prob=1.0,
                     random_token_prob=0.3,
                     unchanged_token_prob=0.3,
                     ignore_index=-100)
    print(dataset[0])