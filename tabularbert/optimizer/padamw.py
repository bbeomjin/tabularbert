# AdamW with proximal operator
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.optim import AdamW

class PAdamW(AdamW):
    def __init__(self,
                 params,
                 proximal,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2,
                 amsgrad: bool = False,
                 *,
                 maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: Optional[bool] = None,
                 ) -> None:
        super(PAdamW, self).__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay, 
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused
        )
        self.proximal = proximal
        
    def __setstate__(self, state):
        super(PAdamW, self).__setstate__(state)
        