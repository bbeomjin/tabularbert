# AdamW with proximal operator
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _get_capturable_supported_devices,
    _get_value,
    _to_scalar,
    DeviceDtypeDict,
)

class PAdamW(Adam):
    def __init__(self,
                 params,
                 proximals,
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
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay, 
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=True,
        )
        
        if proximals is None:
            proximals = [None] * len(self.param_groups)
        
        if not isinstance(proximals, list):
            proximals = list(proximals)
        
        if len(proximals) != len(self.param_groups):
            raise ValueError("Invalid length of argument proximals: {} instead of {}".format(len(proximals), len(self.param_groups)))
        
        for group, proximal in zip(self.param_groups, proximals):
            group.setdefault('proximal', proximal)
        
    def __setstate__(self, state):
        super(PAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group["decoupled_weight_decay"] = True
    
    @_use_grad_for_differentiable    
    def step(self, closure = None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            proximal = group['proximal']
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group = group,
                params_with_grad = params_with_grad,
                grads = grads,
                exp_avgs = exp_avgs,
                exp_avg_sqs = exp_avg_sqs,
                max_exp_avg_sqs = max_exp_avg_sqs,
                state_steps = state_steps,
            )

            padamw(
                proximal,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )

        return loss    



# padamw is just like adamw but with proximal operator
# @_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_padamw)
def padamw(
    proximal,
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    decoupled_weight_decay: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    # if fused and not torch.jit.is_scripting():
    #     func = _fused_adam
    # elif foreach and not torch.jit.is_scripting():
    #     func = _multi_tensor_adam
    # else:
    func = _single_tensor_padamw

    func(
        proximal,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        decoupled_weight_decay=decoupled_weight_decay,
    )
    
def _single_tensor_padamw(
    proximal, 
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)
    else:
        lr = _to_scalar(lr)
        # TODO: Support nonzero-dim Tensor betas, see #147921

    # We only shuffle around the beta when it is a Tensor, otherwise, we prefer
    # treating it as a scalar.
    # Note: ensure type declaration is under conditional check for isinstance
    # or else torchscript will get cranky about the DeviceDict type.
    if isinstance(beta1, Tensor):
        beta1_dict: Optional[DeviceDtypeDict] = {(beta1.device, beta1.dtype): beta1}
    else:
        beta1_dict = None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), (
                f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
            )

        # update step
        step_t += 1

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform stepweight decay
                param.mul_(1 - lr * weight_decay)
            else:
                # Nested if is necessary to bypass jitscript rules
                if differentiable and isinstance(weight_decay, Tensor):
                    if weight_decay.requires_grad:
                        grad = grad.addcmul_(param.clone(), weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        device = param.device

        if beta1_dict is not None:
            dtype = param.dtype  # type: ignore[union-attr]

            # cast to workaround https://github.com/pytorch/pytorch/issues/140601
            key = (device, dtype)
            if key not in beta1_dict:
                beta1_dict[key] = beta1.to(  # type: ignore[union-attr]
                    device=device, dtype=dtype, non_blocking=True
                )

            device_beta1: Union[float, Tensor] = beta1_dict[key]
        else:
            device_beta1 = beta1

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - device_beta1)

        # Nested if is necessary to bypass jitscript rules
        if differentiable and isinstance(beta2, Tensor):
            if beta2.requires_grad:
                # Using lerp to only use 2 operations bc addcmul's value cannot be a tensor
                # Showing equivalence of differentiable path and nondifferentiable path
                # expavg * b2 + grad^2 * (1-b2)
                #           add expavg * (1-b2) - expavg * (1-b2) = 0
                # expavg * b2 + expavg * (1-b2) - expavg * (1-b2) + grad^2 * (1-b2)
                # expavg - expavg * (1-b2) + grad^2 * (1-b2)
                # expavg + (grad^2 - expavg) * (1-b2)
                # expavg.lerp(grad^2, 1-beta2)
                exp_avg_sq.lerp_(torch.square(grad), weight=1 - beta2)
            else:
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            # Nested if is necessary to bypass jitscript rules
            if differentiable and isinstance(beta1, Tensor):
                if beta1.requires_grad:
                    bias_correction1 = 1 - beta1 ** step.clone()
                else:
                    bias_correction1 = 1 - beta1**step
            else:
                bias_correction1 = 1 - beta1**step

            # Nested if is necessary to bypass jitscript rules
            if differentiable and isinstance(beta2, Tensor):
                if beta2.requires_grad:
                    bias_correction2 = 1 - beta2 ** step.clone()
                else:
                    bias_correction2 = 1 - beta2**step
            else:
                bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            if differentiable:
                param.addcdiv_(exp_avg.clone(), denom)
            else:
                param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = bias_correction2**0.5

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            if proximal is None:
                param.addcdiv_(exp_avg, denom, value=-step_size)
            else:
                param.data.copy_(proximal(param.data, exp_avg / denom, step_size))
            
        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])