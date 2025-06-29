from .optimizers.base_optimizer import get_base_optimizer
from .optimizers.bitfit import get_bitfit
from .optimizers.id3 import get_id3
from .optimizers.pafi import get_pafi

def get_selective_optimizer(optimizer_class, peft_to_use):
    """
    Get the selective optimizer class based on the PEFT method.

    Args:
        optimizer_class (torch.optim.Optimizer): The base optimizer class to wrap.
        peft_to_use (str): The PEFT method to use ('id3', 'bitfit', or 'pafi').

    Returns:
        BaseOptimizer: The selective optimizer class.
    """
    base_optimizer = get_base_optimizer(optimizer_class)
    if peft_to_use == "bitfit":
        opt = get_bitfit(base_optimizer)
    elif peft_to_use == "id3":
        opt = get_id3(base_optimizer)
    elif peft_to_use == "pafi":
        opt = get_pafi(base_optimizer)
    else:
        raise ValueError(f"Unsupported PEFT method: {peft_to_use}")
    return opt
