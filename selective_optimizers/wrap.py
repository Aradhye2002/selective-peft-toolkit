from .optimizers.base_optimizer import get_base_optimizer
from .optimizers.bitfit import get_bitfit
from .optimizers.id3 import get_id3
from .optimizers.pafi import get_pafi
from .optimizers.fft import get_fft

def get_selective_optimizer(optimizer, peft_to_use):
    base_optimizer = get_base_optimizer(optimizer)
    if peft_to_use == "bitfit":
        opt = get_bitfit(base_optimizer)
    elif peft_to_use == "id3":
        opt = get_id3(base_optimizer)
    elif peft_to_use == "pafi":
        opt = get_pafi(base_optimizer)
    elif peft_to_use == "fft":
        opt = get_fft(base_optimizer)
    else:
        raise ValueError(f"Unsupported PEFT method: {peft_to_use}")
    return opt
