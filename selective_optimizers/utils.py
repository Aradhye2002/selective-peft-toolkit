import torch

def get_chosen(param, chosen_tensor):
    # param is a 1d or 2d tensor and chosen_list is a list of corresponding indices
    # returns a flattened tensor of chosen parameters from param according to the chosen_list
    if (chosen_tensor):
        ndim = param.ndim
        if ndim == 1:
            chosen = param[chosen_tensor[:, 0]]
        elif ndim == 2:
            chosen = param[chosen_tensor[:, 0], chosen_tensor[:, 1]]
        elif ndim == 3:
            chosen = param[chosen_tensor[:, 0], chosen_tensor[:, 1], chosen_tensor[:, 2]]
        elif ndim == 4:
            chosen = param[chosen_tensor[:, 0], chosen_tensor[:, 1], chosen_tensor[:, 2], chosen_tensor[:, 3]]
        else:
            raise NotImplementedError
    else:
        chosen = torch.tensor([])
    return chosen

def set_chosen(param, chosen_tensor, values):
    # param is a 1d or 2d tensor and chosen_list is a list of corresponding indices
    # values is the tensor of values to which the chosen indices in param is to be set
    if (len(chosen_tensor) > 0):
        ndim = param.ndim
        if ndim == 1:
            param[chosen_tensor[:, 0]] = values
        elif ndim == 2:
            param[chosen_tensor[:, 0], chosen_tensor[:, 1]] = values
        elif ndim == 3:
            param[chosen_tensor[:, 0], chosen_tensor[:, 1], chosen_tensor[:, 2]] = values
        elif ndim == 4:
            param[chosen_tensor[:, 0], chosen_tensor[:, 1], chosen_tensor[:, 2], chosen_tensor[:, 3]] = values
        else:
            raise NotImplementedError

def get_not_chosen(param, chosen_mask):
    not_chosen_mask = torch.logical_not(chosen_mask)
    chosen = param[not_chosen_mask]
    return chosen    