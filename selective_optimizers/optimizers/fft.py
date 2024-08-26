import torch

def get_fft(base_optimizer):
    
    class Fft(base_optimizer):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        @torch.no_grad()
        def init_chosen(self):
            pass
        
        @torch.no_grad()
        def update_chosen(self):
            pass
        
        def mask_gradients(self):
            pass
        
        def post_train_work(self):
            pass
        
        def pre_train_work(self):
            pass
        
    return Fft  
        