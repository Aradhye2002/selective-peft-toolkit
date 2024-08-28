# Selective PEFT Toolkit

## Overview

Welcome to the `selective-peft-toolkit`, the official implementation for the paper **"Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models"**. This toolkit provides a flexible framework for selectively fine-tuning large language models using different selective Parameter-Efficient Fine-Tuning (PEFT) methods.

The toolkit includes the following PEFT methods:
- **FFT** (Full Fine-Tuning)
- **ID3** (Our proposed method)
- **PaFI**
- **BitFit**

These methods are exposed through a package called `selective_optimizers`, which can be installed via pip:

```bash
pip install selective_optimizers
```
## Key Features

- **Selective Optimizers**: Wrappers around standard optimizers (subclasses of torch.optim.Optimizer) that selectively update a budgeted number of parameters in the model.
- **Heuristic-Based Selection**: The selective optimizers update parameters according to various heuristics and selection strategies.
- **Integration with Transformers**: Compatible with transformers.Trainer for easy integration into your existing pipelines.

- **Efficient Storage**: Stores modified weights in a summary object that occupies only O(B) space, where B is the budget.

## Installation 

To install the selective_optimizers package, simply run:
```bash
pip install selective-optimizers
```
## Usage

### Training Workflow

Here's a basic workflow for training with a selective optimizer:
```python
from selective_optimizers.wrap import get_selective_optimizer
from selective_optimizers.load_store import write_summary_to_disk
from torch.optim import AdamW

# Choose your base optimizer
opt = AdamW

# Specify the PEFT method to use (can be one of “fft”, “id3”, “bitfit”, or “pafi”)
peft_to_use = "id3"

# Get the selective optimizer class
optimizer_class = get_selective_optimizer(opt, peft_to_use)

# Initialize the optimizer with additional selective parameters
optimizer = optimizer_class(
    params=model.parameters(), 
    lr=0.0001, 
    budget=100000, 
    exp=0, 
    eps=1e-3, 
    max_steps=1000
)

# Usual training loop
...
...

# Optional post-training work for validation
optimizer.post_train_work()
print("Budget used:", optimizer.get_budget())

# Save the summary of modified weights
summary = optimizer.get_summary(model)
write_summary_to_disk("path/to/summary.pth", summary)
```

### Inference Workflow

```python
from selective_optimizers.load_store import load_summary_from_disk, load_weights_from_summary

# Load your model as usual
...
model = ...
...

# Load the summary from disk
summary = load_summary_from_disk("path/to/summary.pth")

# Apply the modified weights from the summary to the model
load_weights_from_summary(model, summary)

# Usual inference code

...
...
```

# Integration with Transformers

The transformers.Trainer class accepts external optimizers, making it easy to integrate selective optimizers into your workflow:

1. Create a selective optimizer as shown above.
2. Pass it to the Trainer class and call .train() on it.
3. Post training, fetch and store the summary as described above.
4. For inference, just load the summary and update the model as shown in the code.

## Contributing
We welcome contributions to the selective_optimizers package! If you'd like to add a new selective optimizer, follow these steps:

1. Create a new file inside the optimizers/ folder.
2. Subclass optimizers/base_optimizer in your new file.
3. Override init_chosen() to set the initial masks for the parameters.
4. Override update_chosen() to define how the masks evolve with each step.
5. Please open a pull request with your new optimizer, and we’ll be happy to review it!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Citation
If you use this toolkit in your research, please cite our paper:

```sql
@article{Agarwal2024_step_by_step,
  title={Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models},
  author={Aradhye Agarwal and Suhas Kamasetty Ramesh and Ayan Sengupta and Tanmoy Chakraborty}
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024},
}

``````
## Contact
For any questions or issues, feel free to open an issue on the GitHub repository or contact us directly.
