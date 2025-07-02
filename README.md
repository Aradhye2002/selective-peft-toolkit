# Selective PEFT Toolkit

![Neon Bonsai](assets/neon-bonsai.jpg)

## Overview

Welcome to the `selective-peft-toolkit`, the official implementation for the paper **"Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models."** This toolkit provides a flexible framework for selectively fine-tuning large language models using different selective Parameter-Efficient Fine-Tuning (PEFT) methods.

In addition to NLP, these methods can also be applied to other domains like computer vision, as demonstrated in the examples.

The toolkit includes the following selective PEFT methods:

- **ID3** (Our proposed method)
- **PaFI** ([PaFI Paper](https://arxiv.org/abs/2305.16742))
- **BitFit** ([BitFit Paper](https://arxiv.org/abs/2106.10199))

These methods are exposed through a package called `selective_optimizers`, which can be installed via pip:

```bash
pip install selective-optimizers
```

**Note**: The package is named `selective_optimizers` in code but is installed via pip as `selective-optimizers`.

## Key Features

- **Selective Optimizers**: Wrappers around standard optimizers (subclasses of `torch.optim.Optimizer`) that selectively update a budgeted number of parameters in the model.
- **Heuristic-Based Selection**: The selective optimizers update parameters according to various heuristics and selection strategies.
- **Integration with Transformers**: Compatible with `transformers.Trainer` for easy integration into your existing pipelines.
- **Efficient Storage**: Stores modified weights in a summary object that occupies only O(B) space, where B is the budget.
- **Model-Agnostic**: Can be used in conjunction with reparameterization-based PEFT techniques that modify the model, since the selective PEFT techniques implemented here operate directly on the optimization process, and are therefore model-agnostic.
- **Multi-GPU Support**: Seamlessly handles models sharded across multiple devices.

## Parameters

### Common Parameters

All selective optimizers share some common parameters:

- **budget**: The number of parameters you want to update.
- **verify**: A boolean flag indicating whether you want to perform a verification that only a budgeted number of parameters are updated. This is useful when extending the current framework and checking whether a given added PEFT method is not exceeding the budget. Note that in addition to ensuring that the budget is not exceeded, the verification also checks if any non-chosen parameter (indicated by the chosen_masks) have been updated, which would indicate a buggy implementation.

### Selective Parameters for Each PEFT Method

Some PEFT methods require additional parameters. Here are the selective parameters for each method:

#### ID3

- **max_steps**: The total number of optimization steps to be performed (i.e., the number of times `optimizer.step()` is called). This is needed to inform the budget scheduler how many parameters to unmask at each optimization step. Since we operate directly on the optimization process (by wrapping the optimizer class), it is not possible to internally determine how many times `optimizer.step()` will be called.

- **exp** and **eps**: Hyperparameters for the $D^3$ metric (**H**), which is defined as follows:

  $$H(\theta^i) = \frac{|\nabla_{\theta^i}|}{(|\theta^i| + \epsilon)^{\text{exp}}}$$

  where:
  - $\theta^i$ is the parameter,
  - $|\nabla_{\theta^i}|$ is the magnitude of its gradient,
  - $\epsilon$ is a small constant to prevent division by zero,
  - **exp** controls the influence of the parameter magnitude.

## Installation 

To install the `selective_optimizers` package, simply run:

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

# Specify the PEFT method to use (can be one of "id3", "bitfit", or "pafi")
peft_to_use = "id3"

# Get the selective optimizer class
optimizer_class = get_selective_optimizer(opt, peft_to_use)

params = [
    {"params": list_of_params_1, "choose_all": True},
    {"params": list_of_params_2},
]

# 'choose_all': Select all parameters in this group (useful for randomly initialized heads like classification layers).
# If 'choose_all' is not specified or is set to False, selection follows the chosen PEFT method.

# Initialize the optimizer with additional selective parameters
optimizer = optimizer_class(
    params=params, 
    lr=0.0001, 
    budget=100000, 
    exp=0, 
    eps=1e-3, 
    max_steps=1000
)

# Usual training loop
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data)
        # Compute loss
        loss = criterion(outputs, targets)
        # Backward pass
        loss.backward()
        # Optimizer step - the key masking of gradients and updating of internal state happens here
        optimizer.step()

# Optional post-training work for validation
optimizer.post_train_work()
print("Budget used:", optimizer.get_budget_used())

# Save the summary of modified weights
summary = optimizer.get_summary(model)
write_summary_to_disk("path/to/summary.pt", summary)
```

### Inference Workflow

```python
from selective_optimizers.load_store import load_summary_from_disk, load_weights_from_summary

# Load your model as usual
model = ...

# Load the summary from disk
summary = load_summary_from_disk("path/to/summary.pt")

# Apply the modified weights from the summary to the model
load_weights_from_summary(model, summary)

# Usual inference code
outputs = model(input_data)
```

## Integration with Transformers

The `transformers.Trainer` class accepts external optimizers, making it easy to integrate selective optimizers into your workflow:

1. **Create a selective optimizer** as shown above.
2. **Pass it to the `Trainer` class** and call `.train()` on it.
3. **Post-training**, fetch and store the summary as described above.
4. **For inference**, just load the summary and update the model as shown in the inference code.

## Examples

The `examples/` directory contains scripts demonstrating the use of selective optimizers in different scenarios:

1. ``vit_no_trainer.py`` is a self-contained script for training and evaluating a pretrained vision transformer (ViT) on the CIFAR-100 dataset.
2. ``vit_trainer.py`` demonstrates the use of selective optimizers with `transformers.Trainer` for fine-tuning a pretrained ViT on CIFAR-100.
3. ``vit_lora_no_trainer.py`` is a self-contained script for lora-training and evaluating a pretrained ViT on the CIFAR-100 dataset.
4. ``vit_lora_trainer.py`` demonstrates the use of selective optimizers with `transformers.Trainer` for LoRA-fine-tuning a pretrained ViT on CIFAR-100.

**Notes**: 
- LoRA-fine-tuning means wrapping a pretrained model with a LoraModel (which injects lora layers) and performing selective optimization on these lora layers only.

- In the `vit_lora_{trainer, no_trainer}.py` files we have to explicitly set the classifier head to trainable post-creation of the PeftModel (created using `get_peft_model()`). This is because the `get_peft_model()` method automatically sets non-lora layers as not trainable which is not desirable for the classifier since it is initialized from scratch.

- For parameters initialized from scratch—such as the ViT classifier head in examples 1-4—you would almost always want the full parameter to be trainable, since it has been randomly initialized. LoRA layers are another such example. This is, however, minor since these are automatically marked trainable upon injection.

- If you are loading a summary for a selectively fine-tuned model into a pretrained model, it is essential to ensure that all modules have the same initialization as during training. For pretrained modules such as key, query and value matrices this is guaranteed; for other parameters like classifier heads and lora matrices, however, this is not (since these are initialized from scratch). Therefore it is essential to have the same seed during inference and training (in case seperate scripts are used). This can be achieved using the following snippet:

  ```
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  ```
## Contributing

We welcome contributions to the `selective_optimizers` package! If you'd like to add a new selective optimizer, follow these steps:

1. **Create a new file** inside the `optimizers/` folder.
2. **Subclass `optimizers/base_optimizer`** in your new file.
3. **Override `init_chosen()`** to set the initial masks for the parameters.
4. **Override `update_chosen()`** to define how the masks evolve with each step. Note that since the selection is incremental, you will have to ensure that the updates are incremental, meaning that previously chosen parameters cannot be marked as unchosen.
5. **Open a pull request** with your new optimizer, and we'll be happy to review it!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this toolkit in your research, please cite our paper:

```bibtex
@article{Agarwal2024_step_by_step,
  title={Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models},
  author={Agarwal, Aradhye and Ramesh, Suhas Kamasetty and Sengupta, Ayan and Chakraborty, Tanmoy},
  journal={arXiv preprint arXiv:2408.14470},
  year={2024},
}
```

## Contact

For any questions or issues, feel free to open an issue on the GitHub repository or contact us directly.
