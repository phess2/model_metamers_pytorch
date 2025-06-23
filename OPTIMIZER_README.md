# Optimizer Support for Metamer Generation

This document explains how to use different optimizers for metamer generation in the model_metamers_pytorch project.

## Overview

The metamer generation system supports two optimizers for creating adversarial examples:

- **SGD** (default): Uses gradient descent with L2 norm constraints
- **Muon**: Uses the Muon optimizer for potentially better convergence

## Usage

### Command Line Interface

```bash
# Use SGD optimizer (default)
python make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py 0 -O sgd

# Use Muon optimizer
python make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py 0 -O muon
```

### Submit Script

```bash
# In your submit script
python model_analysis_folders/visual_networks/resnet50/make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py $SLURM_ARRAY_TASK_ID -I 3000 -N 8 -O muon
```

### Programmatic Usage

You can also specify the optimizer when calling the model directly:

```python
# Use SGD optimizer
(predictions_out, rep_out, all_outputs_out), xadv = model(
    im_n,
    invert_rep.clone(),
    make_adv=True,
    optimizer="sgd",
    **synth_kwargs,
    with_latent=True,
    fake_relu=True,
)

# Use Muon optimizer
(predictions_out, rep_out, all_outputs_out), xadv = model(
    im_n,
    invert_rep.clone(),
    make_adv=True,
    optimizer="muon",
    **synth_kwargs,
    with_latent=True,
    fake_relu=True,
)
```

## Implementation Details

### SGD Optimizer (Default)

- Uses gradient descent with fixed step size
- Applies L2 norm constraints via projection
- Uses the `step_size` parameter as the step size
- This is the original implementation from the robustness library

### Muon Optimizer

- Uses the Muon optimizer library for gradient updates
- Uses the `step_size` parameter as the learning rate
- May provide better convergence in some cases
- Automatically falls back to SGD if Muon is not installed

### Fallback Behavior

If the Muon optimizer is not available, the system will:
1. Print a warning message
2. Automatically fall back to using the SGD optimizer
3. Continue with the metamer generation process

## Installation

### SGD Optimizer
No additional installation required - this is the default behavior.

### Muon Optimizer
To use the Muon optimizer, install it using pip:

```bash
pip install muon
```

## Example Script

Run the example script to test both optimizers:

```bash
python example_optimizer_usage.py
```

This will demonstrate how to use both optimizers and show the differences in their behavior.

## Performance Comparison

The choice between SGD and Muon optimizers may affect:
- **Convergence speed**: How quickly the optimization reaches a good solution
- **Final loss**: The quality of the generated metamer
- **Computational efficiency**: Time and memory requirements

We recommend testing both optimizers on your specific use case to determine which works best for your needs.

## Troubleshooting

### Muon Import Error
If you see a warning about Muon not being found, the system will automatically fall back to SGD. This is normal behavior and doesn't indicate an error.

### Performance Issues
If you experience performance issues with either optimizer:
1. Try reducing the number of iterations (`-I` parameter)
2. Adjust the step size (`-Z` parameter)
3. Consider using a different constraint type

## Technical Notes

The optimizer selection is implemented in the `robustness.attacker.Attacker` class:
- The `forward` method accepts an `optimizer` parameter
- The optimization loop conditionally uses either SGD or Muon based on this parameter
- All existing functionality remains unchanged when using the default SGD optimizer 