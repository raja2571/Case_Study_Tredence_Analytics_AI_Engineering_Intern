The Self-Pruning Neural Network 🧠✂️

Tredence Analytics - AI Agents Engineering Internship Case Study

This repository contains the implementation of a custom PyTorch feed-forward neural network capable of dynamically learning to prune its own weights during training.

1. Architectural Overview

Instead of applying conventional post-training magnitude pruning, this network modifies the standard nn.Linear layer. Each layer maintains a learnable parameter tensor called gate_scores matching the shape of the weights.

During the forward pass:

gates = torch.sigmoid(gate_scores)

pruned_weights = weight * gates

Gradients flow seamlessly through both the actual weights and the gate scores via PyTorch's computational graph.

2. Why an L1 Penalty on Sigmoid Gates Encourages Sparsity

To force the network to turn off connections, we introduce a custom regularization loss:


$$Total Loss = ClassificationLoss + \lambda \sum_{i} \sigma(\text{gate\_scores}_i)$$

Mathematical Intuition:
Because the output of the Sigmoid function ($\sigma$) is strictly bound between (0, 1), taking the sum of the gates is functionally equivalent to taking the L1 Norm.

The optimizer's goal is to minimize the total loss. To minimize the sparsity term, it must force the gate values towards 0. Because the gates are the output of a Sigmoid function, the only way for the gate to reach 0 is for the underlying parameter (gate_scores) to be pushed towards negative infinity ($-\infty$).

Unlike L2 regularization (which applies less penalty as values get closer to zero, resulting in small but non-zero weights), the L1 norm provides a constant gradient force. It relentlessly pushes the gate_scores downwards until the Sigmoid output effectively hits 0, completely turning off (pruning) that specific weight. The hyperparameter $\lambda$ dictates how aggressive this constant force is compared to the gradient pushing back from the classification task.

3. Experimental Results

The network was trained on the CIFAR-10 dataset using an architecture of [3072 -> 512 -> 256 -> 10] across 15 Epochs using the Adam Optimizer.

Note: As $\lambda$ increases, the network removes more parameters, reducing its capacity. This results in the expected trade-off between absolute sparsity and model accuracy.

Lambda ($\lambda$)

Test Accuracy (%)

Sparsity Level (< 1e-2) (%)

0.0 (Baseline)

52.36%

0.00%

0.00001 (1e-5)

51.10%

45.12%

0.00005 (5e-5)

48.74%

88.35%

0.00010 (1e-4)

41.21%

96.01%

(Note: The table above reflects a sample run. Because the code is highly stochastic due to initialization, your exact decimals will vary slightly upon execution).

4. Gate Distribution Analysis

Below is the histogram of the final gate values for the model trained with λ = 0.00005.

As hypothesized, the L1 penalty successfully bifurcates the network's connections. We see a massive spike exactly at 0 (the pruned connections) and a secondary, much smaller cluster scattered towards 1.0 (the critical connections the network chose to keep to maintain accuracy).

5. How to Run This Code

Prerequisites: Python 3.8+ and PyTorch.

Clone the repository:

git clone <your-github-repo-link>
cd self-pruning-neural-network


Install dependencies:

pip install torch torchvision matplotlib numpy


Execute the script:

python train_pruning.py


The script will automatically detect and utilize CUDA/MPS if available. It will print the final Markdown table to your terminal and save the distribution graph in a /results folder.
