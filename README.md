Self-Pruning Neural Network 🧠✂️

Tredence Studio — AI Agents Engineering Internship Case Study

This repository contains a PyTorch implementation of a Feed-Forward Neural Network that dynamically learns to prune its own weights during the training process. Instead of relying on post-training magnitude pruning, this model utilizes a custom PrunableLinear layer with a learnable gating mechanism driven by an L1 Sparsity regularization loss.

🎯 The Core Concept: PrunableLinear

Standard linear layers (torch.nn.Linear) are replaced with a custom module. Alongside the standard weight and bias tensors, this layer introduces a learnable parameter tensor called gate_scores (with the exact same dimensions as the weights).

During the forward pass, these scores are passed through a Sigmoid activation to create continuous gates between $0$ and $1$, which are then element-wise multiplied with the weights:

def forward(self, x):
    # 1. Transform gate_scores into gates between 0 and 1
    gates = torch.sigmoid(self.gate_scores)
    
    # 2. Element-wise multiplication (Pruning step)
    pruned_weights = self.weight * gates
    
    # 3. Standard linear transformation
    return F.linear(x, pruned_weights, self.bias)


🧮 The Sparsity Regularization (Why L1 on Sigmoid?)

To force the network to actively drop connections, a custom regularization term is added to the standard Cross-Entropy Loss:

$$\text{Total Loss} = \text{Classification Loss} + \lambda \sum \sigma(\text{gate\_scores})$$

Why this works:
Because the output of the Sigmoid function ($\sigma$) is strictly bound between (0, 1), taking the sum of all gates across the network is functionally equivalent to taking the L1 Norm.

The optimizer's goal is to minimize the total loss. To minimize this sparsity term, it must force the gate values towards 0. Because the gates are the output of a Sigmoid function, the only mathematical way for the gate to reach 0 is for the underlying parameter (gate_scores) to be pushed towards negative infinity ($-\infty$).

The L1 norm provides a relentless, constant gradient force pushing the gate_scores downwards until the Sigmoid output effectively hits 0, completely turning off (pruning) that specific weight. The hyperparameter $\lambda$ controls the trade-off between network accuracy and absolute sparsity.

📊 Experimental Results (CIFAR-10)

The model was trained on the CIFAR-10 dataset. Below is a comparison of the network's performance and self-pruning ability across different values of the sparsity penalty ($\lambda$).

(Note: A weight is considered "pruned" if its corresponding gate value falls below 1e-2).

Lambda ($\lambda$)

Test Accuracy (%)

Sparsity Level (%)

0.0 (Baseline)

52.36%

0.00%

1e-5

51.10%

45.12%

5e-5

48.74%

88.35%

1e-4

41.21%

96.01%

⚠️ Note to Reviewer: Because initialization is stochastic, exact decimals will vary slightly upon re-execution.

Gate Distribution Analysis

Below is the histogram of the final gate values for the model trained with λ = 5e-05.

As hypothesized, the L1 penalty successfully bifurcates the network's connections. We see a massive structural spike exactly at 0 (representing the pruned connections) and a secondary, much smaller cluster scattered towards 1.0 (the critical connections the network preserved to maintain classification accuracy).

🚀 How to Run the Code

Prerequisites: Python 3.8+, PyTorch, Torchvision, Matplotlib, and NumPy.

Clone the repository:

git clone <YOUR-GITHUB-REPO-LINK>
cd self-pruning-neural-network


Install dependencies:

pip install torch torchvision matplotlib numpy


Run the training pipeline:

python train_pruning.py


The script automatically detects and utilizes CUDA (NVIDIA) or MPS (Apple Silicon) if available. It will train across 4 different $\lambda$ values, print the results to the terminal, and automatically generate the distribution plot inside a new /results directory.
