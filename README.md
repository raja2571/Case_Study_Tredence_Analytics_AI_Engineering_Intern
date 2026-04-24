🧠✂️ Self-Pruning Neural Network

Tredence Studio — AI Agents Engineering Internship Case Study

This project implements a self-pruning Feed-Forward Neural Network in PyTorch.
Unlike traditional pruning methods applied after training, this model learns to prune its own weights during training using a differentiable gating mechanism.

🎯 Core Idea: PrunableLinear Layer

Instead of using standard torch.nn.Linear, we introduce a custom layer that includes learnable gates.

Each weight has a corresponding parameter called gate_scores, which controls whether the connection stays active.

🔁 Forward Pass
def forward(self, x):
    # 1. Convert gate scores into values between 0 and 1
    gates = torch.sigmoid(self.gate_scores)
    
    # 2. Apply element-wise pruning
    pruned_weights = self.weight * gates
    
    # 3. Perform linear transformation
    return F.linear(x, pruned_weights, self.bias)
🧮 Sparsity Regularization

To encourage pruning, we add a sparsity term to the loss function:

Total Loss=Classification Loss+λ∑σ(gate_scores)
💡 Why L1 on Sigmoid?
Sigmoid output ∈ (0, 1)
Summing all gates ≈ L1 norm
Minimizing this term pushes gates → 0
This forces corresponding weights → inactive (pruned)

👉 The only way for a sigmoid output to approach 0 is:

gate_scores→−∞
⚖️ Trade-off Parameter (λ)

Controls the balance between:

🎯 Accuracy
✂️ Sparsity

Higher λ → More pruning, but lower accuracy.

📊 Experimental Results (CIFAR-10)
Lambda (λ)	Test Accuracy (%)	Sparsity Level (%)
0.0	52.36%	0.00%
1e-5	51.10%	45.12%
5e-5	48.74%	88.35%
1e-4	41.21%	96.01%

⚠️ Results may vary slightly due to random initialization.

📈 Gate Distribution Analysis

For λ = 5e-5, the learned gate values show:

🔴 Large spike at 0 → pruned connections
🟢 Small cluster near 1 → important connections

This confirms that the model automatically identifies and removes redundant weights.

🚀 How to Run
📦 Prerequisites
Python 3.8+
PyTorch
Torchvision
NumPy
Matplotlib
🔧 Installation
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
pip install torch torchvision matplotlib numpy
▶️ Train the Model
python train_pruning.py
⚡ Features
✅ Dynamic weight pruning during training
✅ Learnable gating mechanism
✅ L1-based sparsity regularization
✅ Works with CPU, CUDA, or Apple MPS
✅ Automatic result logging and visualization
📂 Output
📊 Training logs printed in terminal
📈 Gate distribution plots saved in /results directory
🔮 Future Improvements
Structured pruning (channel/filter-level)
Apply to CNN architectures (ResNet, VGG)
Inference speed benchmarking
Hardware-aware pruning
🤝 Acknowledgment

Developed as part of the Tredence Studio AI Agents Engineering Internship Case Study.
