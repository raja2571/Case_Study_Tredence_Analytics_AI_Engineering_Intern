import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# ==========================================
# 1. The Custom "Prunable" Linear Layer
# ==========================================
class PrunableLinear(nn.Module):
    """
    A custom linear layer that learns to prune its own weights dynamically 
    during training by optimizing a gate score for each individual connection.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # The learnable gate parameters (same shape as weights)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters properly to ensure stable training at the start."""
        # Standard Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Standard initialization for biases
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize gate_scores slightly positive so the initial Sigmoid output (the gate) 
        # is around ~0.73. This prevents dead neurons at the very beginning of training.
        nn.init.normal_(self.gate_scores, mean=1.0, std=0.1)

    def forward(self, x):
        # 1. Transform gate_scores into gates between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # 2. Element-wise multiplication to apply the pruning gates to the weights
        pruned_weights = self.weight * gates
        
        # 3. Perform standard linear transformation
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity_loss(self):
        """Calculates the L1 norm of the gates. Since sigmoid output is strictly positive, 
        the sum of the absolute values is just the sum of the gates."""
        gates = torch.sigmoid(self.gate_scores)
        return torch.sum(gates)

    def get_gate_values(self):
        """Returns the flattened gate values for evaluation and plotting."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu().numpy().flatten()


# ==========================================
# 2. The Neural Network Architecture
# ==========================================
class SelfPruningMLP(nn.Module):
    """
    A standard Feed-Forward Neural Network for CIFAR-10 utilizing the custom 
    PrunableLinear layers.
    """
    def __init__(self):
        super(SelfPruningMLP, self).__init__()
        # CIFAR-10 images are 3x32x32 = 3072 flattened features
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10) # 10 classes

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation, handled by CrossEntropyLoss
        return x

    def calculate_total_sparsity_loss(self):
        """Aggregates sparsity loss from all PrunableLinear layers in the model."""
        total_loss = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total_loss += module.get_sparsity_loss()
        return total_loss

    def calculate_sparsity_metrics(self, threshold=1e-2):
        """Calculates the percentage of gates that have been pruned (value < threshold)."""
        total_gates = 0
        pruned_gates = 0
        all_gates = []
        
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gate_values()
                all_gates.extend(gates)
                total_gates += len(gates)
                pruned_gates += np.sum(gates < threshold)
                
        sparsity_percentage = (pruned_gates / total_gates) * 100
        return sparsity_percentage, np.array(all_gates)


# ==========================================
# 3. Training & Evaluation Pipeline
# ==========================================
def train_and_evaluate(lmbda, device, trainloader, testloader, epochs=10):
    print(f"\n{'='*50}\nStarting Run with Lambda (λ) = {lmbda}\n{'='*50}")
    
    model = SelfPruningMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # --- TRAINING LOOP ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Loss formulation: Classification Loss + λ * Sparsity Loss
            classification_loss = criterion(outputs, labels)
            sparsity_loss = model.calculate_total_sparsity_loss()
            total_loss = classification_loss + (lmbda * sparsity_loss)
            
            # Backward pass & Optimize
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Total Loss: {running_loss/len(trainloader):.4f}")

    # --- EVALUATION LOOP ---
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    sparsity_level, all_gates = model.calculate_sparsity_metrics(threshold=1e-2)
    
    print(f"-> Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"-> Final Sparsity Level: {sparsity_level:.2f}%\n")
    
    return test_accuracy, sparsity_level, all_gates

def plot_gate_distribution(gates, lmbda):
    """Generates a histogram plot of the gate values."""
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of Gate Values ($\lambda={lmbda}$)')
    plt.xlabel('Gate Value (after Sigmoid)')
    plt.ylabel('Frequency (Number of Weights)')
    plt.yscale('log') # Log scale helps visualize the massive spike at 0
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/gate_distribution_lambda_{lmbda}.png'
    plt.savefig(filename)
    print(f"Saved distribution plot to {filename}")
    plt.close()


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == '__main__':
    # Determine best device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Loading CIFAR-10 Dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Experiment with different lambda values
    # 0.0 acts as our baseline standard network. 
    # Values around 1e-5 to 1e-4 usually work best for this specific architecture size.
    lambda_values = [0.0, 1e-5, 5e-5, 1e-4]
    results = []

    best_sparsity_all_gates = None
    best_lambda_to_plot = 5e-5 

    for lmbda in lambda_values:
        acc, sparsity, gates = train_and_evaluate(lmbda, device, trainloader, testloader, epochs=15)
        results.append((lmbda, acc, sparsity))
        
        if lmbda == best_lambda_to_plot:
             best_sparsity_all_gates = gates

    # Print Markdown Table to terminal
    print("\n\n=== FINAL RESULTS (Paste this into your Markdown) ===")
    print("| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity Level (%) |")
    print("| :--- | :--- | :--- |")
    for r in results:
        print(f"| {r[0]} | {r[1]:.2f}% | {r[2]:.2f}% |")

    # Generate Plot for the most interesting Lambda value
    if best_sparsity_all_gates is not None:
        plot_gate_distribution(best_sparsity_all_gates, best_lambda_to_plot)