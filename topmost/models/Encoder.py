import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, vocab_size, num_topic, hidden_dim, dropout, num_experts):
        super().__init__()
        
        self.num_experts = num_experts
        self.experts = nn.ModuleList([SingleMLPEncoder(vocab_size, num_topic, hidden_dim, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(vocab_size, num_experts)
        self.std_dev = 1.0  # Controls how much weight is given to nearby experts
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu
    
    def forward(self, x, time_index):
        batch_size = x.shape[0]
        time_indices = time_index.unsqueeze(1).expand(-1, self.num_experts)  # Expand to batch_size x num_experts
        expert_indices = torch.arange(self.num_experts, device=x.device).expand(batch_size, -1)  # Expand to match batch
        
        # Compute proximity-based weights (Gaussian similarity to the timestamp)
        dist = -((expert_indices - time_indices) ** 2) / (2 * self.std_dev ** 2)
        gate_scores = F.softmax(dist, dim=-1)  # Softmax ensures proper probability distribution
        
        mu_list, logvar_list, theta_list = [], [], []
        for i, expert in enumerate(self.experts):
            theta, mu, logvar = expert(x)
            mu_list.append(mu.unsqueeze(1) * gate_scores[:, i].unsqueeze(-1))
            logvar_list.append(logvar.unsqueeze(1) * gate_scores[:, i].unsqueeze(-1))
            theta_list.append(theta.unsqueeze(1) * gate_scores[:, i].unsqueeze(-1))
        
        mu = torch.sum(torch.cat(mu_list, dim=1), dim=1)
        logvar = torch.sum(torch.cat(logvar_list, dim=1), dim=1)
        theta = torch.sum(torch.cat(theta_list, dim=1), dim=1)
        
        return theta, mu, logvar

class SingleMLPEncoder(nn.Module):
    def __init__(self, vocab_size, num_topic, hidden_dim, dropout):
        super().__init__()
        self.fc11 = nn.Linear(vocab_size, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, num_topic)
        self.fc22 = nn.Linear(hidden_dim, num_topic)
        
        self.fc1_drop = nn.Dropout(dropout)
        self.z_drop = nn.Dropout(dropout)
        
        self.mean_bn = nn.BatchNorm1d(num_topic, affine=True)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topic, affine=True)
        self.logvar_bn.weight.requires_grad = False
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu
    
    def forward(self, x):
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_drop(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        theta = self.reparameterize(mu, logvar)
        theta = F.softmax(theta, dim=1)
        theta = self.z_drop(theta)
        return theta, mu, logvar
