from torch import nn

class IdentityEmbedding(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.embedding_dim = input_size
        self.num_embedding = -1
        
    def forward(self, x):
        return x
