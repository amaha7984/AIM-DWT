import torch
import torch.nn as nn
from aim.v1.torch.layers import AttentionPoolingClassifier
from dwt import HaarTransform

class AIMClassificationModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(AIMClassificationModel, self).__init__()

        embed_dim = base_model.preprocessor.patchifier.proj.out_channels
        num_heads = base_model.trunk.blocks[0].attn.num_heads
        self.base_model = base_model
        self.attention_pool = AttentionPoolingClassifier(
            dim=embed_dim,          
            out_features=embed_dim, # Matching output features to embedding dimension
            num_heads=num_heads,   
            num_queries=1, 
            use_batch_norm=True,
            qkv_bias=True, linear_bias=True, average_pool=True
        )
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dwt = HaarTransform(level=1)

    def forward(self, x):

        # Applying DWT
        Yl, xH, xV, xD = self.dwt(x)

        Yl = self.base_model(Yl)
        xH = self.base_model(xH)
        xV = self.base_model(xV)
        xD = self.base_model(xD)

        # Apply Attention Pooling on the processed subbands
        Yl = self.attention_pool(self.base_model(Yl))
        xH = self.attention_pool(self.base_model(xH))
        xV = self.attention_pool(self.base_model(xV))
        xD = self.attention_pool(self.base_model(xD))

        x = torch.cat([Yl, xH, xV, xD], dim=1)
        return self.fc(x)
