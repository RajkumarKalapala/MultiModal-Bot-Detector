import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models
from torch_geometric.nn import GCNConv

class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.out_dim = self.transformer.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Using weights=... is the modern, non-deprecated way to load ResNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = 512

    def forward(self, images):
        x = self.feature_extractor(images)
        return x.view(x.size(0), -1)

class GraphEncoder(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=64, out_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.out_dim = out_channels

    def forward(self, batch_data):
        x, edge_index = batch_data.x, batch_data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        central_node_indices = batch_data.ptr[:-1]
        return x[central_node_indices]

class MultiModalDetector(nn.Module):
    def __init__(self, text_dim=768, vision_dim=512, graph_dim=128):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.vision_encoder = VisionEncoder()
        self.graph_encoder = GraphEncoder(out_channels=graph_dim)

        fused_dim = text_dim + vision_dim + graph_dim

        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, text_inputs, images, graph_batch):
        text_emb = self.text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])
        vision_emb = self.vision_encoder(images)
        graph_emb = self.graph_encoder(graph_batch)
        fused_vector = torch.cat([text_emb, vision_emb, graph_emb], dim=1)
        return self.mlp(fused_vector)
