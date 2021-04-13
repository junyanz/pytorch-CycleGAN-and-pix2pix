import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool

class GraphNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(num_classes, num_classes)
        self.bn1 = BatchNorm(num_classes)
        self.fc = torch.nn.Linear(num_classes, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(self.bn1(x))
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x
