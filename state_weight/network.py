import sys
sys.path.append(r"/home/toc/seongmin")
sys.path.append(r"/home/toc/seongmin/SHORT")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GATv2Conv

class ForwardBackwardGNN(nn.Module):
    def __init__(self):
        super(ForwardBackwardGNN, self).__init__()
        self.forward_conv1 = GATv2Conv(73, 73, add_self_loops=False)
        self.backward_conv1 = GATv2Conv(73, 73, add_self_loops=False)

    def forward(self, forward_graph, backward_graph=None):
        forward_graph = F.relu(self.forward_conv1(forward_graph.x, forward_graph.edge_index) + forward_graph.x)
        backward_graph = F.relu(self.backward_conv1(backward_graph.x, backward_graph.edge_index) + backward_graph.x)
        graph = torch.cat((forward_graph, backward_graph), dim=-1)
        return graph

class StateWeightNet(nn.Module):
    def __init__(self):
        super(StateWeightNet, self).__init__()
        self.forward_backward_gnn = ForwardBackwardGNN()
        self.lin1 = nn.Linear(146, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 1)

    def forward(self, forward_graph, backward_graph=None):
        graph = self.forward_backward_gnn(forward_graph, backward_graph)
        graph = F.relu(self.lin1(graph))
        graph = F.relu(self.lin2(graph))
        graph = F.relu(self.lin3(graph))
        graph = self.lin4(graph)
        graph = graph.view(-1, 7)
        return graph
