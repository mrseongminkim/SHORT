import sys
sys.path.append(r"/home/toc/seongmin")
sys.path.append(r"/home/toc/seongmin/SHORT")

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader

from StateEliminationGame import StateEliminationGame
from dataset import StateWeightDataset
from network import StateWeightNet

'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2"
'''

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=1024)

game = StateEliminationGame(maxN=50)
gfa_transform = game.gfa_to_tensor
batch_size = 64
lr = 0.001
epoch = 2000
#model = nn.DataParallel(StateWeightNet()).cuda()
model = StateWeightNet().cuda()
criterion = torch.nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

#training_data_size = 296135 #upto gfa79999
state_weight_data = StateWeightDataset("./state_weight_data2/annotations_file.csv", "./state_weight_data2/", fa_transform=gfa_transform)
valid_data = StateWeightDataset("./valid_data2/annotations_file.csv", "./valid_data2/", fa_transform=gfa_transform)
#train_data = state_weight_data[:training_data_size]
#test_data = state_weight_data[training_data_size:]
train = DataLoader(state_weight_data, batch_size=batch_size, shuffle=True)
test = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

minimum_loss = float("inf")
print(f"Train on {len(state_weight_data)}, validate on {len(valid_data)} samples.")
for i in range(epoch):
    model.train()
    total_loss = 0
    count = 0
    for fa, label in tqdm(train):
        #forward_graph = fa.contiguous().cuda()
        forward_graph = fa[0].contiguous().cuda()
        backward_graph = fa[1].contiguous().cuda()
        label = label.contiguous().cuda()   
        result = model(forward_graph, backward_graph)
        loss = criterion(result, label)
        total_loss += loss.item()
        count += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Train - Epoch {i + 1}, loss: {total_loss / count}")
    print("result:", result[0][:8])
    print("label:", label[0][:8])
    with torch.no_grad():
        model.eval()
        total_loss = 0
        count = 0
        for fa, label in tqdm(test):
            forward_graph = fa[0].contiguous().cuda()
            backward_graph = fa[1].contiguous().cuda()
            label = label.contiguous().cuda()
            result = model(forward_graph, backward_graph)
            loss = criterion(result, label)
            total_loss += loss.item()
            count += 1
        print(f"Valid - Epoch {i + 1}, loss: {total_loss / count}")
        print("result:", result[0][:8])
        print("label:", label[0][:8])
        if total_loss < minimum_loss:
            minimum_loss = total_loss
            torch.save(model.forward_backward_gnn.state_dict(), f"{total_loss / count}gnn.pth")
