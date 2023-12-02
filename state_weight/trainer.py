import sys
sys.path.append(r"/home/toc/seongmin")
sys.path.append(r"/home/toc/seongmin/SHORT")

from tqdm import tqdm

import torch
import torch.optim as optim

from torch_geometric.loader import DataLoader

from StateEliminationGame import StateEliminationGame
from dataset import StateWeightDataset
from network import StateWeightNet

game = StateEliminationGame(maxN=5)
gfa_transform = game.gfa_to_tensor
batch_size = 32
lr = 0.001
epoch = 2000
model = StateWeightNet().cuda()
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

training_data_size = 296135 #upto gfa79999
state_weight_data = StateWeightDataset("./state_weight_data/annotations_file.csv", "./state_weight_data/", fa_transform=gfa_transform)
train_data = state_weight_data[:training_data_size]
test_data = state_weight_data[training_data_size:]
train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test = DataLoader(test_data, batch_size=batch_size, shuffle=False)

minimum_loss = float("inf")
print(f"Train on {len(train_data)}, validate on {len(test_data)} samples.")
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
    print("result:", result[0])
    print("label:", label[0])
    #'''
    with torch.no_grad():
        model.eval()
        total_loss = 0
        count = 0
        for fa, label in tqdm(test):
            #forward_graph = fa.contiguous().cuda()
            forward_graph = fa[0].contiguous().cuda()
            backward_graph = fa[1].contiguous().cuda()
            label = label.contiguous().cuda()
            result = model(forward_graph, backward_graph)
            loss = criterion(result, label)
            total_loss += loss.item()
            count += 1
        print(f"Valid - Epoch {i + 1}, loss: {total_loss / count}")
        print("result:", result[0])
        print("label:", label[0])
        #if total_loss < minimum_loss:
        #    minimum_loss = total_loss
        #    torch.save(model.left_right_gnn.state_dict(), f"{total_loss / count}gnn.pth")
    #'''
