import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np

from alpha_zero.utils import *
from alpha_zero.Coach import Coach
from alpha_zero.state_elimination.NNet import NNetWrapper
from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame

from config import *

g = StateEliminationGame()
n = NNetWrapper(g)
c = Coach(g, n)
c.loadTrainExamples()
examples = []
for e in c.trainExamplesHistory:
    examples.extend(e)

graphs, pis, vs = list(zip(*examples))
model = n.nnet
optimizer = optim.AdamW(model.parameters(), lr=LR)
for epoch in range(EPOCHS):
    print('EPOCH ::: ' + str(epoch + 1))
    model.train()
    pi_losses = AverageMeter()
    v_losses = AverageMeter()
    batch_count = int(len(examples) / BATCH_SIZE)
    t = tqdm(range(batch_count), desc='Training Net')
    for _ in t:
        sample_ids = np.random.randint(len(examples), size=BATCH_SIZE)
        graphs, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
        batch_loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
        batch = next(iter(batch_loader))
        #전처리 필요
        target_pis = torch.FloatTensor(np.array(pis))
        max_values, max_indices = torch.max(target_pis, dim=1)
        target_pis = torch.zeros_like(target_pis)
        target_pis[torch.arange(128), max_indices] = 1
        target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
        if CUDA:
            batch, target_pis, target_vs = batch.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
        out_pis, out_vs = model(batch)
        l_pi = n.loss_pi(target_pis, out_pis)
        l_v = n.loss_v(target_vs, out_vs)
        total_loss = l_pi + l_v
        pi_losses.update(l_pi.item(), len(graphs))
        v_losses.update(l_v.item(), len(graphs))
        t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()