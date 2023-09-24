import os

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torchviz import make_dot

from alpha_zero.utils import *
from alpha_zero.state_elimination.StateEliminationNNet import StateEliminationNNet as sennet

from config import *

class NNetWrapper():
    def __init__(self, game):
        self.nnet = sennet(game)
        self.action_size = game.getActionSize()
        if CUDA:
            self.nnet.cuda()
        self.verbose = False
        self.kld = torch.nn.KLDivLoss(reduction='batchmean')

    def train(self, examples):
        optimizer = optim.AdamW(self.nnet.parameters(), lr=LR)
        for epoch in range(EPOCHS):
            print('EPOCH ::: ' + str(epoch + 1))
            #print(self.nnet.policy_head1.weight)
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = len(examples) // BATCH_SIZE
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=BATCH_SIZE)
                graphs, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                batch_loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
                batch = next(iter(batch_loader))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                #batch로 인해서 문제 안 생긴다.
                if CUDA:
                    batch, target_pis, target_vs = batch.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                out_pis, out_vs = self.nnet(batch)
                l_pi = self.loss_pi(target_pis, out_pis)
                l_v = self.loss_v(target_vs, out_vs)
                total_loss = l_pi + l_v
                pi_losses.update(l_pi.item(), len(graphs))
                v_losses.update(l_v.item(), len(graphs))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def test_valid_data(self, examples):
        self.nnet.eval()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        batch_count = 1
        t = tqdm(range(batch_count), desc='Run for valid data')
        self.verbose = True
        for _ in t:
            sample_ids = [i for i in range(len(examples))]
            graphs, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
            batch_loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
            batch = next(iter(batch_loader))
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
            #batch로 인해서 문제 안 생긴다.
            if CUDA:
                batch, target_pis, target_vs = batch.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
            out_pis, out_vs = self.nnet(batch)
            l_pi = self.loss_pi(target_pis, out_pis)
            l_v = self.loss_v(target_vs, out_vs)
            pi_losses.update(l_pi.item(), len(graphs))
            v_losses.update(l_v.item(), len(graphs))
            t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
        self.verbose = False

    def predict(self, graph):
        batch_loader = DataLoader([graph], batch_size=1, shuffle=False)
        self.nnet.eval()
        with torch.no_grad():
            batch = next(iter(batch_loader))
            if CUDA:
                batch.contiguous().cuda()
            pi, v = self.nnet(batch)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        if VERBOSE:
            print("targets:", targets[0][:10])
            print("outputs:", torch.exp(outputs[0][:10]))
        return -torch.sum(targets * outputs) / targets.size()[0]
        #return self.kld(outputs, targets)

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        torch.save({"state_dict": self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if CUDA else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
