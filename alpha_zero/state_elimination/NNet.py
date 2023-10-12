import os

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

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

    def train(self, examples):
        optimizer = optim.AdamW(self.nnet.parameters(), lr=LR)
        for epoch in range(EPOCHS):
            print('EPOCH ::: ' + str(epoch + 1))
            if epoch == EPOCHS - 1:
                self.verbose = True
            else:
                self.verbose = False
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = len(examples) // BATCH_SIZE
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=BATCH_SIZE)
                gfas, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                gfas = torch.LongTensor(np.array(gfas))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                if CUDA:
                    gfas, target_pis, target_vs = gfas.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                out_pis, out_vs = self.nnet(gfas)
                l_pi = self.loss_pi(target_pis, out_pis)
                l_v = self.loss_v(target_vs, out_vs)
                total_loss = l_pi + l_v
                pi_losses.update(l_pi.item(), gfas.size(0))
                v_losses.update(l_v.item(), gfas.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.verbose = False
            print("pi loss:", pi_losses.avg)
            print("v loss:", v_losses.avg)

    def test_valid_data(self, examples):
        self.nnet.eval()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        batch_count = 1
        t = tqdm(range(batch_count), desc='Run for valid data')
        self.verbose = True
        for _ in t:
            sample_ids = [i for i in range(len(examples))]
            gfas, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
            gfas = torch.LongTensor(np.array(gfas))
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
            if CUDA:
                gfas, target_pis, target_vs = gfas.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
            out_pis, out_vs = self.nnet(gfas)
            l_pi = self.loss_pi(target_pis, out_pis)
            l_v = self.loss_v(target_vs, out_vs)
            pi_losses.update(l_pi.item(), gfas.size(0))
            v_losses.update(l_v.item(), gfas.size(0))
            t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
        self.verbose = False

    def predict(self, gfa):
        self.nnet.eval()
        with torch.no_grad():
            if CUDA:
                gfa = torch.LongTensor(np.array(gfa)).unsqueeze(0)
                gfa = gfa.contiguous().cuda()
            pi, v = self.nnet(gfa)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        if self.verbose:
            print("targets:", targets[0][:12])
            print("outputs:", torch.exp(outputs[0][:12]))
        return -torch.sum(targets * outputs) / targets.size()[0]

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
