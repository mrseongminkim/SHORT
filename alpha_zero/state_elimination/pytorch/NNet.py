import os
import time

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import copy

from alpha_zero.utils import *
from alpha_zero.NeuralNet import NeuralNet
from alpha_zero.state_elimination.pytorch.StateEliminationNNet import StateEliminationNNet as sennet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.0,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
    'vocab_size': 16,
    'embedding_dim': 4
})

word_to_ix = {'(': 1, ')': 2, '*': 3, '+': 4, '@': 5}

for i in range(10):
    word_to_ix[str(i)] = i + 6


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = sennet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.game = game
        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.AdamW(self.nnet.parameters())
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = int(len(examples) / args.batch_size)
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(
                    len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                
                # print([self.board_to_tensor(board) for board in boards])
                boards = [self.game.gfaToBoard(board) for board in boards]
                
                boards = torch.stack(
                    [self.board_to_tensor(board) for board in boards], dim=0)
                
                # boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                        
    def board_to_tensor(self, board, max_len=20):
        new_board = [[None for i in range(self.board_x)] for j in range(self.board_y)]
        for i in range(len(board)):
            for j in range(len(board[i])):
                new_board[i][j] = [word_to_ix[word] for word in list(
                    str(board[i][j]).replace('@epsilon', '@').replace(' ', ''))]

                if len(new_board[i][j]) > max_len:
                    new_board[i][j] = new_board[i][j][:max_len]
                else:
                    new_board[i][j] = new_board[i][j] + \
                        [0]*(max_len-len(new_board[i][j]))

                assert len(new_board[i][j]) == max_len
                
        if args.cuda:
            new_board = torch.LongTensor(new_board).contiguous().cuda()
                
        return new_board

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()
        
        board = self.game.gfaToBoard(board)
        
        board = self.board_to_tensor(board)
                
        # new_board = np.zeros(self.board_x*self.board_y*max_len).reshape(self.board_x, self.board_y, max_len)       
        
        board = board.view(self.board_x, self.board_y, -1)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
