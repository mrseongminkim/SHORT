import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
from FAdo.conversions import *

from alpha_zero.MCTS import MCTS
from alpha_zero.state_elimination.NNet import NNetWrapper
from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame

from utils.CToken import *

from config import *

log = logging.getLogger(__name__)

class Coach():
    def __init__(self, game, nnet):
        self.game: StateEliminationGame = game
        self.nnet: NNetWrapper = nnet
        self.mcts = MCTS(self.game, self.nnet)
        self.trainExamplesHistory: list = []
        self.skipFirstSelfPlay = False
        #self.load_valid_data()

    def executeEpisode(self):
        trainExamples = []
        gfa: GFA = self.game.get_initial_gfa()
        while self.game.getGameEnded(gfa) != None:
            gfa: GFA = self.game.get_initial_gfa()
        CToken.clear_memory()
        while True:
            pi = self.mcts.getActionProb(gfa)

            #pi 구할 때도 결국 gfa_to_tensor 쓰고 predict를 쓰니까 일단 밑에 먼저 볼까
            gfa_representation = self.game.gfa_to_tensor(gfa) #GFA -> feature extraction 과정이 다 보이고
            #넣어줄 때는 best가 아닌 전부 선택가능한 pi를 넣어줌
            trainExamples.append([gfa_representation, pi])
            best_actions = np.array(np.argwhere(pi == np.max(pi))).flatten()
            best_action = np.random.choice(best_actions) #이게 pi의 최댓값이랑 같은지 봐야 할 듯
            best_pi = [0] * len(pi)
            best_pi[best_action] = 1
            action = np.random.choice(len(best_pi), p=best_pi)
            #action이랑 best_action이랑 같은지 봐야 할 듯
            #굳이 이렇게 하는 이유가 있나...?
            #다음 state로 제대로 삭제되는지 직접 해보는 것도 좋을듯
            gfa = self.game.getNextState(gfa, action)
            #getGameEnded는 일단은 검증 완료함
            r = self.game.getGameEnded(gfa)
            if r != None:
                #NN이 음수보다 양수를 리턴하게 시키고 싶어서 -r을 해줌
                r = -r
                assert r >= 0
                #graph, pi, v 순서로 저장
                return [(x[0], x[1], r) for x in trainExamples]
    
    def make_fake_data(self):
        "graph, pis, vs가 제대로 나눠지는지 확인하기 위한 가짜 데이터"
        import torch
        from torch_geometric.data import Data
        fake_data = []
        for i in range(256):
            x_0 = Data(x=torch.LongTensor([i]))
            x_1 = [i, i]
            x_2 = i
            fake_data.append((x_0, x_1, x_2))
        return fake_data

    def learn(self):
        for i in range(1, NUMBER_OF_ITERATIONS + 1):
            log.info(f"Starting Iter #{i} ...")
            print(f"Iter {i}")
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=MAXIMUM_LENGTH_OF_QUEUE)
                for _ in tqdm(range(NUMBER_OF_EPISODES), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet)
                    iterationTrainExamples += self.executeEpisode()
                self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > NUMBER_OF_ITERATIONS_FOR_TRAIN_EXAMPLES_HISTORY:
                log.warning(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            if i % 100 == 0 or i == 1:
                self.saveTrainExamples(i - 1)
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            self.nnet.train(trainExamples)
            #log.info("Testing for valid data")
            #self.nnet.test_valid_data(self.valid_data)
            log.info("ACCEPTING NEW MODEL")
            self.nnet.save_checkpoint(folder=CHECKPOINT, filename=self.getCheckpointFile(i))

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = CHECKPOINT
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed
    
    def load_initial_data(self):
        examplesFile = os.path.join(LOAD_FOLDER_FILE[0], "initial_data.tar")
        with open(examplesFile, "rb") as f:
            self.trainExamplesHistory = Unpickler(f).load()
        self.skipFirstSelfPlay = True

    def loadTrainExamples(self):
        modelFile = os.path.join(LOAD_FOLDER_FILE[0], LOAD_FOLDER_FILE[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')
            self.skipFirstSelfPlay = True

    def load_valid_data(self):
        examplesFile = os.path.join(LOAD_FOLDER_FILE[0], "valid_data.tar")
        with open(examplesFile, "rb") as f:
            valid_data = Unpickler(f).load()
        self.valid_data = []
        for e in valid_data:
            self.valid_data.extend(e)
        log.info("Valid data loaded")
