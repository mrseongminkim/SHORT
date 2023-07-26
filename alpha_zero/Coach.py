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

from utils.CToken import *

from config import *

log = logging.getLogger(__name__)

class Coach():
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.mcts = MCTS(self.game, self.nnet)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False

    def executeEpisode(self):
        trainExamples = []
        gfa: GFA = self.game.get_initial_gfa()
        CToken.clear_memory()
        episodeStep = 0
        while True:
            episodeStep += 1
            temp = 1
            pi = self.mcts.getActionProb(gfa, temp=temp)
            gfa_representation = self.game.gfa_to_tensor(gfa)
            trainExamples.append([gfa_representation, pi])
            action = np.random.choice(len(pi), p=pi)
            gfa = self.game.getNextState(gfa, action)
            r = self.game.getGameEnded(gfa)
            if r != None:
                r = -r
                return [(x[0], x[1], r) for x in trainExamples]

    def learn(self):
        for i in range(1, NUMBER_OF_ITERATIONS + 1):
            log.info(f"Starting Iter #{i} ...")
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=MAXIMUM_LENGTH_OF_QUEUE)
                for _ in tqdm(range(NUMBER_OF_EPISODES), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet)
                    iterationTrainExamples += self.executeEpisode()
                self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > NUMBER_OF_ITERATIONS_FOR_TRAIN_EXAMPLES_HISTORY:
                log.warning(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            self.saveTrainExamples(i - 1)
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            self.nnet.train(trainExamples)
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
