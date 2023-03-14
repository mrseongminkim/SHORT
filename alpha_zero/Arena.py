import logging
import copy

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        curPlayer = 1
        board_x = self.game.getInitBoard()
        board_y = copy.deepcopy(board_x)
        while self.game.getGameEnded(board_x, curPlayer) == -1:
            action = self.player1(
                self.game.getCanonicalForm(board_x, curPlayer))
            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board_x, curPlayer), 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board_x, curPlayer = self.game.getNextState(
                board_x, curPlayer, action)
        while self.game.getGameEnded(board_y, curPlayer) == -1:
            action = self.player2(
                self.game.getCanonicalForm(board_y, curPlayer))
            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board_y, curPlayer), 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board_y, curPlayer = self.game.getNextState(
                board_y, curPlayer, action)
            
        board_x = self.game.gfaToBoard(board_x)
        board_y = self.game.gfaToBoard(board_y)

            
            
        if board_x[0][self.game.n - 1].treeLength() < board_y[0][self.game.n - 1].treeLength():
            return 1
        elif board_x[0][self.game.n - 1].treeLength() == board_y[0][self.game.n - 1].treeLength():
            return 0
        else:
            return -1

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
        return oneWon, twoWon, draws
