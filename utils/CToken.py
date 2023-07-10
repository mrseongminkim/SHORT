from FAdo.reex import *

from config import *

class CToken(RegExp):
    token_to_regex = dict()
    token_to_string = dict()
    threshold = THRESHOLD

    def __init__(self, regex: RegExp):
        self.hashed_value = hash(regex)
        self.tree_length = regex.treeLength()
        CToken.token_to_regex[self.hashed_value] = regex
        CToken.token_to_string[self.hashed_value] = regex._strP()


    def __str__(self):
        return CToken.token_to_string[self.hashed_value]


    _strP = __str__


    def __repr__(self):
        return "CToken(%d)" % self.hashed_value


    def treeLength(self):
        return self.tree_length


    def __copy__(self):
        return CToken(CToken.token_to_regex[self.hashed_value])

    @classmethod
    def clear_memory(cls):
        del CToken.token_to_regex
        del CToken.token_to_string
        CToken.token_to_regex = dict()
        CToken.token_to_string = dict()
