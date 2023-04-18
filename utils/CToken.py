from FAdo.reex import *

class CToken(RegExp):
    token_to_regex = dict()
    token_to_string = dict()
    threshold = 10

    def __init__(self, regex: RegExp):
        self.Sigma = regex.Sigma
        self.hashed_value = hash(regex)
        self.tree_length = regex.treeLength()
        #Sanity Check
        #if self.hashed_value in CToken.token_to_regex:
        #    assert CToken.token_to_regex[self.hashed_value] == regex
        #Sanity Check
        CToken.token_to_regex[self.hashed_value] = regex
        CToken.token_to_string[self.hashed_value] = regex._strP()
    
    def __str__(self):
        return CToken.token_to_string[self.hashed_value]
        #return str(CToken.token_to_regex[self.hashed_value])

    _strP = __str__

    def __repr__(self):
        return "CToken(%d)" % self.hashed_value

    def treeLength(self):
        return self.tree_length
    
    def __copy__(self):
        return CToken(CToken.token_to_regex[self.hashed_value])
