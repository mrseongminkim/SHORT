from FAdo.reex import *

class CToken(RegExp):
    #Static class variable
    token_to_regex = dict()
    threshold = 10

    def __init__(self, regex: RegExp):
        self.hashed_value = hash(regex)
        self.tree_length = regex.treeLength()
        #Sanity Check
        #if self.hashed_value in CToken.token_to_regex:
        #    assert CToken.token_to_regex[self.hashed_value] == regex
        #Sanity Check
        CToken.token_to_regex[self.hashed_value] = regex
    
    def __str__(self):
        return str(CToken.token_to_regex[self.hashed_value])
    
    _strP = __str__

    def __repr__(self):
        return repr(CToken.token_to_regex[self.hashed_value])

    def treeLength(self):
        return self.tree_length
    
    def __copy__(self):
        return CToken(CToken.token_to_regex[self.hashed_value])