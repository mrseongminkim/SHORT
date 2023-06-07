from FAdo.reex import *

from utils.CToken import *

def is_included(re1: RegExp, re2: RegExp, depth=0):
    MAX_RECURSION_DEPTH = 10
    depth += 1
    if depth > MAX_RECURSION_DEPTH:
        return 2

    if is_epsilon(re1):
        if is_epsilon(re2):
            return 0
        if isinstance(re2, CAtom):
            return 2
        elif isinstance(re2, CDisj):
            left = is_included(re1, re2.arg1, depth)
            right = is_included(re1, re2.arg2, depth)
            if left in [0, 1] or right in [0, 1]:
                return 1
            else:
                return 2
        elif isinstance(re2, CConcat):
            return 2
        elif isinstance(re2, CStar):
            return 1
        elif isinstance(re2, CToken):
            return is_included(re1, CToken.token_to_regex[re2.hashed_value], depth)
    elif is_epsilon(re2):
        return - is_included(re2, re1, depth - 1)

    if isinstance(re1, CAtom):
        if isinstance(re2, CAtom):
            return 0 if re1.val == re2.val else 2
        elif isinstance(re2, CDisj):
            left = is_included(re1, re2.arg1, depth)
            right = is_included(re1, re2.arg2, depth)
            if left in [0, 1] or right in [0, 1]:
                return 1
            else:
                return 2
        elif isinstance(re2, CConcat):
            return 2
        elif isinstance(re2, CStar):
            return is_included(re1, re2.arg, depth)
        elif isinstance(re2, CToken):
            return is_included(re1, CToken.token_to_regex[re2.hashed_value], depth)
    elif isinstance(re2, CAtom):
        return - is_included(re2, re1, depth - 1)

    if isinstance(re1, CDisj):
        if isinstance(re2, CDisj):
            x_in_a = is_included(re1.arg1, re2.arg1, depth)
            x_in_b = is_included(re1.arg1, re2.arg2, depth)
            y_in_a = is_included(re1.arg2, re2.arg1, depth)
            y_in_b = is_included(re1.arg2, re2.arg2, depth)
            if (x_in_a == 0 and y_in_b == 0) or (x_in_b == 0 and y_in_a == 0):
                return 0
            if (x_in_a == 1 or x_in_b == 1) and (y_in_a == 1 or y_in_b == 1):
                return 1
            if (x_in_a == -1 or x_in_b == -1) and (y_in_a == -1 or y_in_b == -1):
                return -1
            return 2
        elif isinstance(re2, CConcat) or isinstance(re2, CStar):
            first_argument = is_included(re1.arg1, re2, depth)
            second_argument = is_included(re1.arg2, re2, depth)
            if first_argument == 1 and second_argument ==  1:
                return 1
            if first_argument == -1 or second_argument == -1:
                return -1
            return 2
        elif isinstance(re2, CToken):
            if hash(re1) == re2.hashed_value:
                return 0
            first_argument = is_included(re1.arg1, CToken.token_to_regex[re2.hashed_value], depth)
            second_argument = is_included(re1.arg2, CToken.token_to_regex[re2.hashed_value], depth)
            if first_argument == 1 and second_argument ==  1:
                return 1
            if first_argument == -1 or second_argument == -1:
                return -1
            return 2
    elif isinstance(re2, CDisj):
        return - is_included(re2, re1, depth - 1)

    if isinstance(re1, CConcat):
        if isinstance(re2, CConcat):
            if re1 == re2:
                return 0
            return 2
        if isinstance(re2, CStar):
            argument = is_included(re1, re2.arg, depth)
            if argument in [0, 1]:
                return 1
            return 2
        if isinstance(re2, CToken):
            if hash(re1) == re2.hashed_value:
                return 0
            argument = is_included(re2, CToken.token_to_regex[re2.hashed_value], depth)
            if argument == 1:
                return 1
            return 2
    elif isinstance(re2, CConcat):
        return - is_included(re2, re1, depth - 1)

    if isinstance(re1, CStar):
        if isinstance(re2, CStar):
            return is_included(re1.arg, re2.arg, depth)
        if isinstance(re2, CToken):
            if hash(re1) == re2.hashed_value:
                return 0
            return is_included(re1.arg, CToken.token_to_regex[re2.hashed_value], depth)
    elif isinstance(re2, CStar):
        return - is_included(re2, re1, depth - 1)
    
    if isinstance(re1, CToken):
        return is_included(CToken.token_to_regex[re1.hashed_value], CToken.token_to_regex[re2.hashed_value], depth)
    else:
        return - is_included(re2, re1, depth - 1)


def is_epsilon(regex: RegExp):
    return isinstance(regex, CEpsilon)
