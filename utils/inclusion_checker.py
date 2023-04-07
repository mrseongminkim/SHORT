from FAdo.reex import *

from utils.CToken import *

def is_included(re1: RegExp, re2: RegExp, depth=0):
    MAX_RECURSION_DEPTH = 5
    depth += 1
    if depth > MAX_RECURSION_DEPTH:
        return 2
    if is_epsilon(re1):
        #re1 == re2로 최적화 가능
        if is_epsilon(re2):
            return 0
        #CAtom은 epsilon 이외의 알파벳만 저장하니 포함하지 않는다.
        if isinstance(re2, CAtom):
            return 2
        #epsilon in CDisj이거나 CDisj가 epsilon을 가질 수 있다. -> 0 or 1
        #epsilon이 다른 regex를 포함하는 일은 없다.
        elif isinstance(re2, CDisj):
            left = is_included(re1, re2.arg1, depth)
            right = is_included(re1, re2.arg2, depth)
            if left in [0, 1] or right in [0, 1]:
                return 1
            else:
                return 2
        #CConcat은 (minimization이 잘 되었다는 가정하에) epsilon을 포함하지 못한다.
        elif isinstance(re2, CConcat):
            return 2
        #regex의 star는 항상 epsilon을 포함한다.
        elif isinstance(re2, CStar):
            return 1
        #trivial
        elif isinstance(re2, CToken):
            return is_included(re1, CToken.token_to_regex[re2.hashed_value], depth)
    #순서를 바꿔서 호출하기 위함이니 depth를 증가시키지 않는다.
    elif is_epsilon(re2):
        return - is_included(re2, re1, depth - 1)
    #이제 모든 경우의 수에서 epsilon을 제외한다.

    if isinstance(re1, CAtom):
        #re1 == re2로 최적화 가능
        if isinstance(re2, CAtom):
            return 0 if re1.val == re2.val else 2
        #atom in CDisj이거나 CDisj가 atom을 가질 수 있다. -> 0 or 1
        #atom이 다른 regex를 포함하는 일은 없다.
        #epsilon과 CDisj 관계와 동일하다.
        elif isinstance(re2, CDisj):
            left = is_included(re1, re2.arg1, depth)
            right = is_included(re1, re2.arg2, depth)
            if left in [0, 1] or right in [0, 1]:
                return 1
            else:
                return 2
        #CConcat은 (minimization이 잘 되었다는 가정하에) atom을 포함하지 못한다.
        #epsilon과 Concat 관계와 동일하다.
        elif isinstance(re2, CConcat):
            return 2
        #CStar의 arg가 re1을 포함하는지를 보면 된다.
        #1 아니면 2 | -2가 나온다.
        elif isinstance(re2, CStar):
            return is_included(re1, re2.arg, depth)
        #trivial
        elif isinstance(re2, CToken):
            return is_included(re1, CToken.token_to_regex[re2.hashed_value], depth)
    elif isinstance(re2, CAtom):
        return - is_included(re2, re1, depth - 1)
    #이제 모든 경우의 수에서 CAtom을 제외한다.

    #제일 헷갈리는 부분
    #여기서부터 -1이 나올 수 있다.
    #minimization이 잘 되었다면 x != y이고 a != b이다.
    #re1 = (x + y)
    #re2 = (a + b)
    if isinstance(re1, CDisj):
        if isinstance(re2, CDisj):
            #new_depth = depth/4?
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
        #같은 경우가 존재한다.
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
    #이제 모든 경우의 수에서 CDisj을 제외한다.

    #Concat은 inclusion 되는 경우가 많이 없을 듯 하다.
    #a와 ab가 있을 때, a는 ab에 include 되어 있는 것인가?
    #아니라고 생각하고 짠 코드.
    if isinstance(re1, CConcat):
        if isinstance(re2, CConcat):
            if re1 == re2:
                return 0
            return 2
        if isinstance(re2, CStar):
            argument = is_included(re1, re2.arg, depth)
            if argument in [0, 1]:
                return 1
            #-1은 일어나지 않을 것이다.
            return 2
        if isinstance(re2, CToken):
            if hash(re1) == re2.hashed_value:
                return 0
            #-1은 일어나지 않을 것이다.
            argument = is_included(re2, CToken.token_to_regex[re2.hashed_value], depth)
            if argument == 1:
                return 1
            return 2
    elif isinstance(re2, CConcat):
        return - is_included(re2, re1, depth - 1)
    #이제 모든 경우의 수에서 CConcat을 제외한다.

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
