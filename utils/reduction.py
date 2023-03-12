from FAdo.reex import *

def is_terminal(regex: RegExp) -> bool:
    return isinstance(regex, CEpsilon) or isinstance(regex, CAtom)

#return minimized regular expression and list of independent items as tuple (internally)
#independent items: CAtom, CEpsilon, and CStar
def minimize_regular_expression(regex: RegExp) -> tuple:
    #base case
    if is_terminal(regex):
        return regex, [regex]

    #recursive case
    if isinstance(regex, CStar):
        arg, arg_items = minimize_regular_expression(regex.arg)
        #Case: (R*)* = R*
        if isinstance(arg, CStar):
            return arg, [arg]
        else:
            return CStar(arg), [CStar(arg)]

    elif isinstance(regex, CDisj):
        first, first_items = minimize_regular_expression(regex.arg1)
        second, second_items = minimize_regular_expression(regex.arg2)

        if isinstance(first, CConcat):
            new_first = first_items[0]
            for i in range(1, len(first_items)):
                new_first = CConcat(new_first, first_items[i])
            first = new_first
            first_items = [first]
        
        if isinstance(second, CConcat):
            new_second = second_items[0]
            for i in range(1, len(second_items)):
                new_second = CConcat(new_second, second_items[i])
            second = new_second
            second_items = [second]

        #Case: R + R = R and hopefully catches (R1 + R2) + R3 = R1 + (R2 + R3)
        if set(first_items) == set(second_items):
            return first, first_items
        else:
            first_items += second_items
            return CDisj(first, second), first_items

    elif isinstance(regex, CConcat):
        first, first_items = minimize_regular_expression(regex.arg1)
        second, second_items = minimize_regular_expression(regex.arg2)

        if isinstance(first, CDisj):
            first_items = [first]
        if isinstance(second, CDisj):
            second_items = [second]

        return CConcat(first, second), first_items + second_items

    else:
        print('further improvement?')
        return regex, [regex]

def rpn(regex: RegExp) -> str:
    if isinstance(regex, CAtom):
        return regex.val
    if isinstance(regex, CEpsilon):
        return Epsilon
    if isinstance(regex, CDisj):
        return rpn(regex.arg1) + rpn(regex.arg2) + '+'
    if isinstance(regex, CStar):
        return rpn(regex.arg) + '*'
    if isinstance(regex, CConcat):
        return rpn(regex.arg1) + rpn(regex.arg2) + '.'

def concat_list(waiting_list: list) -> RegExp:
    if isinstance(waiting_list[0], RegExp):
        result = waiting_list[0]
    else:
        result = CAtom(waiting_list[0])
    for i in range(1, len(waiting_list)):
        if not isinstance(waiting_list[i], RegExp):
            waiting_list[i] = CAtom(waiting_list[i])
        result = CConcat(result, waiting_list[i])
    return result

def rpn_to_str(rpn: str) -> RegExp:
    stack = []
    for c in rpn:
        if c == '.':
            arg2 = stack.pop()
            arg1 = stack.pop()
            if type(arg1) == list:
                arg1 += arg2
                stack.append(arg1)
            elif type(arg2) == list:
                arg2 = [arg1] + arg2
                stack.append(arg2)
            else:
                stack.append([arg1, arg2])
        elif c == '+':
            arg2 = stack.pop()
            arg1 = stack.pop()
            if type(arg1) == list:
                arg1 = concat_list(arg1)
            if type(arg2) == list:
                arg2 = concat_list(arg2)
            stack.append(CDisj(arg1, arg2))
        elif c == '*':
            arg1 = stack.pop()
            if type(arg1) == list:
                arg1 = concat_list(arg1)
            stack.append(CStar(arg1))
        else:
            stack.append(c)
    return stack.pop()

def main():
    #result = '0 (1 2) + 3'
    #result = '(0 1) + (0 + 1)'
    #'((0 (1 2)) + ((0 1) 2))**'
    #'(0 (1 (2 3))) + ((0 1)(2 3))'
    #result = '(0 (1 2))'
    #result = '(0 1) 2'
    #result = '0 (1 + 2) 3'
    #result = '(0 (1 + 2)) 3'
    #result = '(0 ((1 + 2) 3)) + ((0 (1 + 2)) 3)'
    result = '0 1 2 + 0 (1 2)'
    result = str2regexp(result)
    print(result)
    print(repr(result))
    print('before/after')
    result = rpn(result)
    print(result)
    result = rpn_to_str(result)
    print(result)
    print(repr(result))
    exit()

    #((0 (1 2)) + ((0 1) 2))**
    #result = CStar(CStar(CDisj(CConcat(CAtom(0), CConcat(CAtom(1), CAtom(2))), CConcat(CConcat(CAtom(0), CAtom(1)), CAtom(2)))))
    print(result)
    print(repr(result))
    print('before/after')
    result, temp = minimize_regular_expression(result)
    print(result)
    print(repr(result))
    print(temp)


if __name__ == '__main__':
    main()