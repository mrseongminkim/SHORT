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
        #Case: R + R = R and hopefully catches (R1 + R2) + R3 = R1 + (R2 + R3)
        if set(first_items) == set(second_items):
            return first, first_items
        else:
            if isinstance(first, CConcat):
                first_items = [first]
            if isinstance(second, CConcat):
                second_items = [second]
            return CDisj(first, second), first_items + second_items

    elif isinstance(regex, CConcat):
        first, first_items = minimize_regular_expression(regex.arg1)
        second, second_items = minimize_regular_expression(regex.arg2)
        print(first_items, second_items)
        #Case: (R1 ⋅ R2) ⋅ R3 = R1 ⋅ (R2 ⋅ R3)
        #Can't distinguish between 0 + 1 and 01
        #Error - fix it
        return CConcat(first, second), first_items + second_items

    else:
        print('further improvement?')
        return regex, [regex]

def main():
    result = str(CStar(CStar(CDisj(CConcat(CAtom(0), CConcat(CAtom(1), CAtom(2))), CConcat(CConcat(CAtom(0), CAtom(1)), CAtom(2))))))
    if isinstance(result, RegExp):
        print('not string')
    elif isinstance(result, str):
        print('string')
    print(result)

    result = '(0 1 2 + 4)*'
    x = str2regexp(result)
    print(x)
    print(repr(x))
    exit()
    print(result)
    print(repr(result))
    print()
    result = minimize_regular_expression(result)[0]
    print(result)
    print(repr(result))


if __name__ == '__main__':
    main()