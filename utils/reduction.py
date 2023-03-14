from FAdo.reex import *


def is_terminal(regex: RegExp) -> bool:
    return isinstance(regex, CEpsilon) or isinstance(regex, CAtom)

# return minimized regular expression and list of independent items as tuple (internally)
# independent items: CAtom, CEpsilon, and CStar


def minimize_regular_expression(regex: RegExp) -> tuple:
    # base case
    if is_terminal(regex):
        return regex, [regex]

    # recursive case
    if isinstance(regex, CStar):
        arg, arg_items = minimize_regular_expression(regex.arg)
        # Case: (R*)* = R*
        if isinstance(arg, CStar):
            return arg, [arg]
        else:
            return CStar(arg), [CStar(arg)]

    elif isinstance(regex, CDisj):
        first, first_items = minimize_regular_expression(regex.arg1)
        second, second_items = minimize_regular_expression(regex.arg2)
        # Case: R + R = R and hopefully catches (R1 + R2) + R3 = R1 + (R2 + R3)
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
        # Case: (R1 ⋅ R2) ⋅ R3 = R1 ⋅ (R2 ⋅ R3)
        return CConcat(first, second), first_items + second_items

    else:
        print('further improvement?')
        return regex, [regex]
