from utils.fadomata import *

regex = "0 (0 + 1)"
hi = str2regexp(regex)
hi2 = deepcopy(hi)
print("", regex, "\n", hi)
test = CToken(hi)
test2 = CToken(hi2)