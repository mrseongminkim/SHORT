from FAdo.fa import NFA

from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame

from utils.fadomata import convert_nfa_to_gfa, eliminate_with_minimization

game = StateEliminationGame()

nfa = NFA()
nfa.addState("0")
nfa.addState("1")
nfa.addState("2")
nfa.addState("3")
nfa.addState("4")
nfa.setInitial([0])
nfa.setFinal([4])
nfa.addTransition(0, 'a', 1)
nfa.addTransition(1, 'b', 3)
nfa.addTransition(3, 'c', 2)
nfa.addTransition(2, 'b', 3)
nfa.addTransition(2, 'b', 1)
nfa.addTransition(2, 'a', 2)
nfa.addTransition(2, 'b', 2)
nfa.addTransition(2, 'c', 2)
nfa.addTransition(3, 'a', 4)

nfa.epsilonPaths

gfa = convert_nfa_to_gfa(nfa)
for st in [2, 1, 3]:
    eliminate_with_minimization(gfa, st, delete_state=False, tokenize=False, minimize=True)
print(gfa.delta[gfa.Initial][list(gfa.Final)[0]])
print(gfa.delta[gfa.Initial][list(gfa.Final)[0]].treeLength())