# -*- coding: utf-8 -*-
"""**Conversions between objects.**

Deterministic and non-deterministic automata manipulation, conversion and evaluation.
.. *Authors:* Rogério Reis & Nelma Moreira
.. *This is part of FAdo project*   https://fado.dcc.fc.up.pt.

.. *Copyright:* 1999-2020 Rogério Reis & Nelma Moreira {rvr,nam}@dcc.fc.up.pt

.. This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as published
   by the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   675 Mass Ave, Cambridge, MA 02139, USA."""
from copy import *
from .fa import NFA, OFA, DFA
from .common import *
from . import reex
import itertools


def FA2GFA(aut):
    """ Creates a GFA equivalent to NFA

    :arg aut: the automaton
    :type aut: OFA
    :returns: a GFA deep copy
    :rtype: GFA """
    gfa = GFA()
    gfa.setSigma(aut.Sigma)
    if isinstance(aut, NFA):
        # this should be optimized
        fa = aut._toNFASingleInitial()
        gfa.Initial = uSet(fa.Initial)
        gfa.States = fa.States[:]
        gfa.setFinal(fa.Final)
        gfa.predecessors = {}
        for i in range(len(gfa.States)):
            gfa.predecessors[i] = set([])
        for s in fa.delta:
            for c in fa.delta[s]:
                for s1 in fa.delta[s][c]:
                    gfa.addTransition(s, c, s1)
        return gfa
    elif isinstance(aut, DFA):
        gfa.States = aut.States[:]
        gfa.setInitial(aut.Initial)
        gfa.setFinal(aut.Final)
        gfa.predecessors = {}
        for i in range(len(gfa.States)):
            gfa.predecessors[i] = set([])
        for s in aut.delta:
            for c in aut.delta[s]:
                gfa.addTransition(s, c, aut.delta[s][c])
        return gfa
    else:
        raise TypeError()


def FAallRegExps(aut):
    """Evaluates the alphabetic length of the equivalent regular expression using every possible order of state
    elimination.

    :arg aut: the automaton
    :type aut: OFA
    :rtype: list of tuples (int, list of states)"""
    new = aut.dup()
    new.trim()
    gfa = FA2GFA(new)
    for order in itertools.permutations(list(range(len(gfa.States)))):
        return FA2regexpSEO(aut, copy(list(order))).alphabeticLength(), order


def cutPoints(aut):
    """Set of FA's cut points

    :arg aut: the automaton
    :type aut: OFA
    :returns: set of states
    :rtype: set of int"""
    gfa = FA2GFA(aut)
    gfa.normalize()
    # make gfa a graph instead of a digraph
    new_edges = []
    for a in gfa.delta:
        for b in gfa.delta[a]:
            new_edges.append((a, b))
    for i in new_edges:
        if i[1] not in gfa.delta:
            gfa.delta[i[1]] = {}
        else:
            gfa.delta[i[1]][i[0]] = 'x'
    for i in new_edges:
        if i[0] not in gfa.delta[i[1]]:
            gfa.delta[i[1]][i[0]] = 'x'
    # initializations needed for cut point detection
    gfa.c = 1
    gfa.num = {}
    gfa.visited = []
    gfa.parent = {}
    gfa.low = {}
    gfa.cuts = set([])
    gfa.assignNum(gfa.Initial)
    gfa.assignLow(gfa.Initial)
    # initial state is never a cut point, so it should be removed
    gfa.cuts.remove(gfa.Initial)
    cutpoints = copy(gfa.cuts) - gfa.Final
    # remove self-loops and check if the cut points are in a loop
    gfa = FA2GFA(aut)
    gfa.normalize()
    for i in gfa.delta:
        if i in gfa.delta[i]:
            del gfa.delta[i][i]
    cycles = gfa.evalNumberOfStateCycles()
    for i in cycles:
        if cycles[i] != 0 and i in cutpoints:
            cutpoints.remove(i)
    return cutpoints


def _commonCode1(gfa):
    if len(gfa.Final) > 1:
        last = gfa.addState("Last")
        for s in gfa.Final:
            gfa.addTransition(s, Epsilon, last)
        gfa.setFinal([last])
    else:
        last = list(gfa.Final)[0]
    foo = {}
    lfoo = len(gfa.States) - 1
    foo[lfoo], foo[last] = last, lfoo
    gfa.reorder(foo)
    if lfoo != gfa.Initial:
        n = 2
        foo = {lfoo - 1: gfa.Initial, gfa.Initial: lfoo - 1}
        gfa.reorder(foo)
    else:
        n = 1
    return gfa, n


def _commonCode2(gfa, aut, n):
    gfa.completeDelta()
    if n == 1:
        return reex.CStar(gfa.delta[gfa.Initial][gfa.Initial], copy(aut.Sigma)).reduced()
    ii = gfa.Initial
    fi = list(gfa.Final)[0]
    a = gfa.delta[ii][ii]
    b = gfa.delta[ii][fi]
    c = gfa.delta[fi][ii]
    d = gfa.delta[fi][fi]
    # bd*
    re1 = reex.CConcat(b, reex.CStar(d, copy(aut.Sigma)), copy(aut.Sigma))
    # a + bd*c
    re2 = reex.CDisj(a, reex.CConcat(re1, c, copy(aut.Sigma)), copy(aut.Sigma))
    # (a + bd*c)* bd*
    return reex.CConcat(reex.CStar(re2, copy(aut.Sigma)), re1, copy(aut.Sigma)).reduced()


def FA2regexpSE(aut):
    """A regular expression obtained by state elimination algorithm whose language is recognised by the FA aut.

    :arg aut: the automaton
    :type aut: OFA
    :return: the equivalent regular expression
    :rtype: reex.RegExp"""
    new = aut.dup()
    new.trim()
    if not len(new.States):
        return reex.CEmptySet(copy(aut.Sigma))
    if not len(new.Final):
        return reex.CEmptySet(copy(aut.Sigma))
    if len(new.States) == 1 and len(new.delta) == 0:
        return reex.CEpsilon(copy(aut.Sigma))
    elif type(new) == NFA and len(new.Initial) != 0 and len(new.delta) == 0:
        return reex.CEpsilon(copy(aut.Sigma))
    gfa, n = _commonCode1(FA2GFA(new))
    lr = list(range(len(gfa.States) - n))
    gfa.eliminateAll(lr)
    return _commonCode2(gfa, aut, n)


def SP2regexp(aut):
    """ Checks if FA is SP (Serial-PArallel), and if so returns the regular expression whose language is
    recognised by the FA

    :arg aut: the automaton
    :type aut: OFA
    :returns: equivalent regular expression
    :rtype: reex.RegExp
    :raises NotSP: if the automaton is not Serial-Parallel

    .. seealso:: Moreira & Reis, Fundamenta Informatica, Series-Parallel automata and short regular expressions,
       n.91 3-4, pag 611-629.
       https://www.dcc.fc.up.pt/~nam/publica/spa07.pdf

    .. note::
       Automata must be Serial-Parallel"""
    v = 0  # just to satisfy the checker
    gfa = FA2GFA(aut)
    gfa.lab = {}
    gfa.out_index = {}
    for i in range(len(gfa.States)):
        if i not in gfa.delta:
            gfa.out_index[i] = 0
        else:
            gfa.out_index[i] = len(gfa.delta[i])
    topo_order = gfa.topoSort()
    for v in topo_order:  # States should be topologically ordered
        i = len(gfa.predecessors[v])
        while i > 1:
            # noinspection PyProtectedMember
            i = gfa._simplify(v, i)
        if len(gfa.predecessors[v]):
            track = gfa.lab[(list(gfa.predecessors[v])[0], v)]
            rp = gfa.delta[list(gfa.predecessors[v])[0]][v]
        else:
            track = SPLabel([])
            rp = reex.CEpsilon(copy(aut.Sigma))
        try:
            # noinspection PyProtectedMember
            gfa._do_edges(v, track, rp)
        except KeyError:
            pass
    return gfa.delta[list(gfa.predecessors[v])[0]][v]


def FAeliminateSingles(aut):
    """Eliminates every state that only have one successor and one predecessor.

    :arg aut: the automaton
    :type aut: OFA
    :returns: GFA after eliminating states
    :rtype: GFA """
    # DFS to obtain {v:(e, s)} -> convert from {v:(e, s)} to {(e, s):v} -> eliminate all {(1, 1):v}
    gfa = FA2GFA(aut)
    io = {}
    for i in range(len(aut.States)):
        io[i] = [0, 0]
    gfa.DFS(io)
    new = {}
    for i in io:
        if (io[i][0], io[i][1]) in new:
            new[io[i]].append(i)
        else:
            new[io[i]] = [i]
    if (1, 1) not in new:
        return gfa
        # While there are singles, delete them
    while new[(1, 1)]:
        v = new[(1, 1)].pop()
        i = list(gfa.predecessors[v])[0]
        o = list(gfa.delta[v].items())[0][0]
        if o in gfa.delta[i]:
            gfa.delta[i][o] = reex.CDisj(reex.CConcat(gfa.delta[i][v], gfa.delta[v][o], copy(aut.Sigma)),
                                         gfa.delta[i][o])
            new[io[i]].remove(i)
            new[io[o]].remove(o)
            # lists are unhashable
            e0, e1 = io[i]
            io[i] = (e0, e1 - 1)
            e0, e1 = io[o]
            io[o] = (e0 - 1, e1)
            if io[i] in new:
                new[io[i]].append(i)
            else:
                new[io[i]] = [i]
            if io[o] in new:
                new[io[o]].append(o)
            else:
                new[io[o]] = [o]
            gfa.predecessors[o].remove(v)
        else:
            gfa.delta[i][o] = reex.CConcat(gfa.delta[i][v], gfa.delta[v][o], copy(aut.Sigma))
            gfa.predecessors[o].remove(v)
            gfa.predecessors[o].add(i)
        del gfa.delta[i][v]
        del gfa.delta[v][o]
        del gfa.delta[v]
        del gfa.predecessors[v]
        del io[v]
        # Clean up state indexes...
    new_order = {}
    ind = 0
    for i in gfa.delta:
        if i not in new_order:
            new_order[i] = ind
        a = 0
        for j in gfa.delta[i]:
            if j not in new_order:
                a += 1
                new_order[j] = ind + a
        ind += a
    gfa.reorder(new_order)
    gfa.States = gfa.States[:ind + 1]
    return gfa


def FA2regexpCG(aut):
    """Regular expression from state elimination whose language is recognised by the FA. Uses a heuristic to choose
    the order of elimination.

    :arg aut: the automaton
    :type aut: OFA
    :returns: the equivalent regular expression
    :rtype: reex.RegExp"""
    new = aut.dup()
    new.trim()
    gfa = FA2GFA(new)
    if not len(gfa.Final):
        return reex.CEmptySet(copy(aut.Sigma))
    gfa.normalize()
    weights = {}
    for st in range(len(gfa.States)):
        if st != gfa.Initial and st not in gfa.Final:
            weights[st] = gfa.weight(st)
    for i in range(len(gfa.States) - 2):
        m = [(v, u) for (u, v) in list(weights.items())]
        m = min(m)
        m = m[1]
        # After 'm' is eliminated its adjacencies might
        # change their indexes...
        adj = set([])
        for st in gfa.predecessors[m]:
            if st > m:
                adj.add(st - 1)
            else:
                adj.add(st)
        for st in gfa.delta[m]:
            if st > m:
                adj.add(st - 1)
            else:
                adj.add(st)
        gfa.eliminateState(m)
        for st in weights:
            if st > m:
                weights[st - 1] = weights[st]
        for st in adj:
            if st != gfa.Initial and st not in gfa.Final:
                weights[st] = gfa.weight(st)
        del weights[len(gfa.States) - 2]
    return gfa.delta[gfa.Initial][list(gfa.Final)[0]].reduced()


def FA2regexpCG_nn(aut: OFA):
    """Regular expression from state elimination whose language is recognised by the FA. Uses a heuristic to choose
    the order of elimination. The FA is not normalized before the state elimination.

    :arg aut: the automaton
    :type aut: OFA
    :returns: the equivalent regular expression
    :rtype: reex.RegExp"""
    if not len(aut.Final):
        return reex.CEmptySet(copy(aut.Sigma))
    new = aut.dup()
    new.trim()
    gfa, n = _commonCode1(FA2GFA(new))
    weights = {}
    for st in range(len(gfa.States)):
        if st != gfa.Initial and st not in gfa.Final:
            weights[st] = gfa.weight(st)
    for i in range(len(gfa.States) - n):
        m = [(v, u) for (u, v) in list(weights.items())]
        m = min(m)
        m = m[1]
        succs = set([])
        for a in gfa.delta[m]:
            if a != m:
                succs.add(a)
        preds = set([])
        for a in gfa.predecessors[m]:
            if a != m:
                preds.add(a)
        gfa.eliminate(m)
        # update predecessors for weight(st)...
        for s in succs:
            gfa.predecessors[s].remove(m)
            for s1 in preds:
                gfa.predecessors[s].add(s1)
        del gfa.predecessors[m]
        for s in set(list(succs) + list(preds)):
            if s != gfa.Initial and s not in gfa.Final:
                weights[s] = gfa.weight(s)
        del weights[m]
    gfa.completeDelta()
    if n == 1:
        return reex.CStar(gfa.delta[gfa.Initial][gfa.Initial], copy(aut.Sigma)).reduced()
    # noinspection PyProtectedMember
    return gfa._re0()


def FA2regexpSEO(aut, order=None):
    """Regular expression from state elimination whose language is recognised by the FA. The FA is normalized before
    the state elimination.

    :arg aut: the automaton
    :type aut: OFA
    :param list order: state elimination sequence
    :returns: the equivalent regular expression
    :rtype: reex.RegExp"""
    if not order:
        order = []
    new = aut.dup()
    new.trim()
    gfa = FA2GFA(new)
    if order is None:
        order = list(range(len(gfa.States)))
    if not len(gfa.Final):
        return reex.CEmptySet(copy(aut.Sigma))
    gfa.normalize()
    while order:
        st = order.pop(0)
        for i in range(len(order)):
            if order[i] > st:
                order[i] -= 1
        gfa.eliminateState(st)
    return gfa.delta[gfa.Initial][list(gfa.Final)[0]]


def FA2regexpDynamicCycleHeuristic(aut):
    """ State elimination Heuristic based on the number of cycles that passes through each state. Here those
    numbers are evaluated dynamically after each elimination step

    :arg aut: the automaton
    :type aut: OFA
    :returns: an equivalent regular expression
    :rtype: reex.RegExp

    .. seealso::
       Nelma Moreira, Davide Nabais, and Rogério Reis. State elimination ordering strategies: Some experimental
       results. Proc. of 11th Workshop on Descriptional Complexity of Formal Systems (DCFS10),
       pages 169-180.2010. DOI: 10.4204/EPTCS.31.16"""
    if not len(aut.Final):
        return reex.CEmptySet(copy(aut.Sigma))
    new = aut.dup()
    new.trim()
    gfa = FA2GFA(new)
    cycles = gfa.evalNumberOfStateCycles()
    gfa, n = _commonCode1(gfa)
    weights = {}
    for st in range(len(gfa.States)):
        if st != gfa.Initial and st not in gfa.Final:
            weights[st] = gfa.weightWithCycles(st, cycles)
    for i in range(len(gfa.States) - n):
        m = [(v, u) for (u, v) in list(weights.items())]
        m = min(m)
        m = m[1]
        succs = set([])
        for a in gfa.delta[m]:
            if a != m:
                succs.add(a)
        preds = set([])
        for a in gfa.predecessors[m]:
            if a != m:
                preds.add(a)
        gfa.eliminate(m)
        cycles = gfa.evalNumberOfStateCycles()
        # update predecessors for weight(st)...
        for s in succs:
            gfa.predecessors[s].remove(m)
            for s1 in preds:
                gfa.predecessors[s].add(s1)
        del gfa.predecessors[m]
        for s in set(list(succs) + list(preds)):
            if s != gfa.Initial and s not in gfa.Final:
                weights[s] = gfa.weightWithCycles(s, cycles)
        del weights[m]
    gfa.completeDelta()
    if n == 1:
        return reex.CStar(gfa.delta[gfa.Initial][gfa.Initial], copy(aut.Sigma))
    # noinspection PyProtectedMember
    return gfa._re0()


def FA2regexpStaticCycleHeuristic(aut):
    """State elimination Heuristic based on the number of cycles that passes through each state. Here those
    numbers are evaluated statically in the beginning of the process

    :arg aut: the automaton
    :type aut: OFA
    :returns: a equivalent regular expression
    :rtype: reex.RegExp

    .. seealso::
       Nelma Moreira, Davide Nabais, and Rogério Reis. State elimination ordering strategies: Some experimental
       results. Proc. of 11th Workshop on Descriptional Complexity of Formal Systems (DCFS10),
       pages 169-180.2010. DOI: 10.4204/EPTCS.31.16"""
    if not len(aut.Final):
        return reex.CEmptySet(copy(aut.Sigma))
    new = aut.dup()
    new.trim()
    cycles = new.evalNumberOfStateCycles()
    gfa, n = _commonCode1(new)
    weights = {}
    for st in range(len(gfa.States)):
        if st != gfa.Initial and st not in gfa.Final:
            weights[st] = gfa.weightWithCycles(st, cycles)
    for i in range(len(gfa.States) - n):
        m = [(v, u) for (u, v) in list(weights.items())]
        m = min(m)
        m = m[1]
        succs = set([])
        for a in gfa.delta[m]:
            if a != m:
                succs.add(a)
        preds = set([])
        for a in gfa.predecessors[m]:
            if a != m:
                preds.add(a)
        gfa.eliminate(m)
        for s in succs:
            gfa.predecessors[s].remove(m)
            for s1 in preds:
                gfa.predecessors[s].add(s1)
        del gfa.predecessors[m]
        for s in set(list(succs) + list(preds)):
            if s != gfa.Initial and s not in gfa.Final:
                weights[s] = gfa.weightWithCycles(s, cycles)
        del weights[m]
    gfa.completeDelta()
    if n == 1:
        return reex.CStar(gfa.delta[gfa.Initial][gfa.Initial], copy(aut.Sigma))
    # noinspection PyProtectedMember
    return gfa._re0()


def FA2regexpSE_nn(aut, order=None):
    """Regular expression from state elimination whose language is recognised by the FA. The FA is not normalized
    before the state elimination.

    :arg aut: the automaton
    :type aut: OFA
    :param list order: state elimination sequence
    :returns: the equivalent regular expression
    :rtype: reex.RegExp"""
    n = 0  # just to satisfy the checker
    if not order:
        order = []
    gfa = FA2GFA(aut)
    if not len(gfa.Final):
        return reex.CEmptySet(copy(aut.Sigma))
    if order is None:
        if len(gfa.Final) > 1:
            last = gfa.addState("Last")
            gfa.predecessors[last] = set([])
            for s in gfa.Final:
                gfa.addTransition(s, Epsilon, last)
                gfa.predecessors[last].add(s)
            gfa.setFinal([last])
        else:
            last = list(gfa.Final)[0]
        foo = {}
        lfoo = len(gfa.States) - 1
        foo[lfoo], foo[last] = last, lfoo
        gfa.reorder(foo)
        if lfoo != gfa.Initial:
            n = 2
            foo = {lfoo - 1: gfa.Initial, gfa.Initial: lfoo - 1}
            gfa.reorder(foo)
        else:
            n = 1
        order = list(range(len(gfa.States) - n))
    while order:
        st = order.pop(0)
        for i in range(len(order)):
            if order[i] > st:
                order[i] -= 1
        gfa.eliminateState(st)
    return _commonCode2(gfa, aut, n)


def DFA2regexpDijkstra(aut) -> reex.RegExp:
    """Returns a regexp for the current DFA considering the recursive method. Very inefficent.

    :arg aut: the automaton
    :type aut: DFA
    :returns: a regexp equivalent to the current DFA
    :rtype: reex.RegExp"""
    if aut.Initial:
        foo = {0: aut.Initial, aut.Initial: 0}
        aut.reorder(foo)
    n, nstates = len(aut.Final), len(aut.States) - 1
    if not n:
        return reex.CEmptySet(copy(aut.Sigma))
    r = _RPath(aut, 0, uSet(aut.Final), nstates)
    for s in list(aut.Final)[1:]:
        r = reex.CDisj(_RPath(aut, 0, s, nstates), r, copy(aut.Sigma))
    return r


def _RPath(aut, initial, final, m):
    """Recursive path. (Dijsktra algorithm) The recursive function that plays a central role in the creation of
    the RE from a DFA. This suppose that there are no disconnected states."""
    if m == -1:
        if initial == final:
            r = reex.CEpsilon(copy(aut.Sigma))
            try:
                for c in aut.delta[initial]:
                    if aut.delta[initial][c] == initial:
                        r = reex.CDisj(r,
                                       reex.CAtom(c, copy(aut.Sigma)),
                                       copy(aut.Sigma))
            except KeyError:
                pass
            return r.reduced()
        else:
            r = reex.CEmptySet(copy(aut.Sigma))
            try:
                for c in aut.delta[initial]:
                    if aut.delta[initial][c] == final:
                        if not r.emptysetP():
                            r = reex.CDisj(r, reex.CAtom(c, copy(aut.Sigma)))
                        else:
                            r = reex.CAtom(c, copy(aut.Sigma))
            except KeyError:
                pass
            return r.reduced()
    else:
        r = reex.CDisj(_RPath(aut, initial, final, m - 1),
                       reex.CConcat(_RPath(aut, initial, m, m - 1),
                                    reex.CConcat(reex.CStar(_RPath(aut, m, m, m - 1),
                                                            copy(aut.Sigma)),
                                                 _RPath(aut, m, final, m - 1),
                                                 copy(aut.Sigma)),
                                    copy(aut.Sigma)), copy(aut.Sigma))
    return r.reduced()


class GFA(OFA):
    """ Class for Generalized Finite Automata: NFA with a unique initial state and transitions are labeled with RegExp.

    .. inheritance-diagram:: GFA"""

    def toNFA(self):
        raise FAdoNotImplemented

    def _s_lstInitial(self):
        raise FAdoNotImplemented

    def _lstTransitions(self):
        raise FAdoNotImplemented

    def transitions(self):
        raise FAdoNotImplemented

    def transitionsA(self):
        raise FAdoNotImplemented

    def _deleteRefInDelta(self, j, sm, s):
        raise FAdoNotImplemented

    def _deleteRefInitial(self, s):
        raise FAdoNotImplemented

    def star(self, _):
        raise FAdoNotImplemented

    def __or__(self, _):
        raise FAdoNotImplemented

    def __and__(self, _):
        raise FAdoNotImplemented

    def reverseTransitions(self, _):
        raise FAdoNotImplemented

    def finalCompP(self, s):
        raise NImplemented()

    def evalSymbol(self, stil, sym):
        raise NImplemented()

    def __eq__(self, other):
        raise NImplemented()

    def deleteStates(self, del_states):
        raise NImplemented()

    def initialComp(self):
        raise NImplemented()

    def _getTags(self):
        raise NImplemented()

    def __ne__(self, other):
        raise NImplemented()

    def succintTransitions(self):
        raise NImplemented()

    def usefulStates(self):
        raise NImplemented()

    def uniqueRepr(self):
        raise NImplemented()

    def __init__(self):
        super(GFA, self).__init__()
        self.predecessors = None

    def __repr__(self):
        """GFA string representation
        :rtype: str"""
        return 'GFA({0:>s})'.format(self.__str__())

    def addTransition(self, sti1, sym, sti2):
        """Adds a new transition from ``sti1`` to ``sti2`` consuming symbol ``sym``. Label of the transition function
         is a RegExp.

        :param int sti1: state index of departure
        :param int sti2: state index of arrival
        :param str sym: symbol consumed
        :raises DFAepsilonRedefenition: if sym is Epsilon"""
        try:
            self.addSigma(sym)
            sym = reex.CAtom(sym, copy(self.Sigma))
        except DFAepsilonRedefinition:
            sym = reex.CEpsilon(copy(self.Sigma))
        if sti1 not in self.delta:
            self.delta[sti1] = {}
        if sti2 not in self.delta[sti1]:
            self.delta[sti1][sti2] = sym
        else:
            self.delta[sti1][sti2] = reex.CDisj(self.delta[sti1][sti2], sym, copy(self.Sigma))
        # TODO: write cleaner code and get rid of the general catch
        # noinspection PyBroadException
        try:
            self.predecessors[sti2].add(sti1)
        except KeyError:
            pass

    def reorder(self, dictio):
        """Reorder states indexes according to given dictionary.

        :param dict dictio: order

        .. note::
           dictionary does not have to be complete"""
        if len(list(dictio.keys())) != len(self.States):
            for i in range(len(self.States)):
                if i not in dictio:
                    dictio[i] = i
        delta = {}
        preds = {}
        for s in self.delta:
            delta[dictio[s]] = {}
            if dictio[s] not in preds:
                preds[dictio[s]] = set([])
            for s1 in self.delta[s]:
                delta[dictio[s]][dictio[s1]] = self.delta[s][s1]
                if dictio[s1] in preds:
                    preds[dictio[s1]].add(dictio[s])
                else:
                    preds[dictio[s1]] = {dictio[s]}
        self.delta = delta
        self.predecessors = preds

        self.Initial = dictio[self.Initial]
        Final = set()
        for i in self.Final:
            Final.add(dictio[i])
        self.Final = Final
        states = list(range(len(self.States)))
        for i in range(len(self.States)):
            states[dictio[i]] = self.States[i]
        self.States = states

    def eliminate(self, st):
        """Eliminate a state.

        :param int st: state to be eliminated"""
        if st in self.delta and st in self.delta[st]:
            r2 = copy(reex.CStar(self.delta[st][st], copy(self.Sigma)))
            del self.delta[st][st]
        else:
            r2 = None
        for s in self.delta:
            if st not in self.delta[s]:
                continue
            r1 = copy(self.delta[s][st])
            del self.delta[s][st]
            for s1 in self.delta[st]:
                r3 = copy(self.delta[st][s1])
                if r2 is not None:
                    r = reex.CConcat(r1, reex.CConcat(r2, r3, copy(self.Sigma)), copy(self.Sigma))
                else:
                    r = reex.CConcat(r1, r3, copy(self.Sigma))
                if s1 in self.delta[s]:
                    self.delta[s][s1] = reex.CDisj(self.delta[s][s1], r, copy(self.Sigma))
                else:
                    self.delta[s][s1] = r
        del self.delta[st]

    def eliminateAll(self, lr):
        """Eliminate a list of states.

        :param list lr: list of states indexes"""
        for s in lr:
            self.eliminate(s)

    def dup(self):
        """ Returns a copy of a GFA

        :rtype: GFA"""
        new = GFA()
        new.States = copy(self.States)
        new.Sigma = copy(self.Sigma)
        new.Initial = self.Initial
        new.Final = copy(self.Final)
        new.delta = deepcopy(self.delta)
        new.predecessors = deepcopy(self.predecessors)
        return new

    def normalize(self):
        """ Create a single initial and final state with Epsilon transitions.

        .. attention::
           works in place"""
        first = self.addState("First")
        self.predecessors[first] = set([])
        self.addTransition(first, Epsilon, self.Initial)
        self.setInitial(first)

        last = self.addState("Last")
        self.predecessors[last] = set([])
        if len(self.Final) > 1:
            for s in self.Final:
                self.addTransition(s, Epsilon, last)
                self.predecessors[last].add(s)
        else:
            self.addTransition(list(self.Final)[0], Epsilon, last)
        self.setFinal([last])

    # noinspection PyUnresolvedReferences
    def _do_edges(self, v1, t, rp):
        """ Labels for testing if a automaton is SP. used by SPRegExp

        :param int v1: state (node)
        :param SPlabel t: a label
        :param regexprp: reex.RegExp"""
        for v2 in self.delta[v1]:
            if self.out_index[v1] != 1:
                self.lab[(v1, v2)] = t.copy()
                self.lab[(v1, v2)].value.append(v1)
            else:
                self.lab[(v1, v2)] = t.ref()
                self.delta[v1][v2] = reex.CConcat(rp, self.delta[v1][v2], copy(self.Sigma))

    # noinspection PyUnresolvedReferences
    def _simplify(self, v2, i):
        """Used by SPRegExp.
        :param v2:
        :param i:
        :return:
        :raise NotSP:"""
        m, l = 0, []
        for v1 in self.predecessors[v2]:
            size = len(self.lab[(v1, v2)].val())
            if size == m:
                l.append(v1)
            elif size > m:
                m = size
                l = [v1]
        vi = l[-1]
        for vo in l[-2:]:
            if (self.lab[(vi, v2)].lastref() != self.lab[(vo, v2)].lastref()) and (
                    self.lab[(vi, v2)].val() == self.lab[(vo, v2)].val()):
                v = self.lab[(vi, v2)].val()[-1]
                self.out_index[v] -= 1
                self.lab[(vo, v2)] = self.lab[(vi, v2)].ref()
                self.delta[vi][v2] = reex.CDisj(self.delta[vo][v2], self.delta[vi][v2], copy(self.Sigma))
                if self.out_index[v] == 1:
                    self.lab[(vi, v2)].assign(self.lab[(vi, v2)].val()[:-1])
                    try:
                        self.delta[vi][v2] = reex.CConcat(self.delta[list(self.predecessors[v])[0]][v],
                                                          self.delta[vi][v2],
                                                          copy(self.Sigma))
                    except IndexError:
                        pass
                self.predecessors[v2].remove(vo)
                return i - 1
        raise NotSP

    def DFS(self, io):
        """Depth first search

        :param io:"""
        visited = []
        for s in range(len(self.States)):
            self.dfs_visit(s, visited, io)

    def dfs_visit(self, s, visited, io):
        """

        :param s: state
        :param visited: list od states visited
        :param io:"""
        if s not in visited:
            visited.append(s)
            if s in self.delta:
                for dest in self.delta[s]:
                    # lists are unhashable
                    (i, o) = io[s]
                    io[s] = (i, o + 1)
                    (i, o) = io[dest]
                    io[dest] = (i + 1, o)
                    self.dfs_visit(dest, visited, io)

    def weight(self, state):
        """Calculates the weight of a state based on a heuristic

        :param int state: state
        :returns: the weight of the state
        :rtype: int"""
        r = 0
        for i in self.predecessors[state]:
            if i != state:
                r += self.delta[i][state].alphabeticLength() * (len(self.delta[state]) - 1)
        for i in self.delta[state]:
            if i != state:
                r += self.delta[state][i].alphabeticLength() * (len(self.predecessors[state]) - 1)
        if state in self.delta[state]:
            r += self.delta[state][state].alphabeticLength() * (
                    len(self.predecessors[state]) * len(self.delta[state]) - 1)
        return r

    def weightWithCycles(self, state, cycles):
        """

        :param state:
        :param cycles:
        :return:"""
        r = 0
        for i in self.predecessors[state]:
            if i != state:
                r += self.delta[i][state].alphabeticLength() * (len(self.delta[state]) - 1)
        for i in self.delta[state]:
            if i != state:
                r += self.delta[state][i].alphabeticLength() * (len(self.predecessors[state]) - 1)
        if state in self.delta[state]:
            r += self.delta[state][state].alphabeticLength() * (
                    len(self.predecessors[state]) * len(self.delta[state]) - 1)
        r *= (cycles[state] + 1)
        return r

    def deleteState(self, sti):
        """ deletes a state from the GFA
        :param sti:"""
        new_order = {}
        for i in range(sti, len(self.States) - 1):
            new_order[i + 1] = i
        new_order[sti] = len(self.States) - 1
        self.reorder(new_order)
        st = len(self.States) - 1
        del self.delta[st]
        del self.predecessors[st]
        l = set([])
        for i in self.delta:
            if st in self.delta[i]:
                l.add(i)
        for i in l:
            del self.delta[i][st]
            if not len(self.delta[i]):
                del self.delta[i]
        for i in self.predecessors:
            if st in self.predecessors[i]:
                self.predecessors[i].remove(st)
        del self.States[st]

    def eliminateState(self, st):
        """ Deletes a state and updates the automaton

        :param int st: the state to be deleted

        .. attention:
           works in place"""
        for i in self.predecessors[st]:
            for j in self.delta[st]:
                if i != st and j != st:
                    rex = self.delta[i][st]
                    if st in self.delta[st]:
                        rex = reex.CConcat(rex, reex.CStar(self.delta[st][st], copy(self.Sigma)), copy(self.Sigma))
                    rex = reex.CConcat(rex, self.delta[st][j], copy(self.Sigma))
                    if j in self.delta[i]:
                        rex = reex.CDisj(self.delta[i][j], rex, copy(self.Sigma))
                    self.delta[i][j] = rex
                    self.predecessors[j].add(i)
        self.deleteState(st)

    def completeDelta(self):
        """Adds empty set transitions between the automatons final and initial states in order to make it complete.
        It's only meant to be used in the final stage of SEA..."""
        for i in set([self.Initial] + list(self.Final)):
            for j in set([self.Initial] + list(self.Final)):
                if i not in self.delta:
                    self.delta[i] = {}
                if j not in self.delta[i]:
                    self.delta[i][j] = reex.CEmptySet(copy(self.Sigma))

    def stateChildren(self, state, strict=False):
        """Set of children of a state

        :param bool strict: a state is never its own children even if a self loop is in place
        :param int state: state id queried
        :returns: map: children -> alphabetic length
        :rtype: dictionary"""
        l = {}
        if state not in self.delta:
            return l
        for c in self.delta[state]:
            l[c] = self.delta[state][c].alphabeticLength()
        if not strict and state in l:
            del l[state]
        return l

    def _re0(self):
        ii = self.Initial
        fi = list(self.Final)[0]
        a = self.delta[ii][ii]
        b = self.delta[ii][fi]
        c = self.delta[fi][ii]
        d = self.delta[fi][fi]

        # bd*
        re1 = reex.CConcat(b, reex.CStar(d), copy(self.Sigma))
        # a + bd*c
        re2 = reex.CDisj(a, reex.CConcat(re1, c, copy(self.Sigma)), copy(self.Sigma))
        # (a + bd*c)* bd*
        return reex.CConcat(reex.CStar(re2, copy(self.Sigma)), re1, copy(self.Sigma)).reduced()

    # noinspection PyUnresolvedReferences
    def assignNum(self, st):
        """

        :param st:"""
        self.num[st] = self.c
        self.c += 1
        self.visited.append(st)
        if st in self.delta:
            for d in self.delta[st]:
                if d not in self.visited:
                    self.parent[d] = st
                    self.assignNum(d)

    # noinspection PyUnresolvedReferences
    def assignLow(self, st):
        """

        :param st:"""
        self.low[st] = self.num[st]
        if st in self.delta:
            for d in self.delta[st]:
                if self.num[d] > self.num[st]:
                    self.assignLow(d)
                    if self.low[d] >= self.low[st]:
                        self.cuts.add(st)
                    self.low[st] = min(self.low[st], self.low[d])
                else:
                    if st in self.parent:
                        if self.parent[st] != d:
                            self.low[st] = min(self.low[st], self.num[d])
                    else:
                        self.low[st] = self.num[st]

    def evalNumberOfStateCycles(self):
        """Evaluates the number of cycles each state participates

        :returns: state->list of cycle lengths
        :rtype: dict"""
        cycles = {}
        seen = []
        for i, _ in enumerate(self.States):
            cycles[i] = 0
        (bkE, multipl) = self._DFSBackEdges()
        for (x, y) in bkE:
            self._chkForCycles(y, x, cycles, seen, multipl)
        return cycles

    def _chkForCycles(self, y, x, cycles, seen, multipl):
        """Used in evalNumberOfStateCycles"""
        s = y
        path = [x, y]
        stack = [[y for y in self.stateChildren(s)]]
        marked = [y]
        while stack:
            foo = stack.pop()
            if isinstance(foo, list) and len(foo):
                s = foo.pop()
                stack.append(foo)
            else:
                path.pop()
                continue
            if s in marked:
                continue
            elif s == x:
                bar = self._normalizeCycle(path)
                if bar not in seen:
                    seen.append(bar)
                    m = 1
                    for i in range(len(path) - 1):
                        m *= max(1, multipl[(path[i], path[i + 1])])
                    m *= max(1, multipl[(path[-1], path[0])])
                    for i in path:
                        cycles[i] = cycles.get(i, 0) + m
                continue
            else:
                marked.append(s)
                path.append(s)
            stack.append([y for y in self.stateChildren(s)])
        return cycles

    @staticmethod
    def _normalizeCycle(c):
        """Normalizes a cycle with its first element at the begining

        :param list c: cycle"""
        m = min(c)
        i = c.index(m)
        return c[i:] + c[:i]

    def _DFSBackEdges(self):
        """Returns a pair (BE, M) whee BE is the set of backedges form a DFS starting with the initial state as pairs
         (s, d) and M is a map (i, j)->multiplicity

        :returns: as said above
        :rtype: tuple"""
        m_states = set()
        b_edges = set()
        pool = set()
        multipl = {}
        if type(self.Initial) == set:  # NFAs
            m_states += self.Initial
            pool += self.Initial
        else:  # DFAs
            m_states.add(self.Initial)
            pool.add(self.Initial)
        while pool:
            s = pool.pop()
            child = self.stateChildren(s)
            # noinspection PyTypeChecker
            for r in child:
                multipl[(s, r)] = child[r]
            for i in child:
                if i in m_states or i in pool:
                    b_edges.add((s, i))
                else:
                    pool.add(i)
                    m_states.add(i)
        return b_edges, multipl


def DFAsyncWords(aut):
    """Evaluates the regular expression corresponding to the synchronizing pwords of the automata.

    :param DFA aut: the automata
    :return: a regular expression of the sync words of the automata
    :rtype: reex.RegExp"""
    return FA2regexpCG(aut.syncPower())
