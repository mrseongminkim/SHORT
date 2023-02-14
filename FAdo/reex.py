# -*- coding: utf-8 -*-
"""**Regular expressions manipulation**

Regular expression classes and manipulation

.. *Authors:* Rogério Reis & Nelma Moreira

.. Contributions by
    - Marco Almeida
    - Hugo Gouveia
    - Eva Maia
    - Rafaela Bastos

.. *This is part of FAdo project*   https://fado.dcc.fc.up.pt

.. *Version:* 2.0.2

.. *Copyright:* 1999-2022 Rogério Reis <rogerio.reis@fc.up.pt> & Nelma Moreira <nelma.moreira@fc.up.pt>


.. This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
   License as published by the Free Software Foundation; either version 2 of the License, or (at your COption) any
   later version.

   This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
   details.


   You should have received a copy of the GNU General Public License along with this program; if not, write to the
   Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA."""

#  Copyright (c) 2022. Rogério Reis <rogerio.reis@fc.up.pt> and Nelma Moreira <nelma.moreira@fc.up.pt>.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import copy
from . import fa
import lark
# from lark import Lark
# from itertools import chain, combinations
from collections import deque
from .unionFind import UnionFind
from .common import *
from .common import EmptySet


class RegularExpression(object):
    """Abstract base class for all regular expression objects"""
    pass


# noinspection PyProtectedMember
class RegExp(RegularExpression):
    """Base class for regular expressions.

    :ivar Sigma: alphabet set of strings

    .. inheritance-diagram:: RegExp"""

    def __init__(self, sigma=None):
        self.val = None
        self.Sigma = sigma
        self.arg = None

    @abstractmethod
    def rpn(self):
        """ RPN representation

        Returns:
            str: printable RPN representation"""
        pass

    @staticmethod
    @abstractmethod
    def alphabeticLength():
        """Number of occurrences of alphabet symbols in the regular expression.
        Returns:
         int: alphapetic length

        .. attention:: Doesn't include the empty word."""
        return 0

    @staticmethod
    @abstractmethod
    def treeLength():
        """Number of nodes of the regular expression's syntactical tree.

        Returns:
            int: tree lenght"""
        pass

    @staticmethod
    @abstractmethod
    def epsilonLength():
        """Number of occurrences of the empty word in the regular expression.

        Returns:
             int: number of epsilons"""
        pass

    @staticmethod
    @abstractmethod
    def starHeight():
        """Maximum level of nested regular expressions with a star operation applied.

        For instance, starHeight(((a*b)*+b*)*) is 3.

        Returns:
             int: number of nested star"""
        return 0

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def unmark(self):
        pass

    @abstractmethod
    def mark(self):
        """ Make all atoms maked (tag False)

        Returns:
             RegExp: """
        pass

    @abstractmethod
    def linearForm(self):
        """
        Returns:
             dict: linear form """
        pass

    @abstractmethod
    def tailForm(self):
        """
        Returns:
            dict: tail form """
        pass

    @abstractmethod
    def reduced(self):
        pass

    @staticmethod
    def emptysetP():
        """Whether the regular expression is the empty set.

        Returns:
             bool: """
        return False

    @abstractmethod
    def first(self):
        """ First set

        Returns:
            set: first position set"""
        pass

    @abstractmethod
    def followLists(self):
        """ Follow set

        Returns:
             dict: for each key  position a set of follow positions"""
        pass

    @abstractmethod
    def followListsD(self):
        """ Follow set

            Returns:
                dict: for each key  position a set of follow positions"""
        pass

    @abstractmethod
    def last(self):
        """ Last set

        Returns:
             set: last position set"""
        pass

    @abstractmethod
    def _marked(self, _):
        pass

    @abstractmethod
    def _memoLF(self):
        pass

    @staticmethod
    @abstractmethod
    def setOfSymbols():
        """
        Returns:
            set: set of symbols """
        pass

    @abstractmethod
    def derivative(self, _):
        pass

    def _setSigma(self, strict=False):
        """
        Args:
           strict: bool"""
        pass

    @abstractmethod
    def snf(self):
        """ Star Normal Form"""
        pass

    @abstractmethod
    def support(self, side=True):
        """ Set of partial derivatives """
        pass

    @abstractmethod
    def supportlast(self, side=True):
        pass

    @abstractmethod
    def _follow(self, _):
        pass

    @abstractmethod
    def _nfaFollowEpsilonStep(self, _):
        pass

    def toNFA(self, nfa_method="nfaPD"):
        """NFA that accepts the regular expression's language.
        :param nfa_method: """

        return self.__getattribute__(nfa_method)()

    def toDFA(self):
        """DFA that accepts the regular expression's language

        .. versionadded 0.9.6"""
        # return self.dfaAuPoint() Fails in tests
        return self.toNFA().toDFA()

    def unionSigma(self, other):
        """Returns the union of two alphabets

        :type other: RegExp
        :rtype: set """
        if self.Sigma is None:
            return other.Sigma
        elif other.Sigma is None:
            return self.Sigma
        else:
            return self.Sigma.union(other.Sigma)

    def nfaPD(self, pdmethod="nfaPDDAG"):
        """Computes the partial derivative automaton

        Args:
            pdmethod (str): an implementation of the PD automaton. Default value : nfaPDDAG
        Returns:
            NFA: a PD nfa

        .. attention:: for sre classes, CConj and CShuffle  use nfaPDNaive directly"""
        return self.__getattribute__(pdmethod)()

    def nfaPDDAG(self):
        """ Partial derivative automaton using a DAG for the re and partial derivatives

        Returns:
            NFA: a PD nfa build using a DAG

        ..seealso:: s.Konstantinidis, A. Machiavelo, N. Moreira, and r. Reis.
                    Partial derivative automaton by compressing regular expressions.
                    DCFS 2021, volume 13037 of LNCS, pages 100--112. Springer, 2022"""
        return DAG(self).NFA()

    def nfaPDNaive(self):
        """NFA that accepts the regular expression's language,
           and which is constructed from the expression's partial derivatives.

        Returns:
            NFA: partial derivatives [or equation] automaton

        .. seealso:: V. M. Antimirov, Partial Derivatives of Regular Expressions and Finite Automaton Constructions
           .Theor. Comput. Sci.155(2): 291-319 (1996)"""
        aut = fa.NFA()
        i = aut.addState(self)
        aut.addInitial(i)
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        stack = [(self, i)]
        added_states = {self: i}
        while stack:
            state, state_idx = stack.pop()
            state_lf = state.linearForm()
            for head in state_lf:
                tails = state_lf[head]
                aut.addSigma(head)
                for pd in tails:
                    if pd in added_states:
                        pd_idx = added_states[pd]
                    else:
                        try:
                            pd_idx = aut.addState(pd)
                        except DuplicateName:
                            pd_idx = aut.stateIndex(pd)
                        added_states[pd] = pd_idx
                        stack.append((pd, pd_idx))
                    aut.addTransition(state_idx, head, pd_idx)
            if state.ewp():
                aut.addFinal(state_idx)
        return aut

    def nfaPre(self):
        """ Prefix NFA of a regular expression

        Returns:
            NFA: prefix automaton

        .. note States are of the form (RegExp,sym)

        .. seealso:: Maia et al, Prefix and Right-partial derivative automata, 11th CIE 2015, 258-267  LNCS 9136, 2015"""
        aut = fa.NFA()
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        i = aut.addState(CEpsilon())
        aut.addInitial(i)
        todo = []
        added_states = {CEpsilon(): i}
        if self.ewp():
            aut.addFinal(i)
        state_tf = self.tailForm()
        for tail in state_tf:
            heads = state_tf[tail]
            aut.addSigma(tail)
            for p_t in heads:
                p = (p_t, tail)
                if p not in added_states:
                    try:
                        p_idx = aut.addState(p)
                    except DuplicateName:
                        p_idx = aut.stateIndex(p)
                    aut.addFinal(p_idx)
                    added_states[p] = p_idx
                    todo.append((p_t, tail, p_idx))
        while todo:
            state, tail_st, state_idx = todo.pop()
            if state.ewp():
                aut.addTransition(i, tail_st, state_idx)
            state_tf = state.tailForm()
            for tail in state_tf:
                heads = state_tf[tail]
                aut.addSigma(tail)
                for p_t in heads:
                    p = (p_t, tail)
                    if p in added_states:
                        p_idx = added_states[p]
                    else:
                        try:
                            p_idx = aut.addState(p)
                        except DuplicateName:
                            p_idx = aut.stateIndex(p)
                        added_states[p] = p_idx
                        todo.append((p_t, tail, p_idx))
                    aut.addTransition(p_idx, tail_st, state_idx)
        return aut

    # def nfaPreSlow(self):
    #     """
    #     Prefix NFA of a regular expression
    #     :return: prefix automaton
    #     :rtype: NFA
    #     .. seealso:: Maia et al, Prefix and Right-partial derivative automata, 11th CIE 2015, 258-267  LNCS 9136, 2015
    #     ..note:: not working with current tailForm
    #     """
    #     afn = fa.NFA()
    #     if self.Sigma is not None:
    #         afn.setSigma(self.Sigma)
    #     i = afn.addState(CEpsilon())
    #     afn.addInitial(i)
    #     todo = []
    #     added_states = {CEpsilon(): i}
    #     if self.ewp():
    #         afn.addFinal(i)
    #     state_tf = self.tailForm()
    #     for tail in state_tf:
    #         heads = state_tf[tail]
    #         afn.addSigma(tail)
    #         for p, p_t in heads:
    #             # if p_t.epsilonP():
    #             #     p = CAtom(tail, self.sigma)
    #             # else:
    #             #     p = CConcat(p_t, CAtom(tail), self.sigma)
    #             if p not in added_states:
    #                 try:
    #                     p_idx = afn.addState(p)
    #                 except DuplicateName:
    #                     p_idx = afn.stateIndex(p)
    #                 afn.addFinal(p_idx)
    #                 added_states[p] = p_idx
    #                 todo.append((p_t, tail, p_idx))
    #     while todo:
    #         state, tail_st, state_idx = todo.pop()
    #         if state.ewp():
    #             afn.addTransition(i, tail_st, state_idx)
    #         state_tf = state.tailForm()
    #         for tail in state_tf:
    #             heads = state_tf[tail]
    #             afn.addSigma(tail)
    #             for p, p_t in heads:
    #                 # if p_t.epsilonP():
    #                 #     p = CAtom(tail, self.sigma)
    #                 # else:
    #                 #     p = CConcat(p_t, CAtom(tail), self.sigma)
    #                 if p in added_states:
    #                     p_idx = added_states[p]
    #                 else:
    #                     try:
    #                         p_idx = afn.addState(p)
    #                     except DuplicateName:
    #                         p_idx = afn.stateIndex(p)
    #                     added_states[p] = p_idx
    #                     todo.append((p_t, tail, p_idx))
    #                 afn.addTransition(p_idx, tail_st, state_idx)
    #     return afn

    def nfaLoc(self):
        """Location automaton of the regular expression.

        Returns:
            NFA: location nfa

        .. seealso: S. Broda, A. Machiavelo, N. Moreira, and R. Reis.
                "Location based automata for expressions with shuffle" LATA 2021, LNCS 12638,pp 43--54, 2021 """
        aut = fa.NFA()
        initial = aut.addState("Initial")
        aut.addInitial(initial)
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        return self.marked()._faLoc(aut, initial)

    def _faLoc(self, aut, initial):
        if self.ewp():
            aut.addFinal(initial)
        stack = []
        added_states = dict()
        fst = self.first_l()
        for id, loc in fst:
            sym = id.val[0]
            try:
                state_idx = aut.addState(str(loc))
            except DuplicateName:
                state_idx = aut.stateIndex(str(loc))
            added_states[loc] = state_idx
            stack.append((loc, state_idx))
            aut.addTransition(initial, sym, state_idx)
        follow_sets = self.follow_l()
        while stack:
            state, state_idx = stack.pop()
            for id, loc in follow_sets[state]:
                sym = id.val[0]
                if loc in added_states:
                    next_state_idx = added_states[loc]
                else:
                    next_state_idx = aut.addState(str(loc))
                    added_states[loc] = next_state_idx
                    stack.append((loc, next_state_idx))
                aut.addTransition(state_idx, sym, next_state_idx)
        lst = self.last_l()
        for loc in lst:
            if loc in added_states:
                aut.addFinal(added_states[loc])
        aut.epsilon_transitions = False
        return aut

    @staticmethod
    def epsilonP():
        """Whether the regular expression is the empty word.

        Returns:
            bool:"""
        return False

    @staticmethod
    def ewp():
        """Whether the empty word property holds for this regular expression's language.

        Returns:
             bool:"""
        return False

    def _odot(self, sre, d=True):
        """ Concatenates a set of res or tuples of res with a re

        Args
            sre (set): set of res
            d (bool): if True concatenates re on the right else on the left
        Returns:
            RegExp: the resulting expression"""

        if sre == set():
            return set()
        f = list(sre)
        if f[0] is tuple:
            try:
                if f[0][1] == 2:
                    return {(_ifconcat(j, self, d, sigma=self.Sigma), (s, _ifconcat(i, self, d, self.Sigma))) for
                            j, (s, i) in f}
                else:
                    return {(j, _ifconcat(i, self, d, sigma=self.Sigma)) for j, i in f}
            except KeyError:
                raise FAdoGeneralError("tuple with wrong size")
        else:
            return {_ifconcat(i, self, d, sigma=self.Sigma) for i in f}

    def __eq__(self, r):
        """Whether the string representations of two regular expressions are equal.

        Args:
            r (RegExp): regular expression to compare with self
        Returns:
             bool: True if the two string representations are equal"""
        if type(r) == type(self) and ((self.Sigma is None) or (r.Sigma is None) or (self.Sigma == r.Sigma)):
            return repr(self) == repr(r)
        else:
            return False

    def __ne__(self, r):
        """Whether the string representations of two regular expressions are different.

        Args:
            r (RegExp): regular expression to compare with self
        Returns:
            bool: True if the two string representations are diffenrent
        """
        return not self.__eq__(r)

    def __hash__(self):
        """Hash over regular expression's string representation"""
        return hash(repr(self))

    def _faPosition(self, aut, initial, lstar=True):
        if self.ewp():
            aut.addFinal(initial)
        stack = []
        added_states = dict()
        for sym in self.first():
            try:
                state_idx = aut.addState(str(sym))
            except DuplicateName:
                state_idx = aut.stateIndex(str(sym))
            added_states[sym] = state_idx
            stack.append((sym, state_idx))
            aut.addTransition(initial, sym.symbol(), state_idx)
        if lstar is False:
            follow_sets = self.followLists()
        else:
            follow_sets = self.followListsD()
        while stack:
            state, state_idx = stack.pop()
            for sym in follow_sets[state]:
                if sym in added_states:
                    next_state_idx = added_states[sym]
                else:
                    next_state_idx = aut.addState(str(sym))
                    added_states[sym] = next_state_idx
                    stack.append((sym, next_state_idx))
                aut.addTransition(state_idx, sym.symbol(), next_state_idx)
        for sym in self.last():
            if sym in added_states:
                aut.addFinal(added_states[sym])
        aut.epsilon_transitions = False
        return aut

    def nfaPosition(self, lstar=True):
        """Position automaton of the regular expression.

        Args:
            lstar (bool): if not None followlists are computed as disjunct
        Returns:
            NFA: Position NFA

        .. seealso:  Glushkov, 61

        .. attention: if lstar None no check of repeated positions is done"""
        aut = fa.NFA()
        initial = aut.addState("Initial")
        aut.addInitial(initial)
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        return self.marked()._faPosition(aut, initial, lstar)

    def dfaYMG(self):
        """ DFA Yamada-McNaugthon-Gluskov acconding to Nipkow

        Returns:
            DFA:  Y-M-G DFA

        .. seealso:: Tobias Nipkow and Dmitriy Traytel, Unified Decision Procedures for
            Regular Expression Equivalence"""

        aut = fa.DFA()
        sigma = self.setOfSymbols()
        s = (True, self.mark())
        i = aut.addState(s)
        stack = [(s, i)]
        added = {s}
        aut.setInitial(i)
        while stack:
            ((m, r), i) = stack.pop()
            if r._final() or (m and r.ewp()):
                aut.addFinal(i)
            for a in sigma:
                s1 = (False, r._follow(m)._read(a))
                if s1 not in added:
                    i1 = aut.addState(s1)
                    stack.append((s1, i1))
                    added.add(s1)
                else:
                    i1 = aut.stateIndex(s1)
                aut.addTransition(i, a, i1)
        return aut

    def dfaAuPoint(self):
        """ DFA "au-point" acconding to Nipkow

        Returns:
            DFA: "au-point" DFA

        .. seealso:: Andrea Asperti, Claudio Sacerdoti Coen and Enrico Tassi, Regular Expressions, au point. arXiv 2010

        .. seealso:: Tobias Nipkow and Dmitriy Traytel, Unified Decision Procedures for
            Regular Expression Equivalence"""
        aut = fa.DFA()
        sigma = self.setOfSymbols()
        s = (self.ewp(), self.mark()._follow(True))
        i = aut.addState(s)
        stack = [(s, i)]
        added = {s}
        aut.setInitial(i)
        while stack:
            ((m, r), i) = stack.pop()
            if m:
                aut.addFinal(i)
            for a in sigma:
                foo = r._read(a)
                s1 = (foo._final(), foo._follow(False))
                if s1 not in added:
                    i1 = aut.addState(s1)
                    stack.append((s1, i1))
                    added.add(s1)
                else:
                    i1 = aut.stateIndex(s1)
                aut.addTransition(i, a, i1)
        return aut

    def _dfaPosition(self):
        """Deterministic Position automaton of a regular expression.

        Returns:
            DFA: Position DFA

        Raises:
             common.DFAnotNFAFAdo: if not DFA

        .. note:: If this expression is not linear (cf. linearP()), exception may be raised
                  on non-deterministic transitions.

        .. seealso:  Glushkov, 61"""
        dfa = fa.DFA()
        initial = dfa.addState("Initial")
        dfa.setInitial(initial)
        if self.Sigma is not None:
            dfa.setSigma(self.Sigma)
        return self.marked()._faPosition(dfa, initial)

    def nfaPSNF(self):
        """Position or Glushkov automaton of the regular expression constructed from the expression's star normal form.

        Returns:
            NFA: Position automaton

        .. seeall: Brüggemann-Klein, 92"""
        return self.snf().nfaPosition(lstar=False)

    def nfaPDO(self):
        """NFA that accepts the regular expression's language, and which is constructed from the expression's partial
         derivatives.

        Returns:
            NFA: partial derivatives [or equation] automaton

        .. note:: optimized version"""
        aut = fa.NFA()
        i = aut.addState(self)
        aut.addInitial(i)
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        stack = [(self, i)]
        added_states = {self: i}
        while stack:
            state, state_idx = stack.pop()
            state._memoLF()
            for head in state._lf:
                aut.addSigma(head)
                for pd in state._lf[head]:
                    if pd in added_states:
                        pd_idx = added_states[pd]
                    else:
                        pd_idx = aut.addState(pd)
                        added_states[pd] = pd_idx
                        stack.append((pd, pd_idx))
                    aut.addTransition(state_idx, head, pd_idx)
            if state.ewp():
                aut.addFinal(state_idx)
        self._delAttr("_lf")
        return aut

    def _dfaD(self):
        """Word derivatives automaton of the regular expression

        Returns:
            DFA: word derivatives automaton

        .. attention:
             This is a probably non terminating method. Must be removed. (nam)
        .. seealso:
            J. A. Brzozowski, Derivatives of Regular Expressions. J. ACM 11(4): 481-494 (1964)"""
        dfa = fa.DFA()
        initial = self
        initial_idx = dfa.addState(initial)
        dfa.setInitial(initial_idx)
        if self.Sigma is not None:
            dfa.setSigma(self.Sigma)
        dfa.setSigma(initial.setOfSymbols())
        stack = [(initial, initial_idx)]
        while stack:
            state, state_idx = stack.pop()
            for sigma in dfa.Sigma:
                d = state.derivative(sigma).reduced()
                if d not in dfa.States:
                    d_idx = dfa.addState(d)
                    stack.append((d, d_idx))
                else:
                    d_idx = dfa.stateIndex(d)
                dfa.addTransition(state_idx, sigma, d_idx)
            if state.ewp():
                dfa.addFinal(state_idx)
        return dfa

    def _delAttr(self, attr):
        if hasattr(self, attr):
            delattr(self, attr)

    def marked(self):
        """Regular expression in which every alphabetic symbol is marked with its Position.

        The kind of regular expression returned is known, depending on the literary source, as marked,
        linear or restricted regular expression.

        Returns:
            RegExp: linear regular expression

        .. seealso:: r. McNaughton and H. Yamada, Regular Expressions and State Graphs for Automata,
            IEEE Transactions on Electronic Computers, V.9 pp:39-47, 1960

        ..attention: mark and unmark do not preserve the alphabet, neither set the new alphabet """
        return self._marked(0)[0]

    def setSigma(self, symbolset=None, strict=False):
        """ Set the alphabet for a regular expression and all its nodes

        Args:
            symbolset (set or list): accepted symbols. If None, alphabet is unset.
            strict (bool): if True checks if setOfSymbols is included in symbolSet


        ..attention: Normally this attribute is not defined in a RegExp()"""
        if symbolset is not None:
            if strict and not (self.setOfSymbols() <= symbolset):
                raise regexpInvalidSymbols()
            self.Sigma = set(symbolset)
        else:
            self.Sigma = None
        # self._setSigma(symbolset)
        self._setSigma(strict)

    def equivalentP(self, other):
        """Tests equivalence

        Args:
            other (RegExp): other regexp
        Returns:
             bool: True if regexps are equivalent

        .. versionadded: 0.9.6"""
        if issubclass(type(other), fa.OFA):
            return equivalentP(self, other)
        return self.compare(other)

    def compareMinimalDFA(self, r, nfa_method="nfaPosition"):
        """Compare with another regular expression for equivalence through minimal DFAs.

        Args:
            r (RegExp):
            nfa_method (str): NTFA construction
        Returns:
            bool: True if equivalent"""
        fa0 = self.toNFA(nfa_method).toDFA()
        fa1 = r.toNFA(nfa_method).toDFA()
        return fa0 == fa1

    def evalWordP(self, word):
        """Verifies if a word is a member of the language represented by the regular expression.

        Args:
            word (str): the word
        Returns:
             bool: True if word belongs to the language"""

        return self.wordDerivative(word).ewp()

    def compare(self, r, cmp_method="compareMinimalDFA", nfa_method="nfaPD"):
        """Compare with another regular expression for equivalence.

        Args:
            r (RegExp):
            cmp_method (str):
            nfa_method (str): NFA construction
        Returns:
              bool: True if the expressions are equivalent"""
        if cmp_method == "compareMinimalDFA":
            return self.compareMinimalDFA(r, nfa_method)
        return self.__getattribute__(cmp_method)(r)

    def nfaNaiveFollow(self):
        """NFA that accepts the regular expression's language, and is equal in structure to the follow automaton.

        Returns:
            NFA: NFA follow

        .. note:: Included for testing purposes.

        .. seealso:: Ilie & Yu (Follow Automata, 2003)"""
        return self.snf().marked().nfaGlushkov().minimal().unmark()

    def dfaNaiveFollow(self):
        """DFA that accepts the regular expression's language, and is obtained from the follow automaton.

                Returns:
                    NFA: DFA follow

                .. note:: Included for testing purposes.

                .. seealso:: Ilie & Yu (Follow Automata, 2003)"""
        return self.nfaNaiveFollow().toDFA()

    def dfaPD(self):
        return self.nfaPD().toDFA()

    def nfaFollow(self):
        """NFA that accepts the regular expression's language, whose structure, equiand construction.

        Returns:
            NFA: NFA follow

        .. note: The regular expression must be reduced

        .. seealso:: Ilie & Yu (Follow Automata, 03)"""
        aut = self.nfaFollowEpsilon(False).toNFA()
        queue = deque(aut.Initial)
        inverse_topo_order = deque()
        visited = set(aut.Initial)
        while queue:
            state = queue.popleft()
            if aut.hasTransitionP(state, Epsilon):
                inverse_topo_order.appendleft(state)
            if state in aut.delta:
                for symbol in aut.delta[state]:
                    for s in aut.delta[state][symbol]:
                        if s not in visited:
                            queue.append(s)
                            visited.add(s)
        for i in inverse_topo_order:
            aut.closeEpsilon(i)
        aut.trim()
        aut.epsilon_transitions = False
        return aut

    def _nfaGlushkovStep(self, aut, initial, final):
        """
        Args:
            aut (FA)
            initial (state):
            final (state):
        Returns:
        """
        return None, None

    def nfaGlushkov(self):
        """ Position or Glushkov automaton of the regular expression. Recursive method.

        Returns:
             NFA: NFA position

        .. seealso: Yu, S. "Regular Languages" in Handbook Formal Languages, 1998 """
        aut = fa.NFA()
        initial = aut.addState("Initial")
        aut.addInitial(initial)
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        _, final = self._nfaGlushkovStep(aut, aut.Initial, set())
        aut.Final = final
        aut.epsilon_transitions=  False
        return aut

    def nfaFollowEpsilon(self, trim=True):
        """Epsilon-NFA constructed with Ilie and Yu's method () that accepts the regular expression's language.

        Args:
            trim (bool): if True automaton is trimmed at the end
        Returns:
            NFAe:  possibly with epsilon transitions

        .. note:: The regular expression must be reduced

        .. seealso:: Ilie & Yu, Follow automta, Inf. Comp. ,v. 186 (1),140-162,2003

        .. _a link: http://dx.doi.org/10.1016/S0890-5401(03)00090-7"""
        aut = fa.NFAr()
        initial = aut.addState("Initial")
        final = aut.addState("Final")
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        self._nfaFollowEpsilonStep((aut, initial, final))
        if len(aut.delta.get(initial, [])) == 1 and \
                len(aut.delta[initial].get(Epsilon, [])) == 1:
            new_initial = aut.delta[initial][Epsilon].pop()
            del (aut.delta[initial])
            aut.deltaReverse[new_initial][Epsilon].remove(initial)
            initial = new_initial
        aut.setInitial([initial])
        aut.setFinal([final])
        if trim:
            aut.trim()
        return aut

    def wordDerivative(self, word):
        """Derivative of the regular expression in relation to the given word,
           which is represented by a list of symbols.

        Args:
            word (list): list of arbitrary symbols.
        Returns:
             RegExp: regular expression

        .. seealso:: J. A. Brzozowski, Derivatives of Regular Expressions. J. ACM 11(4): 481-494 (1964)

        .. note: semantically, the list represents a catenation of symbols (word), and its alphabet is not checked."""
        d = copy.deepcopy(self)
        for sigma in word:
            d = d.derivative(sigma)
        return d

    def equivP(self, other, strict=True):
        """
        Test RE equivalence with extended Hopcroft-Karp method

        Args:
            other (RegExp): RE
            strict (bool): if True checks for same alphabets
        Returns:
            bool:
        """
        if strict and self.Sigma != other.Sigma:
            return False
        i1 = frozenset([self])
        i2 = frozenset([other])
        s = UnionFind(auto_create=True)
        s.union(i1, i2)
        stack = [(i1, i2)]
        while stack:
            (p, q) = stack.pop()
            if (True in {pd.ewp() for pd in p}) != (True in {pd.ewp() for pd in q}):
                return False
            p_lf = dict()
            for pd in p:
                pdd = pd.linearForm()
                for head in pdd:
                    if head in p_lf:
                        p_lf[head].update(pdd[head])
                    else:
                        p_lf[head] = pdd[head]
            q_lf = dict()
            for pd in q:
                pdd = pd.linearForm()
                for head in pdd:
                    if head in q_lf:
                        q_lf[head].update(pdd[head])
                    else:
                        q_lf[head] = pdd[head]
            for sigma in set(list(p_lf.keys()) + list(q_lf.keys())):
                p1 = s.find(frozenset(p_lf.get(sigma, [])))
                q1 = s.find(frozenset(q_lf.get(sigma, [])))
                if p1 != q1:
                    s.union(p1, q1)
                    stack.append((p1, q1))
        return True

    def notEmptyW(self):
        """
        Witness of non emptyness

        Returns:
             word or None:"""
        done = set()
        not_done = set()
        pref = dict()
        si = self
        pref[si] = Epsilon
        not_done.add(si)
        while not_done:
            p = not_done.pop()
            if p.ewp():
                return pref[p]
            done.add(p)
            p_lf = p.linearForm()
            for sigma in p_lf:
                tails = p_lf[sigma]
                for p1 in tails:
                    if p1 in done or p1 in not_done:
                        continue
                    pref[p1] = _concat(pref[p], sigma)
                    not_done.add(p1)
        return None

    def _equivP(self, r):
        """Verifies if two regular expressions are equivalent.

        Args:
            r (RegExp): the other regexp
        Returns:
            bool: True if equivalent
        .. note: uses Brzozowski derivatives, so it may not stop; for use with sre"""
        s = [(self, r)]
        h = {(self, r)}
        sigma = self.setOfSymbols().union(r.setOfSymbols())
        while s:
            s1, s2 = s.pop()
            if s1.ewp() != s2.ewp():
                return False
            for a in sigma:
                der1 = s1.derivative(a)
                der2 = s2.derivative(a)
                if (der1, der2) not in h:
                    h.add((der1, der2))
                    s.append((der1, der2))
        return True

    def dfaBrzozowski(self, memo=None):
        """Word derivatives automaton of the regular expression

        Args:
            memo: if True memorizes the states already computed
        Returns:
            DFA: word derivatives automaton

        .. seealso:: J. A. Brzozowski, Derivatives of Regular Expressions. J. ACM 11(4): 481-494 (1964)"""
        states = None
        dfa = fa.DFA()
        i = dfa.addState(self)
        dfa.setInitial(i)
        stack = [(self, i)]
        sigma = self.setOfSymbols()
        if memo:
            states = {self: i}
        while stack:
            state, idx = stack.pop()
            state.setSigma(sigma)
            for symbol in sigma:
                st = state.derivative(symbol)
                if memo:
                    if st in states:
                        i = states[st]
                    else:
                        i = dfa.addState(st)
                        states[st] = i
                        stack.append((st, i))
                else:
                    if st in dfa.States:
                        i = dfa.stateIndex(st)
                    else:
                        i = dfa.addState(st)
                        stack.append((st, i))
                dfa.addTransition(idx, symbol, i)
            if state.ewp():
                dfa.addFinal(idx)
        return dfa

    def _dot(self, r):
        """Computes the concatenation of two regular expressions.

        Args:
            r (RegExp): a regular expression
        Returns
            RegExp: concatenation

        .. note: used in sre expressions"""
        if r.epsilonP() or r.emptysetP():
            return r._dot(self)
        elif type(r) is SConcat:
            r0 = SConcat((self,) + r.arg, self.Sigma)
            if self.Sigma and r0.Sigma:
                r0.Sigma = r.Sigma | self.Sigma
            return r0
        else:
            r0 = SConcat((self, r), self.Sigma)
            if self.Sigma and r.Sigma:
                r0.Sigma = r.Sigma | self.Sigma
            return r0

    def _plus(self, r0):
        """Computes the disjunction of two regular expressions.

        Args:
            r0 (RegExp): a regular expression
        Returns:
            RegExp: disjunction"""
        if r0 == self:
            if self.Sigma and r0.Sigma:
                r0.Sigma = r0.Sigma | self.Sigma
            return r0
        elif type(r0) is CEmptySet or type(r0) is CSigmaS or type(r0) is SDisj:
            return r0._plus(self)
        else:
            r = SDisj(frozenset([self, r0]), self.Sigma)
            if self.Sigma and r0.Sigma:
                r.Sigma = r0.Sigma | self.Sigma
            return r

    def _inter(self, r):
        """Computes the conjunction of two regular expressions.

        Args:
            r (RegExp): a regular expression
        Returns:
            RegExp: conjunction"""
        if r == self:
            if self.Sigma and r.Sigma:
                r.Sigma = r.Sigma | self.Sigma
            return r
        elif r.emptysetP() or type(r) is CSigmaS or type(r) is SConj:
            return r._inter(self)
        else:
            r1 = SConj(frozenset([self, r]), self.Sigma)
            if self.Sigma and r.Sigma:
                r1.Sigma = r.Sigma | self.Sigma
            return r1


class CAtom(RegExp):
    """ Simple Atom (symbol)

    :ivar Sigma: alphabet set of strings
    :ivar val: the actual symbol

    .. inheritance-diagram:: RegExp"""

    def __init__(self, val, sigma=None):
        """Constructor of a regular expression symbol.

        :arg val: the actual symbol"""
        super(CAtom, self).__init__()
        self.val = val
        self.Sigma = sigma

    def __repr__(self):
        """Representation of the regular expression's syntactical tree."""
        return 'CAtom({0:>s})'.format(self.__str__())

    def __str__(self):
        """String representation of the regular expression."""
        return str(self.val)

    _strP = __str__

    def __len__(self):
        """Size of the RE (the tree length)

        Returns:
             int: tree length"""
        return self.treeLength()

    def mark(self):
        """
        Returns:
            MAtom: """
        return MAtom(self.val, False, self.Sigma)

    @abstractmethod
    def unmark(self):
        pass

    def rpn(self):
        """RPN representation

        Returns:
            str: printable RPN representation"""
        return "%s" % repr(self.val)

    def __copy__(self):
        """Reconstruct the regular expression's syntactical tree, or, in other words,
           perform a shallow copy of the tree.

        Returns:
             CAtom:

        .. note::
           References to the expression's symbols in the leafs are preserved.

        .. attention:: Raw modifications on the regular expression's tree should be performed over a copy returned
        by this method, so that cached methods do not interfere."""
        return CAtom(self.val)

    def setOfSymbols(self):
        """Set of symbols that occur in a regular expression..

        Returns:
            set: set of symbols"""
        return {self.val}

    def stringLength(self):
        """Length of the string representation of the regular expression.

        Returns:
            int: string length"""
        return len(str(self))

    @staticmethod
    def measure(from_parent=None):
        """A list with four measures for regular expressions.

        Args:
            from_parent:
        Returns:
             [int,int,int,int]: the measures

        [alphabeticLength, treeLength, epsilonLength, starHeight]

        1. alphabeticLength: number of occurences of symbols of the alphabet;

        2. treeLength: number of functors in the regular expression, including constants.

        3. epsilonLength: number of occurrences of the empty word.

        4. starHeight: highest level of nested Kleene stars, starting at one for one star occurrence.

        5. disjLength: number of occurrences of the disj operator

        6. concatLength: number of occurrences of the concat operator

        7. starLength: number of occurrences of the star operator

        8. conjLength: number of occurrences of the conj operator

        9. starLength: number of occurrences of the shuffle operator

        .. attention::
           Methods for each of the measures are implemented independently. This is the most effective for obtaining
           more than one measure."""
        if not from_parent:
            from_parent = 9 * [0]
        from_parent[0] += 1
        from_parent[1] += 1
        return from_parent

    @staticmethod
    def alphabeticLength():
        """Number of occurrences of alphabet symbols in the regular expression.

        Returns:
             int:

        .. attention:: Doesn't include the empty word."""
        return 1

    @staticmethod
    def treeLength():
        """Number of nodes of the regular expression's syntactical tree.

        Returns:
             int:"""
        return 1

    @staticmethod
    def syntacticLength():
        """Number of nodes of the regular expression's syntactical tree (sets).

        Returns:
             int:"""
        return 1

    @staticmethod
    def epsilonLength():
        """Number of occurrences of the empty word in the regular expression.

        Returns:
             int:"""
        return 0

    @staticmethod
    def starHeight():
        """Maximum level of nested regular expressions with a star operation applied.
            For instance, starHeight(((a*b)*+b*)*) is 3.

        Returns:
             int:"""
        return 0

    def reduced(self, has_epsilon=False):
        """Equivalent regular expression with the following cases simplified:

        1. Epsilon.RE = RE.Epsilon = RE

        2. EmptySet.RE = RE.EmptySet = EmptySet

        3. EmptySet + RE = RE + EmptySet = RE

        4. Epsilon + RE = RE + Epsilon = RE, where Epsilon is in L(RE)

        5. RE** = RE*

        6. EmptySet* = Epsilon* = Epsilon

        7.Epsilon:RE = RE:Epsilon= RE

        Args:
            has_epsilon (bool): used internally to indicate that the language of which this term is a subterm has the empty
            word.
        Returns:
            RegExp: regular expression

        .. attention::
           Returned structure isn't strictly a duplicate. Use __copy__() for that purpose."""
        return self

    _reducedS = reduced

    def linearP(self):
        """Whether the regular expression is linear; i.e., the occurrence of a symbol in the expression is unique.

        Returns:
             bool:"""
        return len(self.setOfSymbols()) is self.alphabeticLength()

    def _nfaFollowEpsilonStep(self, conditions):
        """Construction step of the Epsilon-NFA defined by Ilie & Yu for this class.

        Args:
            conditions (tuple): A tuple consisting of an NFA, the initial state, and the final state in the context. A
        sub-automaton within the given automaton is thus constructed."""
        aut, initial, final = conditions
        aut.addSigma(self.val)
        aut.addTransition(initial, self.val, final)

    def _nfaGlushkovStep(self, aut, initial, final):
        try:
            target = aut.addState(self.val)
        except DuplicateName:
            target = aut.addState()
            # target = aut.stateIndex(self.val)
        for source in initial:
            aut.addTransition(source, self.val, target)
        final.add(target)
        return initial, final

    def first_l(self):
        """ First set for locations"""
        if not hasattr(self, "_firstL"):
            self._firstL = set([(self.unmarked(), self)])
        return self._firstL

    def last_l(self):
        """ Last set for locations"""
        if not hasattr(self, "_lastL"):
            self._lastL = set([self])
        return self._lastL

    def follow_l(self):
        """ Follow set for locations"""
        if not hasattr(self, "_followL"):
            self._followL = {self: set()}
        return self._followL

    def first(self, parent_first=None):
        """List of possible symbols matching the first symbol of a string in the language of the regular expression.

        Args:
            parent_first (list):
        Return:
            list: list of symbols"""
        if parent_first is None:
            return [self]
        parent_first.append(self)
        return parent_first

    def last(self, parent_last=None):
        """List of possible symbols matching the last symbol of a string in the language of the regular expression.

        Args:
            parent_last (list):
        Returns:
            list: list of symbols"""
        if parent_last is None:
            return [self]
        parent_last.append(self)
        return parent_last

    def followLists(self, lists=None):
        """Map of each symbol's follow list in the regular expression.

        Args:
            lists (dict):
        Returns:
            dict: map of symbols' follow lists {symbol: list of symbols}

        .. attention::
           For first() and last() return lists, the follow list for certain symbols might have repetitions in the
           case  of follow maps calculated from Star operators. The union of last(),
           first() and follow() sets are always disjoint when the regular expression is in Star normal form (
           Brüggemann-Klein, 92), therefore FAdo implements them as lists. You should order exclusively,
           or take a set from a list in order to resolve repetitions."""
        if lists is None:
            return {self: []}
        if self not in lists:
            lists[self] = []
        return lists

    def followListsD(self, lists=None):
        """Map of each symbol's follow list in the regular expression.

        Args:
            lists (dict):
        Returns:
            dict: map of symbols' follow list {symbol: list of symbols}

        .. attention::
           For first() and last() return lists, the follow list for certain symbols might have repetitions in the case
           of follow maps calculated from star operators. The union of last(), first() and follow() sets are always
           disjoint

        .. seealso:: Sabine Broda, António Machiavelo, Nelma Moreira, and Rogério Reis. On the average size of
            glushkov and partial derivative automata. International Journal of Foundations of Computer Science,
            23(5):969-984, 2012."""
        if lists is None:
            return {self: []}
        if self not in lists:
            lists[self] = []
        return lists

    def followListsStar(self, lists=None):
        """Map of each symbol's follow list in the regular expression under a star.

        Args:
            lists (dict):
        Returns:
            dict: map of symbols' follow lists {symbol: list of symbols}

         .. seealso:: Sabine Broda, António Machiavelo, Nelma Moreira, and Rogério Reis. On the average size of
            glushkov and partial derivative automata. International Journal of Foundations of Computer Science,
            23(5):969-984, 2012."""
        if lists is None:
            return {self: [self]}
        if self not in lists:
            lists[self] = [self]
        return lists

    def _marked(self, pos):
        pos += 1
        return Position((self.val, pos)), pos

    def unmarked(self):
        """The unmarked form of the regular expression. Each leaf in its syntactical tree becomes a RegExp(),
        the CEpsilon() or the CEmptySet().

        Returns:
             RegExp: (general) regular expression"""
        return self.__copy__()

    @abstractmethod
    def _follow(self, _):
        pass

    def derivative(self, sigma):
        """Derivative of the regular expression in relation to the given symbol.

        Args:
            sigma (str): an arbitrary symbol.
        Returns:
            RegExp: regular expression

        .. note:: whether the symbols belong to the expression's alphabet goes unchecked. The given symbol will be
           matched against the string representation of the regular expression's symbol.

        .. seealso:: J. A. Brzozowski, Derivatives of Regular Expressions. J. ACM 11(4): 481-494 (1964)"""
        if str(sigma) == str(self):
            return CEpsilon(self.Sigma)
        return CEmptySet(self.Sigma)

    def partialDerivatives(self, sigma):
        """Set of partial derivatives of the regular expression in relation to given symbol.

        Args:
            sigma (str): symbol in relation to which the derivative will be calculated.
        Returns:
            set: set of regular expressions

        .. seealso:: Antimirov, 95"""
        if sigma == self.val:
            return {CEpsilon(self.Sigma)}
        return set()

    def linearForm(self):
        """Linear form of the regular expression , as a mapping from heads to sets of tails, so that each pair (head,
        tail) is a monomial in the set of linear forms.

         Returns:
            dict: dictionary mapping heads to sets of tails

        .. seealso:: Antimirov, 95"""
        return {self.val: {CEpsilon(self.Sigma)}}

    def PD(self):
        """Closure of partial derivatives of the regular expression in relation to all words.

        Returns:
         set: set of regular expressions

        .. seealso:: Antimirov, 95"""
        pd = set()
        stack = [self]
        while stack:
            r = stack.pop()
            pd.add(r)
            lf = r.linearForm()
            for head in lf:
                for tail in lf[head]:
                    if tail not in pd:
                        stack.append(tail)
        return pd

    def support(self, side=True):
        """Support of a regular expression.

        Args:
            side (bool): if True concatenation of a set on the left if False on the right (prefix support)
        Returns:
            set: set of regular expressions

        .. seealso::
            Champarnaud, J.M., Ziadi, D.: From Mirkin's prebases to Antimirov's word partial derivative.
            Fundam. Inform. 45(3), 195-205 (2001)

         .. seealso::
            Maia et al, Prefix and Right-partial derivative automata, 11th CIE 2015, 258-267  LNCS 9136, 2015"""

        if side:
            return {CEpsilon(self.Sigma)}
        else:
            return {self}

    def supportlast(self, side=True):
        """ Subset of support such that elements have ewp

        Args:
            side (bool): if True left-derivatives else right-derivatives
        Returns:
            set: set of partial derivatives
        """
        if side:
            return {CEpsilon(self.Sigma)}
        else:
            return {self}

    def _memoLF(self):
        if not hasattr(self, "_lf"):
            self._lf = {self.val: {CEpsilon(self.Sigma)}}

    def tailForm(self):
        """

        Returns:
            dict: tail form
        """
        return {self.val: {CEpsilon(self.Sigma)}}

    def snf(self, hollowdot=False):
        """Star Normal Form (SNF) of the regular expression.

        Args:
            hollowdot (bool): if True computes hollow dot function else black dot
        Returns:
             RegExp: regular expression in star normal form

        .. seealso: Brüggemann-Klein, 92"""
        return self

    def nfaThompson(self):
        """Epsilon-NFA constructed with Thompson's method that accepts the regular expression's language.

        Returns:
            NFA: NFA Thompson

        .. seealso:: K. Thompson. Regular Expression Search Algorithm. CACM 11(6), 419-422 (1968)"""
        aut = fa.NFA()
        s0 = aut.addState()
        s1 = aut.addState()
        aut.setInitial([s0])
        if self.Sigma is None:
            aut.setSigma([self.val])
        else:
            aut.setSigma(self.Sigma)
        aut.setFinal([s1])  # val
        aut.addTransition(s0, self.val, s1)  # >(0)---->((1))
        return aut

    def reversal(self):
        """Reversal of RegExp

        Returns:
            Regexp:"""
        return self.__copy__()

    def partialDerivativesC(self, sigma):
        """
        Args:
            sigma (str): symbol
        Returns:
            set: set of partial derivatives"""
        if self.val == sigma:
            return {CSigmaP(self.Sigma)}
        else:
            return {CSigmaS(self.Sigma)}

    def linearFormC(self):
        """
        Returns:
            dict: linear form
        """
        lf = dict()
        for i in self.Sigma:
            if i == self.val:
                lf[i] = {CSigmaP(self.Sigma)}
            else:
                lf[i] = {CSigmaS(self.Sigma)}
        return lf


class MAtom(CAtom):
    """ Base class for pointed (marked) regular expressions

        Used directly to represent atoms (characters). This class is used to obtain Yamada or Asperti automata.
        There is no evident use for it, outside this module. """

    def __init__(self, val, mark, sigma=None):
        """
        :param val: symbol
        :type mark: bool
        :param sigma: alphabet"""
        super(MAtom, self).__init__(val, sigma)
        self.val = val
        self.mark = mark
        self.Sigma = sigma

    def __repr__(self):
        """Representation of the regular expression's syntactical tree."""
        return 'matom(%s,%s)' % (str(self.val), str(self.mark))

    def __str__(self):
        """String representation of the regular expression."""
        if self.mark:
            return "." + str(self.val)
        else:
            return str(self.val)

    _strP = __str__

    def unmark(self):
        """ Conversion back to RegExp

        :rtype: reex.RegExp"""
        return CAtom(self.val, self.Sigma)

    def _final(self):
        """ Nipkow auxiliary function final
        :rtype: bool"""
        return self.mark

    def _read(self, val):
        """ Nipkow auxiliary function final

        :param val: symbol
        :returns: the p_regexp with all but val marks removed
        :rtype: p_regexp """
        if self.val == val:
            return self
        else:
            return MAtom(self.val, False, self.Sigma)

    def _follow(self, flag):
        """ Nipkow follow function
        :type flag: bool
        :rtype: MAtom"""
        return MAtom(self.val, flag, self.Sigma)


class SpecialConstant(RegExp):
    """Base class for Epsilon and EmptySet

    .. inheritance-diagram:: SpecialConstant"""

    def __init__(self, sigma=None):
        """
        :param sigma: alphabet"""
        super(SpecialConstant, self).__init__()
        self.Sigma = sigma

    def __copy__(self):
        """
        :return: """
        return self

    @staticmethod
    def setOfSymbols():
        """
        :return: """
        return set()

    @staticmethod
    def alphabeticLength():
        """
        :return: """
        return 0

    @staticmethod
    def treeLength():
        return 1

    @staticmethod
    def starHeight():
        return 0

    @staticmethod
    def epsilonLength():
        return 0

    def snf(self):
        return self

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def rpn(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def partialDerivatives(self, _):
        pass

    def _marked(self, pos):
        """
        :param pos:
        :return: """
        return self, pos

    def mark(self):
        return self

    def unmark(self):
        """ Conversion back to unmarked atoms
        :rtype: SpecialConstant """
        return self

    @staticmethod
    def _final():
        """ Nipkow auxiliary function final
        :rtype: bool"""
        return False

    def _read(self, _):
        """ Nipkow auxiliary function final

        :returns: the p_regexp with all but val marks removed
        :rtype: p_regexp """
        return self

    def _follow(self, _):
        """ Nipkow follow function
        :rtype: SpecialConstant"""
        return self

    def unmarked(self):
        """The unmarked form of the regular expression. Each leaf in its syntactical tree becomes a RegExp(),
        the CEpsilon() or the CEmptySet().

        :rtype: (general) regular expression"""
        return self

    def reduced(self, has_epsilon=False):
        return self

    _reducedS = reduced

    def first_l(self):
        return set()

    def last_l(self):
        return set()

    def follow_l(self):
        return dict()

    @staticmethod
    def first(parent_first=None):
        """
        :param parent_first:
        :return: """
        if parent_first is None:
            return []
        return parent_first

    def last(self, parent_last=None):
        """
        :param parent_last:
        :return: """
        if parent_last is None:
            return []
        return parent_last

    def followLists(self, lists=None):
        """
        :param lists:
        :return: """
        if lists is None:
            return dict()
        return lists

    def followListsD(self, lists=None):
        """
        :param lists:
        :return: """
        if lists is None:
            return dict()
        return lists

    @staticmethod
    def followListsStar(lists=None):
        """
        :param lists:
        :return: """
        if lists is None:
            return dict()
        return lists

    def derivative(self, sigma):
        """
        :param sigma:
        :return: """
        return CEmptySet(self.Sigma)

    def wordDerivative(self, word):
        """
        :param word:
        :return: """
        return self

    def linearForm(self):
        """
        :return: """
        return dict()

    def tailForm(self):
        """
        :return: """
        return dict()

    def linearFormC(self):
        lf = dict()
        for a in self.Sigma:
            lf[a] = {CSigmaS(self.Sigma)}
        return lf

    def _memoLF(self):
        """
        :return: """
        return self._lf

    def _delAttr(self, attr):
        """

        :param attr:
        :return:"""
        pass

    _lf = dict()

    def support(self, side=True):
        """
        :return:"""
        return set()

    def supportlast(self, side=True):
        """
        :return:"""
        return set()

    def reversal(self):
        """Reversal of RegExp

        :rtype: reex.RegExp"""
        return self.__copy__()

    def partialDerivativesC(self, sigma):
        """
        :param sigma:
        :return: """
        if self.val == sigma:
            return {CSigmaP()}
        else:
            return {CSigmaS()}

    def distDerivative(self, sigma):
        """
        :param sigma: an arbitrary symbol.
        :rtype: regular expression"""
        pd = self.partialDerivatives(sigma)
        if pd == set():
            return CEmptySet()
        elif len(pd) == 1:
            der = pd.pop()
            return der
        else:
            return SDisj(pd)

    def _linearFormC(self):
        """
        :return:"""
        lf = dict()
        for sigma in self.Sigma:
            lf[sigma] = {CSigmaS(self.Sigma)}
        return lf


class CEpsilon(SpecialConstant):
    """Class that represents the empty word.

    .. inheritance-diagram:: CEpsilon"""

    def __init__(self, sigma=None):
        """Constructor of a regular expression symbol.

        :arg val: the actual symbol"""
        super(CEpsilon, self).__init__()
        self._ewp = True
        self.Sigma = sigma

    def __repr__(self):
        """
        :return: str"""
        return "CEpsilon()"

    def __str__(self):
        """
        :return: str"""
        return Epsilon

    _strP = __str__

    def rpn(self):
        """
        :return: str"""
        return Epsilon

    def __hash__(self):
        """
        :return: """
        return hash(Epsilon)

    @staticmethod
    def epsilonLength():
        return 1

    @staticmethod
    def epsilonP():
        """
        :rtype: bool"""
        return True

    @staticmethod
    def odot(sre, _):
        return sre

    @staticmethod
    def measure(from_parent=None):
        """
        :param from_parent:
        :return: measures"""
        if not from_parent:
            return [0, 1, 1, 0, 0, 0, 0, 0, 0]
        from_parent[1] += 1
        from_parent[2] += 1
        return from_parent

    def ewp(self):
        """
        :rtype: bool"""
        if hasattr(self, "_ewp"):
            self._ewp = True
        return True

    def nfaThompson(self):
        """
        :rtype: NFA """
        aut = fa.NFA()
        s0 = aut.addState()
        s1 = aut.addState()
        aut.setInitial([s0])
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        else:
            aut.setSigma([])
        aut.setFinal([s1])
        aut.addTransition(s0, Epsilon, s1)
        return aut

    def _nfaGlushkovStep(self, aut, initial, final):
        """
        :param aut:
        :param initial:
        :param final:
        :return: """
        final.update(initial)
        return initial, final

    def _nfaFollowEpsilonStep(self, conditions):
        """
        :param conditions:
        :return: """
        aut, initial, final = conditions
        aut.addTransition(initial, Epsilon, final)

    def snf(self, _hollowdot=False):
        """
        :param _hollowdot:
        :return: """
        if _hollowdot:
            return CEmptySet(self.Sigma)
        return self

    def partialDerivatives(self, _):
        """
        :return: """
        return set()

    def partialDerivativesC(self, _):
        """
        :return:"""
        return {CSigmaS(self.Sigma)}

    def _dot(self, r):
        """
        :param r:
        :return:"""
        if self.Sigma and r.Sigma:
            r.Sigma = r.Sigma | self.Sigma
        return r


class CEmptySet(SpecialConstant):
    """Class that represents the empty set.

    .. inheritance-diagram:: CEmptySet"""

    def __init__(self, sigma=None):
        """Constructor of a regular expression symbol.

        :arg val: the actual symbol"""
        super(CEmptySet, self).__init__()
        self._ewp = False
        self.Sigma = sigma

    def __repr__(self):
        """
        :return: """
        return "CEmptySet()"

    def __str__(self):
        """
        :return: """
        return EmptySet

    @staticmethod
    def odot(_, _a=None):
        return set()

    def rpn(self):
        """
        :return: """
        return EmptySet

    _strP = __str__

    def __hash__(self):
        """
        :return: """
        return hash(EmptySet)

    @staticmethod
    def emptysetP():
        """
        :return: """
        return True

    @staticmethod
    def epsilonP():
        """
        :return: """
        return False

    @staticmethod
    def measure(from_parent=None):
        """
        :param from_parent:
        :return: """
        if not from_parent:
            return [0, 1, 0, 0, 0, 0, 0, 0, 0]
        from_parent[1] += 1
        return from_parent

    @staticmethod
    def epsilonLength():
        """
        :return: """
        return 0

    def ewp(self):
        """
        :return: """
        if not hasattr(self, "_ewp"):
            self._ewp = False
        return False

    def nfaThompson(self):
        aut = fa.NFA()
        s0 = aut.addState()
        s1 = aut.addState()
        aut.setInitial([s0])
        aut.setFinal([s1])
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        return aut

    def _nfaGlushkovStep(self, aut, initial, final):
        return initial, final

    def _nfaFollowEpsilonStep(self, conditions):
        pass

    def snf(self, _hollowdot=False):
        return self

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def partialDerivatives(self, _):
        """Partial derivatives"""
        return set()

    def partialDerivativesC(self, _):
        """
        :return:"""
        return {CSigmaS(self.Sigma)}

    def _dot(self, r):
        """
        Args:
            r (RegExp):
        Returns:
            RegExp:"""
        if self.Sigma and r.Sigma:
            return CEmptySet(r.Sigma | self.Sigma)
        return self

    def _plus(self, r):
        """
        Args:
            r (RegExp):
        Returns:
            RegExp:"""
        if self.Sigma and r.Sigma:
            r.Sigma = r.Sigma | self.Sigma
        return r

    def _inter(self, r):
        """
        Args
            r (RegExp):
        Returns:
            RegExp:
            """
        if self.Sigma and r.Sigma:
            self.Sigma = r.Sigma | self.Sigma
        return self


# noinspection PyProtectedMember
class Connective(RegExp):
    """Base class for (binary) operations: concatenation, disjunction, etc

    .. inheritance-diagram:: Connective"""

    def __init__(self, arg1, arg2, sigma=None):
        super(Connective, self).__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.Sigma = sigma

    def __repr__(self):
        return "%s(%s,%s)" % (self.__class__.__name__,
                              repr(self.arg1), repr(self.arg2))

    def __copy__(self):
        return self.__class__(self.arg1.__copy__(), self.arg2.__copy__(), copy.copy(self.Sigma))

    def _setSigma(self, strict=False):
        self.arg1.setSigma(self.Sigma, strict)
        self.arg2.setSigma(self.Sigma, strict)

    @abstractmethod
    def mark(self):
        pass

    @abstractmethod
    def unmark(self):
        pass

    @abstractmethod
    def rpn(self):
        pass

    @abstractmethod
    def _follow(self, _):
        pass

    @abstractmethod
    def _memoLF(self):
        pass

    @abstractmethod
    def linearForm(self):
        pass

    @abstractmethod
    def snf(self):
        pass

    @abstractmethod
    def derivative(self, _):
        pass

    @abstractmethod
    def reduced(self):
        pass

    def unmarked(self):
        return self.__class__(self.arg1.unmarked(), self.arg2.unmarked())

    def _marked(self, pos):
        (r1, pos1) = self.arg1._marked(pos)
        (r2, pos2) = self.arg2._marked(pos1)
        return self.__class__(r1, r2), pos2

    def setOfSymbols(self):
        set_o_s = self.arg1.setOfSymbols()
        set_o_s.update(self.arg2.setOfSymbols())
        return set_o_s

    def measure(self, from_parent=None):
        if not from_parent:
            from_parent = 9 * [0]
        measure = self.arg1.measure(from_parent)
        starh, measure[3] = measure[3], 0
        measure = self.arg2.measure(measure)
        measure[1] += 1
        measure[3] = max(measure[3], starh)
        if type(self) == CDisj:
            measure[4] += 1
        elif type(self) == CConcat:
            measure[5] += 1
        elif type(self) == CConj:
            measure[7] += 1
        elif type(self) == CShuffle:
            measure[8] += 1
        return measure

    def alphabeticLength(self):
        return self.arg1.alphabeticLength() + self.arg2.alphabeticLength()

    def treeLength(self):
        return 1 + self.arg1.treeLength() + self.arg2.treeLength()

    def epsilonLength(self):
        return self.arg1.epsilonLength() + self.arg2.epsilonLength()

    def starHeight(self):
        return max(self.arg1.starHeight(), self.arg2.starHeight())

    def _cross(self, lists):
        """ Computes the pairs lastxfirst and firstxlast of the arguments

        :param lists:
        :return: pairs as a dictionary
        :rtype: dictionary"""
        for symbol in self.arg1.last():
            if symbol not in lists:
                lists[symbol] = self.arg2.first()
            else:
                lists[symbol] += self.arg2.first()
        for symbol in self.arg2.last():
            if symbol not in lists:
                lists[symbol] = self.arg1.first()
            else:
                lists[symbol] += self.arg1.first()
        return lists

    def first(self, parent_first=None):
        pass

    def last(self, parent_last=None):
        pass

    def followLists(self, lists=None):
        pass

    def followListsD(self, lists=None):
        pass

    def followListsStar(self, lists=None):
        pass


class CDisj(Connective):
    """Class for disjunction/union operation on regular expressions.

    .. inheritance-diagram:: CDisj"""

    def __str__(self):
        return "%s + %s" % (self.arg1._strP(), self.arg2._strP())

    def _strP(self):
        return "(%s + %s)" % (self.arg1._strP(), self.arg2._strP())

    def mark(self):
        """ Convertion to marked atoms
        :rtype: CDisj """
        return CDisj(self.arg1.mark(), self.arg2.mark(), self.Sigma)

    def rpn(self):
        return "+%s%s" % (self.arg1.rpn(), self.arg2.rpn())

    def ewp(self):
        if not hasattr(self, "_ewp"):
            self._ewp = self.arg1.ewp() or self.arg2.ewp()
        return self._ewp

    def unmark(self):
        """ Conversion back to unmarked atoms
        :rtype: CDisj """
        return CDisj(self.arg1.unmark(), self.arg2.unmark(), self.Sigma)

    def _final(self):
        """ Nipkow auxiliary function final
        :rtype: bool"""
        return self.arg1._final() or self.arg2._final()

    def _follow(self, flag):
        """ Nipkow follow function
        :type flag: bool
        :rtype: CDisj"""
        return CDisj(self.arg1._follow(flag), self.arg2._follow(flag), self.Sigma)

    def _read(self, val):
        """ Nipkow auxiliary function final

        :param val: symbol
        :returns: the p_regexp with all but val marks removed
        :rtype: CDisj """
        return CDisj(self.arg1._read(val), self.arg2._read(val), self.Sigma)

    def first_l(self):
        """ First sets for locations"""
        if not hasattr(self, "_firstL"):
            self._firstL = self.arg1.first_l() | self.arg2.first_l()
        return self._firstL

    def last_l(self):
        """ Last sets for locations"""
        if not hasattr(self, "_lastL"):
            self._lastL = self.arg1.last_l() | self.arg2.last_l()
        return self._lastL

    def follow_l(self):
        """ Follow sets for locations"""
        if not hasattr(self, "_followL"):
            self._followL =  self.arg1.follow_l() | self.arg2.follow_l()
        return self._followL

    def first(self, parent_first=None):
        parent_first = self.arg1.first(parent_first)
        return self.arg2.first(parent_first)

    def last(self, parent_last=None):
        parent_last = self.arg1.last(parent_last)
        return self.arg2.last(parent_last)

    def followLists(self, lists=None):
        if lists is None:
            lists = dict()
        self.arg1.followLists(lists)
        return self.arg2.followLists(lists)

    def followListsD(self, lists=None):
        if lists is None:
            lists = dict()
        self.arg1.followListsD(lists)
        return self.arg2.followListsD(lists)

    def followListsStar(self, lists=None):
        if lists is None:
            lists = dict()
        self.arg1.followListsStar(lists)
        self.arg2.followListsStar(lists)
        return self._cross(lists)

    def reduced(self, has_epsilon=False):
        left = self.arg1.reduced(has_epsilon or self.arg2.ewp())
        right = self.arg2.reduced(has_epsilon or left.ewp())
        if left.emptysetP():
            return right
        if right.emptysetP():
            return left
        if left.epsilonP() and (has_epsilon or right.ewp()):
            return right
        if right.epsilonP() and (has_epsilon or left.ewp()):
            return left
        if left is self.arg1 and right is self.arg2:
            return self
        reduced = CDisj(left, right, self.Sigma)
        reduced._reduced = True
        return reduced

    _reducedS = reduced

    def derivative(self, sigma):
        left = self.arg1.derivative(sigma)
        right = self.arg2.derivative(sigma)
        return CDisj(left, right, self.Sigma)

    def partialDerivatives(self, sigma):
        pdset = self.arg1.partialDerivatives(sigma)
        pdset.update(self.arg2.partialDerivatives(sigma))
        return pdset

    def linearForm(self):
        arg1_lf = self.arg1.linearForm()
        arg2_lf = self.arg2.linearForm()
        lf = dict()
        for head in set(list(arg1_lf.keys()) + list(arg2_lf.keys())):
            tails = arg1_lf.get(head, set()) | arg2_lf.get(head, set())
            if tails != set():
                lf[head] = tails
        return lf

    def support(self, side=True):
        p = self.arg1.support(side)
        p.update(self.arg2.support(side))
        return p

    def supportlast(self, side=True):
        p = self.arg1.supportlast(side)
        p.update(self.arg2.supportlast(side))
        return p

    def _delAttr(self, attr):
        if hasattr(self, attr):
            delattr(self, attr)
            self.arg1._delAttr(attr)
            self.arg2._delAttr(attr)

    def _memoLF(self):
        if hasattr(self, "_lf"):
            return
        self.arg1._memoLF()
        self.arg2._memoLF()
        self._lf = dict()
        for head in self.arg1._lf:
            self._lf[head] = set(self.arg1._lf[head])
        for head in self.arg2._lf:
            try:
                self._lf[head].update(self.arg2._lf[head])
            except KeyError:
                self._lf[head] = set(self.arg2._lf[head])

    def tailForm(self):
        arg1 = self.arg1.tailForm()
        arg2 = self.arg2.tailForm()
        tf = dict()
        for head in set(list(arg1.keys()) + list(arg2.keys())):
            tails = arg1.get(head, set()) | arg2.get(head, set())
            if tails != set():
                tf[head] = tails
        return tf

    def snf(self, hollowdot=False):
        return CDisj(self.arg1.snf(hollowdot), self.arg2.snf(hollowdot), self.Sigma)

    def nfaThompson(self):
        """ Returns an NFA (Thompson) that accepts the RE.

    :rtype: NFA

    .. graphviz::

       digraph dij {
        "0" -> "si1" [label=e];
        "si1" -> "sf1" [label="arg1"];
        "sf1" -> "1" [label=e];
        "0" -> "si2" [label=e];
        "si2" -> "sf2" [label="arg2"];
        "sf2" -> "1" [label=e];
        }"""
        au = fa.NFA()
        if self.Sigma is not None:
            au.setSigma(self.Sigma)
        s0, s1 = au.addState(), au.addState()
        au.setInitial([s0])
        au.setFinal([s1])
        si1, sf1 = au._inc(self.arg1.nfaThompson())
        au.addTransition(s0, Epsilon, si1)
        au.addTransition(sf1, Epsilon, s1)
        si2, sf2 = au._inc(self.arg2.nfaThompson())
        au.addTransition(s0, Epsilon, si2)
        au.addTransition(sf2, Epsilon, s1)
        return au

    def _nfaGlushkovStep(self, aut, initial, final):
        _, new_final = self.arg1._nfaGlushkovStep(aut, initial, set(final))
        _, final = self.arg2._nfaGlushkovStep(aut, initial, final)
        final.update(new_final)
        return initial, final

    def _nfaFollowEpsilonStep(self, conditions):
        self.arg1._nfaFollowEpsilonStep(conditions)
        self.arg2._nfaFollowEpsilonStep(conditions)

    def reversal(self):
        """ Reversal of RegExp

        :rtype: reex.RegExp"""
        return CDisj(self.arg1.reversal(), self.arg2.reversal(), sigma=self.Sigma)


class Unary(RegExp):
    """
    Base class for unary operations: star, option, not, unary shuffle, etc

    .. inheritance-diagram:: Unary"""

    def __init__(self, arg, sigma=None):
        super().__init__()
        self.arg = arg
        self.Sigma = sigma

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                    repr(self.arg))

    def __copy__(self):
        # copy.copy(self.arg)
        return self.__class__(self.arg.__copy__(), copy.copy(self.Sigma))

    def _delAttr(self, attr):
        if hasattr(self, attr):
            delattr(self, attr)
            self.arg._delAttr(attr)

    def setOfSymbols(self):
        return self.arg.setOfSymbols()

    def _setSigma(self, strict=False):
        self.arg.setSigma(self.Sigma, strict)

    def alphabeticLength(self):
        return self.arg.alphabeticLength()

    def treeLength(self):
        return 1 + self.arg.treeLength()

    @abstractmethod
    def starHeight(self):
        pass

    @abstractmethod
    def epsilonLength(self):
        pass

    def mark(self):
        return self.__class__(self.arg.mark(), self.Sigma)

    def unmark(self):
        """ Conversion back to RegExp

            Returns:
                reex.__class__: """
        return self.__class__(self.arg.unmark(), self.Sigma)

    @abstractmethod
    def rpn(self):
        pass

    def _follow(self, _):
        pass

    def _memoLF(self):
        pass

    @abstractmethod
    def linearForm(self):
        pass

    def snf(self):
        pass

    @abstractmethod
    def derivative(self, _):
        pass

    @abstractmethod
    def partialDerivatives(self, _):
        pass

    @abstractmethod
    def reduced(self):
        pass

    def unmarked(self):
        return self.__class__(self.arg.unmarked())

    def _marked(self, pos):
        (r1, pos1) = self.arg._marked(pos)
        return self.__class__(r1), pos1

    @abstractmethod
    def first(self, parent_first=None):
        pass

    @abstractmethod
    def last(self, parent_last=None):
        pass

    def followLists(self, lists=None):
        pass

    def followListsD(self, lists=None):
        pass

    def followListsStar(self, lists=None):
        pass

    @abstractmethod
    def ewp(self):
        pass

    def reversal(self):
        """Reversal of RegEx

        :rtype: reex.RegExp"""
        return self.__class__(self.arg.reversal(), sigma=self.Sigma)


class Compl(Unary):
    """Class for not operation  on regular expressions.

    .. inheritance-diagram:: Compl"""

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def supportlast(self, side=True):
        raise FAdoNotImplemented()

    def support(self, side=True):
        raise FAdoNotImplemented()

    def tailForm(self):
        raise FAdoNotImplemented()

    def __str__(self):
        return "{0:s}({1:s})".format(Not, self.arg._strP())

    _strP = __str__

    def ewp(self):
        if hasattr(self, "_ewp"):
            return self._ewp
        return not self.arg.ewp()

    def mark(self):
        pass

    def snf(self):
        pass

    def rpn(self):
        pass

    def derivative(self, _):
        pass

    def _follow(self, _):
        pass

    def unmark(self):
        pass

    def linearForm(self):
        pass

    def followListsD(self):
        pass

    def first(self, _):
        pass

    def followLists(self):
        pass

    def followListsStar(self):
        pass

    def starHeight(self):
        pass

    def epsilonLength(self):
        pass

    def reduced(self):
        pass

    def _marked(self, _):
        pass

    def last(self):
        pass

    def treeLength(self):
        pass

    def _memoLF(self):
        pass


class Power(RegExp):
    """Class for Power operation  on regular expressions.

    .. inheritance-diagram:: Power"""

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def supportlast(self, side=True):
        raise FAdoNotImplemented()

    def support(self, side=True):
        raise FAdoNotImplemented()

    def tailForm(self):
        raise FAdoNotImplemented()

    def __init__(self, arg, n=1, sigma=None):
        super(Power, self).__init__()
        self.arg = arg
        self.pw = n
        self.Sigma = sigma

    def __str__(self):
        return "%s^(%s)" % (self.arg._strP(), self.pw)

    _strP = __str__

    def __repr__(self):
        return "Power(%s,%s)" % (repr(self.arg), repr(self.pw))

    def __copy__(self):
        return Power(copy.copy(self.arg), self.pw, copy.copy(self.Sigma))

    def setOfSymbols(self):
        return self.arg.setOfSymbols()

    def reversal(self):
        """ Reversal of RegExp

        :rtype: reex.RegExp"""
        return Power(self.arg.reversal(), self.pw, self.Sigma)

    def mark(self):
        pass

    def snf(self):
        pass

    def rpn(self):
        pass

    def alphabeticLength(self):
        pass

    def derivative(self, _):
        pass

    def _follow(self, _):
        pass

    def unmark(self):
        pass

    def linearForm(self):
        pass

    def followListsD(self):
        pass

    def first(self):
        pass

    def followLists(self):
        pass

    def starHeight(self):
        pass

    def epsilonLength(self):
        pass

    def reduced(self):
        pass

    def _marked(self, _):
        pass

    def last(self):
        pass

    def treeLength(self):
        pass

    def _memoLF(self):
        pass


class COption(Unary):
    """Class for option operation  (reg + @epsilon)  on regular expressions.

    .. inheritance-diagram:: COption"""

    def tailForm(self):
        raise FAdoNotImplemented()

    def __str__(self):
        return "({0:s})?".format(self.arg._strP())

    _strP = __str__

    def setOfSymbols(self):
        return self.arg.setOfSymbols()

    def snf(self, _hollowdot=False):
        if not _hollowdot:
            if self.arg.ewp():
                return self.arg.snf()
            return COption(self.arg.snf(), self.Sigma)
        return self.arg.snf(True)  # COption(self.arg.snf(True),self.sigma)

    def rpn(self):
        return "?{0:s}".format(self.arg.rpn())

    def derivative(self, sigma):
        return self.arg.derivative(sigma)

    def _follow(self, _):
        pass

    def linearForm(self):
        return self.arg.linearForm()

    def first_l(self):
        """ First sets for locations"""
        if not hasattr(self, "_firstL"):
            self._firstL = self.arg.first_l().copy()
        return self._firstL

    def last_l(self):
        """ Last sets for locations"""
        if not hasattr(self, "_lastL"):
            self._lastL = self.arg.last_l().copy()
        return self._lastL

    def follow_l(self):
        """ Follow sets for locations"""
        if not hasattr(self, "_followL"):
            self._followL = self.arg.follow_l().copy()
        return self._followL

    def first(self, parent_first=None):
        return self.arg.first(parent_first)

    def followLists(self, lists=None):
        return self.arg.followLists(lists)

    def followListsD(self, lists=None):
        return self.arg.followLists(lists)

    def followListsStar(self, lists=None):
        """to be fixed """
        return self.arg.followListsStar(lists)

    def starHeight(self):
        return self.arg.starHeight()

    def epsilonLength(self):
        return 1 + self.arg.epsilonLength()

    def reduced(self):
        return COption(self.arg.reduced())

    def last(self, parent_first=None):
        return self.arg.last(parent_first)

    def _memoLF(self):
        if hasattr(self, "_lf"):
            return
        self.arg._memoLF()
        self._lf = self.arg._lf

    def partialDerivatives(self, sigma):
        return self.arg.partialDerivatives(sigma)

    def support(self, side=True):
        return self.arg.support(side)

    def supportlast(self, side=True):
        return self.arg.supportlast(side)

    def ewp(self):
        if hasattr(self, "_ewp"):
            return self._ewp
        self._ewp = True
        return True

    def nfaThompson(self):
        """ Returns a NFA that accepts the RE.

    :rtype: NFA

    .. graphviz::

       digraph foo {
        "0" -> "1" [label=e];
        "0" -> "a" [label=e];
        "a" -> "b" [label=A];
        "b" -> "1" [label=e];
        }"""

        sun = self.arg.nfaThompson()
        au = sun.dup()
        (s0, s1) = (au.addState(), au.addState())
        if self.Sigma is not None:
            au.setSigma(self.Sigma)
        au_initial = au.Initial.pop()
        au.addTransition(s0, Epsilon, s1)
        au.addTransition(s0, Epsilon, au_initial)
        au.addTransition(list(au.Final)[0], Epsilon, s1)  # we know by contruction
        au.setInitial([s0])  # that there is only one final state,
        au.setFinal([s1])  # and only one initial state
        return au

    def _nfaGlushkovStep(self, aut, initial, final):
        _, final = self.arg._nfaGlushkovStep(aut, initial, final)
        final.update(initial)
        return initial, final

    def _nfaFollowEpsilonStep(self, conditions):
        aut, initial, final = conditions
        self.arg._nfaFollowEpsilonStep((aut, initial, final))
        if aut.hasTransitionP(initial, Epsilon, final):
            return
        aut.addTransition(initial, Epsilon, final)


class CStar(Unary):
    """Class for iteration operation (aka Kleene star, or Kleene closure) on regular expressions.

    .. inheritance-diagram:: CStar"""

    def __str__(self):
        return "%s*" % self.arg._strP()

    _strP = __str__

    def _final(self):
        """ Nipkow auxiliary function final

        :rtype: bool"""
        return self.arg._final()

    def _follow(self, flag):
        """ Nipkow follow function

        :type flag: bool
        :rtype: CEmptySet"""
        return CStar(self.arg._follow(self.arg._final() or flag), self.Sigma)

    def _read(self, val):
        """ Nipkow auxiliary function final

        :param val: symbol
        :returns: the p_regexp with all but val marks removed
        :rtype: CStar """
        return CStar(self.arg._read(val), self.Sigma)

    def rpn(self):
        return "*%s" % self.arg.rpn()

    def measure(self, from_parent=None):
        if not from_parent:
            from_parent = 9 * [0]
        measure = self.arg.measure(from_parent)
        measure[1] += 1
        measure[3] += 1
        measure[6] += 1
        return measure

    def epsilonLength(self):
        return self.arg.epsilonLength()

    def starHeight(self):
        return 1 + self.arg.starHeight()

    def first_l(self):
        """ First sets for locations"""
        if not hasattr(self, "_firstL"):
            self._firstL = self.arg.first_l().copy()
        return self._firstL

    def last_l(self):
        """ Last sets for locations"""
        if not hasattr(self, "_lastL"):
            self._lastL = self.arg.last_l().copy()
        return self._lastL

    def follow_l(self):
        """ Follow sets for locations"""
        if not hasattr(self, "_followL"):
            fol = self.arg.follow_l()
            for s in self.arg.last_l():
                fol[s] = fol.get(s, set()).union(self.arg.first_l())
            self._followL = fol
        return self._followL

    def first(self, parent_first=None):
        return self.arg.first(parent_first)

    def last(self, parent_first=None):
        return self.arg.last(parent_first)

    def followLists(self, lists=None):
        lists = self.arg.followLists(lists)
        for symbol in self.arg.last():
            if symbol not in lists:
                lists[symbol] = self.arg.first()
            else:
                lists[symbol] += self.arg.first()
        return lists

    def followListsD(self, lists=None):
        return self.arg.followListsStar(lists)

    def followListsStar(self, lists=None):
        return self.arg.followListsStar(lists)

    def unmarked(self):
        return CStar(self.arg.unmarked())

    def _marked(self, pos):
        (r1, pos1) = self.arg._marked(pos)
        return CStar(r1), pos1

    def reduced(self, has_epsilon=False):
        rarg = self.arg._reducedS(True)
        if rarg.epsilonP() or rarg.emptysetP():
            return CEpsilon(self.Sigma)
        if self.arg is rarg:
            return self
        reduced = CStar(rarg, self.Sigma)
        return reduced

    # noinspection PyUnusedLocal
    def _reducedS(self, has_epsilon=False):
        return self.arg._reducedS(True)

    def derivative(self, sigma):
        d = self.arg.derivative(sigma)
        return CConcat(d, self, self.Sigma)

    def partialDerivatives(self, sigma):
        arg_pdset = self.arg.partialDerivatives(sigma)
        pds = set()
        for pd in arg_pdset:
            if pd.emptysetP():
                pds.add(CEmptySet(self.Sigma))
            elif pd.epsilonP():
                pds.add(self)
            else:
                pds.add(CConcat(pd, self, self.Sigma))
        return pds

    def linearForm(self):
        arg_lf = self.arg.linearForm()
        lf = dict()
        for head in arg_lf:
            lf[head] = set()
            for tail in arg_lf[head]:
                if tail.emptysetP():
                    lf[head].add(CEmptySet(self.Sigma))
                elif tail.epsilonP():
                    lf[head].add(self)
                else:
                    lf[head].add(CConcat(tail, self, self.Sigma))
        return lf

    def support(self, side=True):
        return self._odot(self.arg.support(side), side)

    def supportlast(self, side=True):
        return self._odot(self.arg.supportlast(side), side)

    def _memoLF(self):
        if hasattr(self, "_lf"):
            return
        self.arg._memoLF()
        self._lf = dict()
        for head in self.arg._lf:
            pd_set = set()
            self._lf[head] = pd_set
            for tail in self.arg._lf[head]:
                if tail.emptysetP():
                    pd_set.add(CEmptySet(self.Sigma))
                elif tail.epsilonP():
                    pd_set.add(self)
                else:
                    pd_set.add(CConcat(tail, self, self.Sigma))

    def ewp(self):
        if hasattr(self, "_ewp"):
            self._ewp = True
        return True

    def tailForm(self):
        arg_tf = self.arg.tailForm()
        tf = dict()
        for tail in arg_tf:
            tf[tail] = set()
            for head in arg_tf[tail]:
                if head.emptysetP():
                    tf[tail].add((CEmptySet(self.Sigma), CEmptySet(self.Sigma)))
                elif head.epsilonP():
                    tf[tail].add(self)
                else:
                    new = _ifconcat(self, head, both=True, sigma=self.Sigma)
                    tf[tail].add(new)
        return tf

    def snf(self, _hollowdot=False):
        if _hollowdot:
            return self.arg.snf(True)
        return CStar(self.arg.snf(True), self.Sigma)

    def nfaThompson(self):
        """ Returns a NFA that accepts the RE.

    :rtype: NFA

    .. graphviz::

       digraph foo {
        "0" -> "1" [label=e];
        "0" -> "a" [label=e];
        "a" -> "b" [label=A];
        "b" -> "1" [label=e];
        "1" -> "0" [label=e];
        }"""

        sun = self.arg.nfaThompson()
        au = sun.dup()
        (s0, s1) = (au.addState(), au.addState())
        if self.Sigma is not None:
            au.setSigma(self.Sigma)
        au_initial = au.Initial.pop()
        au.addTransition(s0, Epsilon, s1)
        au.addTransition(s1, Epsilon, s0)
        #        au.addTransition(list(au.Final)[0], Epsilon, au_initial)
        au.addTransition(s0, Epsilon, au_initial)
        au.addTransition(list(au.Final)[0], Epsilon, s1)  # we know by contruction
        au.setInitial([s0])  # that there is only one final state,
        au.setFinal([s1])  # and only one initial state
        return au

    def _nfaGlushkovStep(self, aut, initial, final):
        previous_trans = dict()
        for i_state in initial:
            if i_state in aut.delta:
                previous_trans[i_state] = aut.delta[i_state]
                del aut.delta[i_state]
        new_initial, final = self.arg._nfaGlushkovStep(aut, initial, final)
        for i_state in initial:
            if i_state in aut.delta:
                for symbol in aut.delta[i_state]:
                    for target in aut.delta[i_state][symbol]:
                        for f_state in final:
                            aut.addTransition(f_state, symbol, target)
        for i_state in previous_trans:
            for sym in previous_trans[i_state]:
                for target in previous_trans[i_state][sym]:
                    aut.addTransition(i_state, sym, target)
        final.update(initial)
        return new_initial, final

    def _nfaFollowEpsilonStep(self, conditions):
        aut, initial, final = conditions
        if initial is final:
            iter_state = final
        else:
            iter_state = aut.addState()
        self.arg._nfaFollowEpsilonStep((aut, iter_state, iter_state))
        tomerge = aut.epsilonPaths(iter_state, iter_state)
        aut.mergeStatesSet(tomerge)
        if initial is not final:
            aut.addTransition(initial, Epsilon, iter_state)
            aut.addTransition(iter_state, Epsilon, final)


class CConcat(Connective):
    """Class for catenation operation on regular expressions.

    .. inheritance-diagram:: CConcat"""

    def __str__(self):
        return "%s %s" % (self.arg1._strP(), self.arg2._strP())

    def _strP(self):
        return "(%s %s)" % (self.arg1._strP(), self.arg2._strP())

    def mark(self):
        return CConcat(self.arg1.mark(), self.arg2.mark(), self.Sigma)

    def rpn(self):
        """
        :rtype: str"""
        return ".%s%s" % (self.arg1.rpn(), self.arg2.rpn())

    def ewp(self):
        if not hasattr(self, "_ewp"):
            self._ewp = self.arg1.ewp() and self.arg2.ewp()
        return self._ewp

    def first_l(self):
        """ First sets for locations"""
        if not hasattr(self, "_firstL"):
            a = self.arg1.first_l()
            if self.arg1.ewp():
                a |= self.arg2.first_l()
            self._firstL = a
        return self._firstL

    def last_l(self):
        """ Last sets for locations"""
        if not hasattr(self, "_lastL"):
            a = self.arg2.last_l()
            if self.arg2.ewp():
                a |= self.arg1.last_l()
            self._lastL = a
        return self._lastL

    def follow_l(self):
        """ Follow sets for locations"""
        if not hasattr(self, "_followL"):
            lists = self.arg1.follow_l() | self.arg2.follow_l()
            for symbol in self.arg1.last_l():
                lists[symbol] = lists.get(symbol, set()).union(self.arg2.first_l())
            self._followL = lists
        return self._followL

    def first(self, parent_first=None):
        if self.arg1.ewp():
            return self.arg2.first(self.arg1.first(parent_first))
        else:
            return self.arg1.first(parent_first)

    def last(self, parent_last=None):
        if self.arg2.ewp():
            return self.arg2.last(self.arg1.last(parent_last))
        else:
            return self.arg2.last(parent_last)

    def followLists(self, lists=None):
        lists = self.arg1.followLists(lists)
        self.arg2.followLists(lists)
        for symbol in self.arg1.last():
            if symbol not in lists:
                lists[symbol] = self.arg2.first()
            else:
                lists[symbol] += self.arg2.first()
        return lists

    def followListsD(self, lists=None):
        lists = self.arg1.followListsD(lists)
        self.arg2.followListsD(lists)
        for symbol in self.arg1.last():
            if symbol not in lists:
                lists[symbol] = self.arg2.first()
            else:
                lists[symbol] += self.arg2.first()
        return lists

    def followListsStar(self, lists=None):
        if lists is None:
            lists = dict()
        if self.arg1.ewp():
            if self.arg2.ewp():
                self.arg1.followListsStar(lists)
                self.arg2.followListsStar(lists)
            else:
                self.arg1.followListsD(lists)
                self.arg2.followListsStar(lists)
        elif self.arg2.ewp():
            self.arg1.followListsStar(lists)
            self.arg2.followListsD(lists)
        else:
            self.arg1.followListsD(lists)
            self.arg2.followListsD(lists)
        return self._cross(lists)

    def reduced(self, has_epsilon=False):
        left = self.arg1.reduced()
        right = self.arg2.reduced()
        if left.emptysetP() or right.emptysetP():
            return CEmptySet(self.Sigma)
        if left.epsilonP():
            if has_epsilon:
                return self.arg2.reduced(True)
            return right
        if right.epsilonP():
            if has_epsilon:
                return self.arg1.reduced(True)
            return left
        if left is self.arg1 and right is self.arg2:
            return self
        reduced = CConcat(left, right, self.Sigma)
        return reduced

    _reducedS = reduced

    def derivative(self, sigma):
        left = CConcat(self.arg1.derivative(sigma), self.arg2, self.Sigma)
        if self.arg1.ewp():
            right = self.arg2.derivative(sigma)
            return CDisj(left, right, self.Sigma)
        return left

    def partialDerivatives(self, sigma):
        pds = set()
        for pd in self.arg1.partialDerivatives(sigma):
            if pd.emptysetP():
                pds.add(CEmptySet(self.Sigma))
            elif pd.epsilonP():
                pds.add(self.arg2)
            else:
                pds.add(CConcat(pd, self.arg2, self.Sigma))
        if self.arg1.ewp():
            pds.update(self.arg2.partialDerivatives(sigma))
        return pds

    def linearForm(self):
        arg1_lf = self.arg1.linearForm()
        lf = dict()
        for head in arg1_lf:
            lf[head] = set()
            for tail in arg1_lf[head]:
                if tail.emptysetP():
                    lf[head].add(CEmptySet(self.Sigma))
                elif tail.epsilonP():
                    lf[head].add(self.arg2)
                else:
                    lf[head].add(CConcat(tail, self.arg2, self.Sigma))
        if self.arg1.ewp():
            arg2_lf = self.arg2.linearForm()
            for head in arg2_lf:
                if head in lf:
                    lf[head].update(arg2_lf[head])
                else:
                    lf[head] = set(arg2_lf[head])
        return lf

    def _memoLF(self):
        if hasattr(self, "_lf"):
            return
        self.arg1._memoLF()
        self._lf = dict()
        for head in self.arg1._lf:
            pd_set = set()
            self._lf[head] = pd_set
            for tail in self.arg1._lf[head]:
                if tail.emptysetP():
                    pd_set.add(CEmptySet(self.Sigma))
                elif tail.epsilonP():
                    pd_set.add(self.arg2)
                else:
                    pd_set.add(CConcat(tail, self.arg2, self.Sigma))
        if self.arg1.ewp():
            self.arg2._memoLF()
            for head in self.arg2._lf:
                if head in self._lf:
                    self._lf[head].update(self.arg2._lf[head])
                else:
                    self._lf[head] = set(self.arg2._lf[head])

    def support(self, side=True):
        if side:
            arg2 = self.arg2
            arg1 = self.arg1
        else:
            arg2 = self.arg1
            arg1 = self.arg2
        p = arg2._odot(arg1.support(side), side)
        p.update(arg2.support(side))
        return p

    def supportlast(self, side=True):
        if side:
            arg2 = self.arg2
            arg1 = self.arg1
        else:
            arg2 = self.arg1
            arg1 = self.arg2
        p = arg2.supportlast(side)
        if arg2.ewp():
            p.update(arg2._odot(arg1.supportlast(side), side))
        return p

    def tailForm(self):
        arg2_tf = self.arg2.tailForm()
        tf = dict()
        for tail in arg2_tf:
            tf[tail] = set()
            for head in arg2_tf[tail]:
                new = _ifconcat(self.arg1, head, both=True, sigma=self.Sigma)
                tf[tail].add(new)
        if self.arg2.ewp():
            arg1_tf = self.arg1.tailForm()
            for tail in arg1_tf:
                tf.setdefault(tail, set()).update(arg1_tf[tail])
        return tf

    def snf(self, _hollowdot=False):
        if not _hollowdot:
            return CConcat(self.arg1.snf(), self.arg2.snf(), self.Sigma)
        if self.ewp():
            return CDisj(self.arg1.snf(True), self.arg2.snf(True), self.Sigma)
        if self.arg1.ewp():
            return CConcat(self.arg1.snf(), self.arg2.snf(True), self.Sigma)
        if self.arg2.ewp():
            return CConcat(self.arg1.snf(True), self.arg2.snf(), self.Sigma)
        return CConcat(self.arg1.snf(), self.arg2.snf(), self.Sigma)

    def nfaThompson(self):  # >(0)--arg1-->(1)--->(2)--arg2-->((3))
        au = fa.NFA()
        if self.Sigma is not None:
            au.setSigma(self.Sigma)
        s0, s1 = au._inc(self.arg1.nfaThompson())
        s2, s3 = au._inc(self.arg2.nfaThompson())
        au.setInitial([s0])
        au.setFinal([s3])
        au.addTransition(s1, Epsilon, s2)
        return au

    def _nfaGlushkovStep(self, aut, initial, final):
        initial, final = self.arg1._nfaGlushkovStep(aut, initial, final)
        return self.arg2._nfaGlushkovStep(aut, final, set())

    def _nfaFollowEpsilonStep(self, conditions):
        aut, initial, final = conditions
        interm = aut.addState()
        self.arg1._nfaFollowEpsilonStep((aut, initial, interm))
        # At this stage, if the intermediate state has a single
        # incoming transition, and it's through Epsilon, then the
        # source becomes the new intermediate state:
        new_interm = aut.unlinkSoleIncoming(interm)
        if new_interm is not None:
            interm = new_interm
        self.arg2._nfaFollowEpsilonStep((aut, interm, final))
        # At this stage, if the intermediate state has a single
        # outgoing transition, and it's through Epsilon, then we merge
        # it with the target.
        if aut.hasTransitionP(interm, Epsilon, final):
            return
        target = aut.unlinkSoleOutgoing(interm)
        if target is not None:
            aut.mergeStates(target, interm)

    def reversal(self):
        """Reversal of RegExp

        :rtype: reex.RegExp"""
        return CConcat(self.arg2.reversal(), self.arg1.reversal(), sigma=self.Sigma)

    def unmark(self):
        """ Conversion back to unmarked atoms
        :rtype: CConcat """
        return CConcat(self.arg1.unmark, self.arg2.unmark, self.Sigma)

    def _final(self):
        """ Nipkow auxiliary function final
        :rtype: bool"""
        return self.arg2._final() or (self.arg2.ewp() and self.arg1._final())

    def _follow(self, flag):
        """ Nipkow follow function
        :type flag: bool
        :rtype: CEmptySet"""
        return CConcat(self.arg1._follow(flag), self.arg2._follow(self.arg1._final() or (flag and self.arg1.ewp())),
                       self.Sigma)

    def _read(self, val):
        """ Nipkow auxiliary function final

        :param val: symbol
        :returns: the p_regexp with all but val marks removed
        :rtype: CConcat """
        return CConcat(self.arg1._read(val), self.arg2._read(val), self.Sigma)


class CShuffle(Connective):
    """Shuffle operation of regexps
    """

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def __str__(self):
        return "{1:s} {0:s} {2:s}".format(Shuffle, self.arg1._strP(), self.arg2._strP())

    def _strP(self):
        return "({1:s} {0:s}  {2:s})".format(Shuffle, self.arg1._strP(), self.arg2._strP())

    def rpn(self):
        return "{0:s}{1:s}{2:s}".format(Shuffle, self.arg1.rpn(), self.arg2.rpn())

    def ewp(self):
        if hasattr(self, "_ewp"):
            return self._ewp
        self._ewp = self.arg1.ewp() and self.arg2.ewp()
        return self._ewp

    def reduced(self, has_epsilon=False):
        left = self.arg1.reduced()
        right = self.arg2.reduced()
        if left.emptysetP() or right.emptysetP():
            return CEmptySet(self.Sigma)
        if left.epsilonP():
            if has_epsilon:
                return self.arg2.reduced(True)
            return right
        if right.epsilonP():
            if has_epsilon:
                return self.arg1.reduced(True)
            return left
        if left is self.arg1 and right is self.arg2:
            return self
        reduced = CShuffle(left, right, self.Sigma)
        #        reduced._reduced = True
        return reduced

    _reducedS = reduced

    def _memoLF(self):
        raise regexpInvalidMethod()

    def snf(self):
        raise regexpInvalidMethod()

    def mark(self):
        raise regexpInvalidMethod()

    def unmark(self):
        raise regexpInvalidMethod()

    def _follow(self, _):
        raise regexpInvalidMethod()

    def derivative(self, sigma):
        return CDisj(CShuffle(self.arg1.derivative(sigma), self.arg2, self.Sigma),
                     CShuffle(self.arg1, self.arg2.derivative(sigma), self.Sigma), self.Sigma)

    def partialDerivatives(self, sigma):
        pdset = self.arg1.partialDerivatives(sigma)
        a1 = {_dotshuffle(a, self.arg2, self.Sigma) for a in pdset}
        pd2set = self.arg2.partialDerivatives(sigma)
        a1.update({_dotshuffle(self.arg1, a, self.Sigma) for a in pd2set})
        return a1

    def support(self, side=True):
        """ """
        p = self.arg1.support(side)
        q = self.arg2.support(side)
        p0 = _odotshuffle(p, q, self.Sigma)
        p0.update(_odotshuffle({self.arg1}, q, self.Sigma))
        p0.update(_odotshuffle(p, {self.arg2}, self.Sigma))
        return p0

    def supportlast(self, side=True):
        """ """
        p = self.arg1.supportlast()
        q = self.arg2.supportlast()
        p0 = _odotshuffle(p, q, self.Sigma)
        if self.arg1.ewp():
            p0.update(_odotshuffle({self.arg1}, q, self.Sigma))
        if self.arg2.ewp():
            p0.update(_odotshuffle(p, {self.arg2}, self.Sigma))
        return p0

    def linearForm(self):
        arg1_lf = self.arg1.linearForm()
        arg2_lf = self.arg2.linearForm()
        lf = dict()
        for head in arg1_lf:
            lf[head] = set()
            for tail in arg1_lf[head]:
                lf[head].add(_dotshuffle(tail, self.arg2, self.Sigma))
        for head in arg2_lf:
            if head in lf:
                lf[head].update({_dotshuffle(self.arg1, tail, self.Sigma) for tail in arg2_lf[head]})
            else:
                lf[head] = set()
                for tail in arg2_lf[head]:
                    lf[head].add(_dotshuffle(self.arg1, tail, self.Sigma))
        return lf

    def tailForm(self):
        arg2_tf = self.arg2.tailForm()
        arg1_tf = self.arg1.tailForm()
        tf = dict()
        for tail in arg1_tf:
            tf[tail] = set()
            for head in arg1_tf[tail]:
                tf[tail].add(_dotshuffle(head, self.arg2, self.Sigma))
        for tail in arg2_tf:
            for head in arg2_tf[tail]:
                new = _dotshuffle(self.arg1, head, self.Sigma)
                tf.setdefault(tail, set()).add(new)
        return tf

    def reversal(self):
        return CShuffle(self.arg1.reversal(), self.arg2.reversal(), sigma=self.Sigma)

    def first_l(self):
        """ First sets for locations"""
        if not hasattr(self, "_firstL"):
            first = self.arg1.first_l()
            sec = self.arg2.first_l()
            inf = {(s, (p, 0)) for (s, p) in first}
            inf.update({(s, (0, p)) for (s, p) in sec})
            self._firstL = inf
        return self._firstL

    def last_l(self):
        """ Last sets for locations"""
        if not hasattr(self, "_lastL"):
            first = self.arg1.last_l()
            sec = self.arg2.last_l()
            lsh = {(p, p1) for p in first for p1 in sec}
            if self.arg1.ewp():
                lsh.update({(0, p) for p in sec})
            if self.arg2.ewp():
                lsh.update({(p, 0) for p in first})
            self._lastL = lsh
        return self._lastL

    def follow_l(self):
        """ Follow sets for locations"""
        if not hasattr(self, "_followL"):
            lists = self.arg1.follow_l()
            sec = self.arg2.follow_l()
            first = self.arg1.first_l()
            fsec = self.arg2.first_l()
            lfol = {}
            for p1 in lists:
                f0 = {(s, (r, 0)) for (s, r) in lists[p1]} | {(s, (p1, r)) for (s, r) in fsec}
                lfol[(p1, 0)] = lfol.get((p1, 0), set([])).union(f0)
                for p2 in sec:
                    f1 = {(s, (r, p2)) for (s, r) in lists[p1]}
                    lfol[(p1, p2)] = lfol.get((p1, p2), set([])).union(f1)
            for p2 in sec:
                f0 = {(s, (0, r)) for (s, r) in sec[p2]} | {(s, (r, p2)) for (s, r) in first}
                lfol[(0, p2)] = lfol.get((0, p2), set([])).union(f0)
                for p1 in lists:
                    f1 = {(s, (p1, r)) for (s, r) in sec[p2]}
                    lfol[(p1, p2)] = lfol.get((p1, p2), set([])).union(f1)
            self._followL = lfol
        return self._followL

    def first(self, parent_list=None):
        """

        :param parent_list:
        :return:
        """

        def _totuple0(i):
            if type(i) != tuple:
                i = (i.val, i)
            return i[0], (i[1], 0)

        def _totuple1(i):
            if type(i) != tuple:
                i = (i.val, i)
            return i[0], (0, i[1])

        parent_list = list(map(_totuple0, self.arg1.first(parent_list)))
        return list(map(_totuple1, self.arg2.first(parent_list)))

    def followListsD(self, lists=None):
        """ in progress """
        lists = self.arg1.followListsD(lists)
        newlist = dict()
        for i in lists:
            if type(i) != tuple:
                newlist[(i.val, i)] = list[i]
            else:
                newlist[i] = lists[i]
        self.arg2.followListsD(lists)


class CShuffleU(Unary):
    """ Unary Shuffle operation of regexps
    """

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def __str__(self):
        return " {0:s} {1:s}".format(UShuffle, self.arg._strP())

    def _strP(self):
        return "({0:s} {1:s})".format(Shuffle, self.arg._strP())

    def rpn(self):
        return "{0:s}{1:s}".format(Shuffle, self.arg.rpn())

    def starHeight(self):
        return self.arg.starHeight()

    def epsilonLength(self):
        return self.arg.epsilonLength()

    def ewp(self):
        return self.arg.ewp()

    def reduced(self, has_epsilon=False):
        arg_r = self.arg.reduced()
        if arg_r.emptysetP():
            return CEmptySet(self.Sigma)
        if arg_r.epsilonP():
            return CEpsilon(self.Sigma)
        if arg_r is self.arg:
            return self
        reduced = CShuffleU(arg_r, self.Sigma)
        #        reduced._reduced = True
        return reduced

    _reducedS = reduced

    def _memoLF(self):
        raise regexpInvalidMethod()

    def snf(self):
        raise regexpInvalidMethod()

    def mark(self):
        raise regexpInvalidMethod()

    def unmark(self):
        raise regexpInvalidMethod()

    def _follow(self, _):
        raise regexpInvalidMethod()

    def derivative(self, sigma):
        return _dotshuffle(self.arg.derivative(sigma), self.arg, self.Sigma)

    def partialDerivatives(self, sigma):
        pdset = self.arg.partialDerivatives(sigma)
        return {_dotshuffle(a, self.arg, self.Sigma) for a in pdset}

    def support(self, side=True):
        """ """
        p = self.arg.support(side)
        lp = list(p)
        p0 = {_dotshuffle(lp[i1], lp[i2], self.Sigma) for i1 in range(len(lp)) for i2 in range(0, i1+1)}
        p0.update(_odotshuffle(p, {self.arg}, self.Sigma))
        return p0

    def supportlast(self, side=True):
        """ """
        p = self.arg.supportlast(side)
        lp = list(p)
        p0 = {_dotshuffle(lp[i1], lp[i2], self.Sigma) for i1 in range(len(lp)) for i2 in range(0, i1 + 1)}
        if self.arg.ewp():
            p0.update(_odotshuffle(p, {self.arg}, self.Sigma))
        return p0

    def linearForm(self):
        arg_lf = self.arg.linearForm()
        lf = dict()
        for head in arg_lf:
            lf[head] = {_dotshuffle(tail, self.arg, self.Sigma) for tail in arg_lf[head]}
        return lf

    def tailForm(self):
        arg_tf = self.arg.tailForm()
        tf = dict()
        for tail in arg_tf:
            tf[tail] = {_dotshuffle(head, self.arg, self.Sigma) for head in arg_tf[tail]}
        return tf

    def first(self, parent_list=None):
        """
        :param parent_list:
        :return:
        """
        pass

    def last(self, parent_last=None):
        pass

    def followListsD(self, lists=None):
        """ in progress """
        pass


class CConj(Connective):
    """ Intersection operation of regexps
    """
    def first_l(self):
        """ First sets for locations"""
        if not hasattr(self, "_firstL"):
            first = self.arg1.first_l()
            sec = self.arg2.first_l()
            inf = set([])
            for (s, p) in first:
                for (s1, p1) in sec:
                    if s == s1:
                        inf.add((s, (p, p1)))
            self._firstL = inf
        return self._firstL

    def last_l(self):
        """ Last sets for locations"""
        if not hasattr(self, "_lastL"):
            first = self.arg1.last_l()
            sec = self.arg2.last_l()
            self._lastL = {(p, p1) for p in first for p1 in sec}
        return self._lastL

    def follow_l(self):
        """ Follow sets for locations"""
        if not hasattr(self, "_followL"):
            lists = self.arg1.follow_l()
            sec = self.arg2.follow_l()
            lfol = {}
            for p1 in lists:
                for p2 in sec:
                    f1 = {(s, (r, r1)) for (s, r) in lists[p1] for (s1, r1) in sec[p2] if s == s1}
                    lfol[(p1, p2)] = lfol.get((p1, p2), set([])).union(f1)
            self._followL = lfol
        return self._followL

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def supportlast(self, side=True):
        raise FAdoNotImplemented()

    def tailForm(self):
        raise FAdoNotImplemented()

    def _follow(self, _):
        raise regexpInvalidMethod()

    def derivative(self, sigma):
        left = self.arg1.derivative(sigma)
        right = self.arg2.derivative(sigma)
        return CConj(left, right, self.Sigma)

    def unmark(self):
        raise regexpInvalidMethod()

    def mark(self):
        raise regexpInvalidMethod()

    def snf(self):
        raise regexpInvalidMethod()

    def _memoLF(self):
        raise regexpInvalidMethod()

    def __str__(self):
        return "{1:s} {0:s} {2:s}".format(Conj, self.arg1._strP(), self.arg2._strP())

    def _strP(self):
        return "({1:s} {0:s}  {2:s})".format(Conj, self.arg1._strP(), self.arg2._strP())

    def rpn(self):
        return "{0:s}{0:s}{1:s}".format(Conj, self.arg1.rpn(), self.arg2.rpn())

    def ewp(self):
        if hasattr(self, "_ewp"):
            return self._ewp
        return self.arg1.ewp() and self.arg2.ewp()

    def reduced(self, has_epsilon=False):
        left = self.arg1.reduced()
        right = self.arg2.reduced()
        if left.emptysetP() or right.emptysetP():
            return CEmptySet(self.Sigma)
        if left.epsilonP():
            if has_epsilon:
                return self.arg2.reduced(True)
            return right
        if right.epsilonP():
            if has_epsilon:
                return self.arg1.reduced(True)
            return left
        if left is self.arg1 and right is self.arg2:
            return self
        reduced = CConj(left, right, self.Sigma)
        #        reduced._reduced = True
        return reduced

    _reducedS = reduced

    def partialDerivatives(self, sigma):
        pdset = self.arg1.partialDerivatives(sigma)
        pd2set = self.arg2.partialDerivatives(sigma)
        return _odotconj(pdset, pd2set, self.Sigma)

    def support(self, side=True):
        """ """
        p = self.arg1.support(side)
        q = self.arg2.support(side)
        return _odotconj(p, q, self.Sigma)

    def linearForm(self):
        arg1_lf = self.arg1.linearForm()
        arg2_lf = self.arg2.linearForm()
        lf = dict()
        for head in set(arg1_lf.keys()) & set(arg2_lf.keys()):
            lf[head] = _odotconj(arg1_lf[head], arg2_lf[head], self.Sigma)
        return lf

    def reversal(self):
        return CConj(self.arg1.reversal(), self.arg2.reversal(), sigma=self.Sigma)


class Position(CAtom):
    """Class for marked regular expression symbols.

    .. inheritance-diagram:: Position"""

    def unmark(self):
        raise regexpInvalidMethod()

    def _follow(self, flag):
        raise regexpInvalidMethod()

    def __repr__(self):
        return "Position%s" % repr(self.val)

    def __copy__(self):
        return Position(self.val)

    def setOfSymbols(self):
        return {self.val}

    def unmarked(self):
        return CAtom(self.val[0], self.Sigma)

    def symbol(self):
        return self.val[0]


class CSigmaS(SpecialConstant):
    """
    Special regular expressions modulo associativity, commutativity, idempotence of disjunction and intersection;
      associativity of concatenation; identities sigma^* and sigma^+.

       CSigmaS: Class that represents the complement of the EmptySet set (sigma^*)

    .. seealso: SConnective

    .. inheritance-diagram:: CSigmaS
    """

    def __init__(self, sigma=None):
        """Constructor of a regular expression symbol.

        :arg val: the actual symbol"""
        super(CSigmaS, self).__init__()
        self._ewp = True
        self.Sigma = sigma

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def rpn(self):
        return self.__repr__()

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        """
        :return:"""
        if self.Sigma is not None:
            return "CSigmaS({})".format(self.Sigma)
        return "CSigmaS()"

    def __str__(self):
        """
        :return:"""
        return SigmaS

    _strP = __str__

    def __eq__(self, other):
        return self.Sigma == other.Sigma and type(self) == type(other)

    def ewp(self):
        """
        :return:"""
        return True

    def linearForm(self):
        """
        :return:"""
        lf = dict()
        for sigma in self.Sigma:
            lf[sigma] = {self}
        return lf

    def linearFormC(self):
        """
        :return:"""
        return dict()

    def derivative(self, sigma):
        """
        :param sigma:
        :return:"""
        return self

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def partialDerivatives(self, sigma):
        """
        :param sigma:
        :return:"""
        return {self}

    def partialDerivativesC(self, sigma):
        """
        :param sigma:
        :return:"""
        return set()

    def support(self, side=True):
        """
        :return: """
        return {self}

    def _plus(self, r):
        """
        :param: r
        :return: """
        if self.Sigma and r.Sigma:
            self.Sigma = r.Sigma | self.Sigma
        return self

    def _inter(self, r):
        """
        :param: r
        :return: """
        if self.Sigma and r.Sigma:
            r.Sigma = r.Sigma | self.Sigma
        return r


class CSigmaP(SpecialConstant):
    """
    Special regular expressions modulo associativity, commutativity, idempotence of disjunction and intersection;
      associativity of concatenation; identities sigma^* and sigma^+.

       CSigmaP: Class that represents the complement of the EmptySet word (sigma^+)

    .. seealso: SConnective
    .. inheritance-diagram:: CSigmaP
      """

    def __init__(self, sigma=None):
        """Constructor of a regular expression symbol.

        :arg val: the actual symbol"""
        super(CSigmaP, self).__init__()
        self._ewp = False
        self.Sigma = sigma

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented

    def rpn(self):
        return self.__repr__()

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        """
        :return:"""
        if self.Sigma is not None:
            return "CSigmaP({})".format(self.Sigma)
        return "CSigmaP()"

    def __str__(self):
        """
        :return:"""
        return SigmaP

    _strP = __str__

    # def __eq__(self, other):
    #     return self.sigma == other.sigma and type(self) == type(other)

    def ewp(self):
        """
        :return: """
        return False

    def derivative(self, sigma):
        """
        :param sigma:
        :return: """
        return CSigmaS(self.Sigma)

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def partialDerivatives(self, sigma):
        """
        :param sigma:
        :return: """
        return {CSigmaS(self.Sigma)}

    def partialDerivativesC(self, _):
        """
        :param _:
        :return: """
        return set()

    def linearForm(self):
        """
        :return:"""
        lf = dict()
        for sigma in self.Sigma:
            lf[sigma] = {CSigmaS(self.Sigma)}
        return lf

    def linearFormC(self):
        """
        :return: """
        return dict()

    def support(self, side=True):
        """
        :return: """
        return {CSigmaS(self.Sigma)}


class SConnective(RegExp):
    """ Special regular expressions modulo associativity, commutativity, idempotence of disjunction and intersection;
      associativity of concatenation; identities sigma^* and sigma^+. Connectives are:
       SDisj: disjunction
       SConj: intersection
       SConcat: concatenation

       For parsing use str2sre

     .. seealso:  Manipulation of Extended Regular expressions. Rafaela Bastos, Msc Thesis 2015
                    https://www.dcc.fc.up.pt/~nam/resources/rafaelamsc.pdf

     .. inheritance-diagram:: SConnective
    """

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def tailForm(self):
        pass

    @abstractmethod
    def support(self, side=True):
        pass

    @abstractmethod
    def supportlast(self, side=True):
        pass

    @abstractmethod
    def _nfaFollowEpsilonStep(self, _):
        pass

    def __init__(self, arg, sigma=None):
        super(SConnective, self).__init__()
        self.arg = arg
        self.Sigma = sigma
        self.hash = None

    def __hash__(self):
        if self.hash:
            return self.hash
        else:
            h = hash((self.arg, type(self)))
            self.hash = h
            return h

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __len__(self):
        """
        :return: """
        return len(self.arg)

    def mark(self):
        raise regexpInvalidMethod()

    def snf(self):
        raise regexpInvalidMethod()

    def rpn(self):
        raise regexpInvalidMethod()

    def derivative(self, _):
        raise regexpInvalidMethod()

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def _follow(self, _):
        raise regexpInvalidMethod()

    def unmark(self):
        raise regexpInvalidMethod()

    def linearForm(self):
        raise regexpInvalidMethod()

    def followListsD(self):
        raise regexpInvalidMethod()

    def first(self):
        raise regexpInvalidMethod()

    def starHeight(self):
        raise regexpInvalidMethod()

    def followLists(self):
        raise regexpInvalidMethod()

    def reduced(self):
        raise regexpInvalidMethod()

    def _marked(self, _):
        raise regexpInvalidMethod()

    def last(self):
        raise regexpInvalidMethod()

    def _memoLF(self):
        raise regexpInvalidMethod()

    def alphabeticLength(self):
        """
        :return: """
        length = 0
        for i in self.arg:
            length += i.alphabeticLength()
        return length

    def epsilonLength(self):
        """
        :return: """
        length = 0
        for i in self.arg:
            length += i.epsilonLength()
        return length

    def treeLength(self):
        """
        :return: """
        length = 0
        for i in self.arg:
            length += i.treeLength()
        return length + 1

    def syntacticLength(self):
        """
        :return: """
        length = len(self.arg) - 1
        for i in self.arg:
            length += i.syntacticLength()
        return length

    def setOfSymbols(self):
        """
        :return: """
        s = set()
        for i in self.arg:
            s = s | i.setOfSymbols()
        return s

    def _setSigma(self, strict=False):
        for r in self.arg:
            r.setSigma(self.Sigma, strict=strict)


class SConcat(SConnective):
    """Class that represents the concatenation operation.

    .. inheritance-diagram:: CConcat
    """

    def tailForm(self):
        raise FAdoNotImplemented()

    def supportlast(self, side=True):
        raise FAdoNotImplemented()

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def __str__(self):
        """
        :return: """
        return "{0:s}".format(self._strP())

    def _strP(self):
        """
        :return: """
        rep = ""
        for i in self.arg:
            rep += "{0:s}".format(str(i))
        return rep

    def __repr__(self):
        """
        :return: """
        return "SConcat({0:s})".format(repr(self.arg))

    def ewp(self):
        """
        :return: """
        for i in self.arg:
            if not (i.ewp()):
                return False
        return True

    def head(self):
        """
        :return: """
        re1 = self.arg[0]
        re1.setSigma(self.Sigma)
        return re1

    def tail(self):
        """
        :return: """
        if len(self.arg) == 2:
            re1 = self.arg[1]
            re1.setSigma(self.Sigma)
            return re1
        else:
            re1 = SConcat(self.arg[1:])
            re1.setSigma(symbolset=self.Sigma)
            return re1

    def head_rev(self):
        """
        :return: """
        re1 = self.arg[-1]
        re1.setSigma(self.Sigma)
        return re1

    def tail_rev(self):
        """
        :return: """
        if len(self.arg) == 2:
            re1 = self.arg[0]
            re1.setSigma(self.Sigma)
            return re1
        else:
            re1 = SConcat(self.arg[:-1])
            re1.setSigma(self.Sigma)
            return re1

    def derivative(self, sigma):
        """
        :param sigma:
        :return: """
        head = self.head()
        tail = self.tail()
        der = head.derivative(sigma)._dot(tail)
        if head.ewp():
            der_tail = tail.derivative(sigma)
            return der._plus(der_tail)
        return der

    def partialDerivatives(self, sigma):
        """
        :param sigma:
        :return: """
        head = self.head()
        tail = self.tail()
        pd_head = head.partialDerivatives(sigma)
        pd = _concat_set(pd_head, tail)
        if head.ewp():
            pd_tail = tail.partialDerivatives(sigma)
            return pd | pd_tail
        else:
            return pd

    def partialDerivativesC(self, sigma):
        """
        :param sigma:
        :return: """
        head = self.head()
        tail = self.tail()
        pd_head = head.partialDerivatives(sigma)
        pd = _negation_set(_concat_set(pd_head, tail), self.Sigma)
        if head.ewp():
            pd_tail = tail.partialDerivativesC(sigma)
            return _intersection_set(pd, pd_tail)
        else:
            return pd

    def linearFormC(self):
        """
        :return: """
        head = self.head()
        tail = self.tail()
        lf_head = head.linearForm()
        lf = _negation_mon(_concat_mon(lf_head, tail), self.Sigma)
        if head.ewp():
            lfc_tail = tail.linearFormC()
            return _intersection_mon(lf, lfc_tail)
        return lf

    def linearForm(self):
        """
        :return: """
        head = self.head()
        tail = self.tail()
        lf_head = head.linearForm()
        lf = _concat_mon(lf_head, tail)
        if head.ewp():
            lf_tail = tail.linearForm()
            return _union_mon(lf, lf_tail)
        return lf

    def support(self, side=True):
        """
        :return: """
        head = self.head()
        tail = self.tail()
        pi = _concat_set(head.support(side), tail)
        return pi | tail.support(side)

    def _dot(self, r):
        """
        :param r:
        :return: """
        if type(r) is CEpsilon or type(r) is CEmptySet:
            return r._dot(self)
        if type(r) is SConcat:
            r_conc = SConcat(self.arg + r.arg)
            if self.Sigma and r.Sigma:
                r_conc.Sigma = self.Sigma | r.Sigma
            return r_conc
        else:
            r_conc = SConcat(self.arg + (r,))
            if self.Sigma and r.Sigma:
                r_conc.Sigma = self.Sigma | r.Sigma
            return r_conc


class SDisj(SConnective):
    """Class that represents the disjunction operation for special regular expressions.

    .. inheritance-diagram:: SDisj
    """

    def tailForm(self):
        raise FAdoNotImplemented()

    def supportlast(self, side=True):
        raise FAdoNotImplemented()

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def __repr__(self):
        """
        :return: """
        return "SDisj({0:s})".format(repr(self.arg))

    def __str__(self):
        """
        :return: """
        return "({0:s})".format(self._strP())

    def _strP(self):
        """
        :return: """
        return " + ".join([str(i) for i in self.arg])

    def ewp(self):
        """
        :return: """
        for i in self.arg:
            if i.ewp():
                return True
        return False

    def _marked(self, pos):
        """
        :param pos:
        :return: """
        s = set()
        for i in self.arg:
            mark, pos = i._marked(pos)
            s.add(mark)
        return SDisj(s), pos

    def first(self):
        """
        :return: """
        fst = set()
        for ri in self.arg:
            fst.update(ri.first())
        return fst

    def last(self):
        """
        :return: """
        lst = set()
        for ri in self.arg:
            lst.update(ri.last())
        return lst

    def followLists(self, lists=None):
        """
        :param lists:
        :return: """
        if lists is None:
            lists = dict()
        for ri in self.arg:
            ri.followLists(lists)
        return lists

    def followListsStar(self, lists=None):
        """
        :param lists:
        :return: """
        if lists is None:
            lists = dict()
        s = set()
        for ri in self.arg:
            ri.followListsStar(lists)
            s.add(ri)
            # inspect
            self.cross(ri, s, lists)
        return lists

    @staticmethod
    def cross(ri, s, lists):
        """
        Returns:
             list:"""
        first_ri = ri.first()
        last_ri = ri.last()
        for rj in s:
            first_rj = rj.first()
            for symbol in last_ri:
                if symbol in lists:
                    lists[symbol].update(first_rj)
                else:
                    lists[symbol] = first_rj
            last_rj = rj.last()
            for symbol in last_rj:
                if symbol in lists:
                    lists[symbol].update(first_ri)
                else:
                    lists[symbol] = first_ri
        return lists

    def derivative(self, sigma):
        """:param sigma:
         :return: """
        der = CEmptySet(self.Sigma)
        for alpha_i in self.arg:
            der = der._plus(alpha_i.derivative(sigma))
        return der

    def partialDerivatives(self, sigma):
        """
        :param sigma:
        :return: """
        pd = set()
        for i in self.arg:
            pd_re = i.partialDerivatives(sigma)
            pd.update(pd_re)
        return pd

    def partialDerivativesC(self, sigma):
        """
        :param sigma:
        :return: """
        pd = set()
        flag = True
        for re1 in self.arg:
            pd_re = re1.partialDerivativesC(sigma)
            if not pd_re:
                return pd_re
            elif flag:
                flag = False
                pd = pd_re
            else:
                pd = _intersection_set(pd, pd_re)
        return pd

    def linearForm(self):
        """
        :return: """
        lf = dict()
        for ri in self.arg:
            lf_re = ri.linearForm()
            lf = _union_mon(lf, lf_re)
        return lf

    def linearFormC(self):
        """
        :return: """
        lf = dict()
        flag = True
        for re1 in self.arg:
            if flag:
                flag = False
                lf = re1.linearFormC()
            else:
                lf_re = re1.linearFormC()
                lf = _intersection_mon(lf, lf_re)
        return lf

    def support(self, side=True):
        """
        :return: """
        s = set()
        for i in self.arg:
            s.update(i.support(side))
        return s

    def _plus(self, re1):
        """
        :param re1:
        :return: """
        if re1 == self:
            if self.Sigma and re1.Sigma:
                re1.Sigma = re1.Sigma | self.Sigma
            return re1
        elif type(re1) is CEmptySet or type(re1) is CSigmaS:
            return re1._plus(self)
        if type(re1) is SDisj:
            r_disj = SDisj(self.arg | re1.arg)
            if self.Sigma and re1.Sigma:
                re1.Sigma = re1.Sigma | self.Sigma
            return r_disj
        else:
            r_disj = SDisj(self.arg | frozenset([re1]))
            if self.Sigma and re1.Sigma:
                r_disj.Sigma = re1.Sigma | self.Sigma
            return r_disj


class SConj(SConnective):
    """Class that represents the conjunction operation.

    .. inheritance-diagram:: CConcat
    """

    def tailForm(self):
        raise FAdoNotImplemented()

    def supportlast(self, side=True):
        raise FAdoNotImplemented()

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def __str__(self):
        """
        :return:"""
        return "({0:s})".format(self._strP())

    def _strP(self):
        """
        :return:"""
        return "&".join([str(i) for i in self.arg])

    def __repr__(self):
        """
        :return: """
        return "SConj({0:s})".format(repr(self.arg))

    def ewp(self):
        """
        :return: """
        for i in self.arg:
            if not i.ewp():
                return False
        return True

    def derivative(self, sigma):
        """:param sigma:
        :return: """
        der = CSigmaS(self.Sigma)
        for alpha_i in self.arg:
            der = der._inter(alpha_i.derivative(sigma))
        return der

    def partialDerivatives(self, sigma):
        """
        :param sigma:
        :return: """
        pd = set()
        flag = True
        for i in self.arg:
            pd_re = i.partialDerivatives(sigma)
            if not pd_re:
                return pd_re
            elif flag:
                flag = False
                pd = pd_re
            else:
                pd = _intersection_set(pd, pd_re)
        return pd

    def partialDerivativesC(self, sigma):
        """
        :param sigma:
        :return: """
        pd = set()
        for re1 in self.arg:
            pd_re = re1.partialDerivativesC(sigma)
            pd.update(pd_re)
        return pd

    def linearForm(self):
        """
        :return: """
        lf = dict()
        flag = True
        for re1 in self.arg:
            if flag:
                flag = False
                lf = re1.linearForm()
            else:
                lf_re = re1.linearForm()
                lf = _intersection_mon(lf, lf_re)
        return lf

    def linearFormC(self):
        lf = dict()
        for re1 in self.arg:
            lf_re = re1.linearFormC()
            lf = _union_mon(lf, lf_re)
        return lf

    def support(self, side=True):
        """
        :return: """
        pi = set()
        flag = True
        for re1 in self.arg:
            pi_re = re1.support(side)
            if flag:
                flag = False
                pi = pi_re
            elif pi_re == set():
                return set()
            else:
                pi = _intersection_set(pi, pi_re)
        return pi

    def _inter(self, re1):
        """
        :param re1:
        :return: """
        if re1 == self:
            if self.Sigma and re1.Sigma:
                re1.Sigma = re1.Sigma | self.Sigma
            return re1
        elif type(re1) is CEmptySet or type(re1) is CSigmaS:
            return re1._inter(self)
        if type(re1) is SConj:
            r_conj = SConj(self.arg | re1.arg)
            if self.Sigma and re1.Sigma:
                re1.Sigma = re1.Sigma | self.Sigma
            return r_conj
        else:
            r_conj = SConj(self.arg | frozenset([re1]))
            if self.Sigma and re1.Sigma:
                r_conj.Sigma = re1.Sigma | self.Sigma
            return r_conj


class SStar(CStar):
    """Special regular expressions modulo associativity, commutativity, idempotence of disjunction and intersection;
      associativity of concatenation; identities sigma^* and sigma^+.

       SStar: Class that represents Kleene star

    .. seealso: SConnective

    .. inheritance-diagram::  SStar
    """

    def __init__(self, arg, sigma=None):
        super(SStar, self).__init__(arg, sigma)
        self.hash = None

    def __str__(self):
        """
        :return: """
        if type(self.arg) is SConcat or type(self.arg) is SNot:
            return "({0:s})*".format(str(self.arg))
        else:
            return "{0:s}*".format(str(self.arg))

    def __repr__(self):
        """
        :return: """
        return "SStar({0:s})".format(repr(self.arg))

    def __hash__(self):
        if self.hash:
            return self.hash
        else:
            h = hash((self.arg, type(self)))
            self.hash = h
            return h

    def __eq__(self, other):
        return hash(self) == hash(other)

    def _setSigma(self, strict=False):
        self.arg.setSigma(self.Sigma, strict=strict)

    def derivative(self, sigma):
        """
        :param sigma:
        :return: """
        der = self.arg.derivative(sigma)
        return der._dot(self)

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def partialDerivatives(self, sigma):
        """
        :param sigma:
        :return:"""
        der = self.arg.partialDerivatives(sigma)
        return _concat_set(der, self)

    def partialDerivativesC(self, sigma):
        """
        :param sigma:
        :return: """
        pd = self.arg.partialDerivatives(sigma)
        return _negation_set(_concat_set(pd, self), self.Sigma)

    def linearForm(self):
        """
        :return: """
        lf_arg = self.arg.linearForm()
        return _concat_mon(lf_arg, self)

    def linearFormC(self):
        lf = self.linearForm()
        return _negation_mon(lf, self.Sigma)

    def support(self, side=True):
        """
        :return: """
        p = self.arg.support(side)
        return _concat_set(p, self)


class SNot(RegExp):
    """ Special regular expressions modulo associativity, commutativity, idempotence of disjunction and intersection;
        associativity of concatenation; identities sigma^* and sigma^+.
        SNot: negation

    .. seealso: SConnective

    .. inheritance-diagram:: SNot
    """

    def tailForm(self):
        raise FAdoNotImplemented()

    def supportlast(self, side=True):
        raise FAdoNotImplemented()

    def _nfaFollowEpsilonStep(self, _):
        raise FAdoNotImplemented()

    def mark(self):
        pass

    def snf(self):
        pass

    def rpn(self):
        pass

    def _follow(self, _):
        pass

    def unmark(self):
        pass

    def followListsD(self):
        pass

    def first(self):
        pass

    def starHeight(self):
        pass

    def followLists(self):
        pass

    def reduced(self):
        pass

    def _marked(self, _):
        pass

    def last(self):
        pass

    def _memoLF(self):
        pass

    def __init__(self, arg, sigma=None):
        super(SNot, self).__init__()
        if sigma is None:
            sigma = set()
        self.arg = arg
        self.Sigma = sigma
        self.hash = None

    def __hash__(self):
        if self.hash:
            return self.hash
        else:
            h = hash((self.arg, type(self)))
            self.hash = h
            return h

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        """
        :return: """
        if type(self.arg) is SConcat or type(self.arg) is SStar:
            return "{0:s}({1:s})".format(Not, self.arg._strP())
        else:
            return "{0:s}{1:s}".format(Not, self.arg._strP())

    _strP = __str__

    def __repr__(self):
        """
        :return:"""
        return "SNot(%s)" % repr(self.arg)

    def _setSigma(self, strict=False):
        self.arg.setSigma(self.Sigma, strict=strict)

    def ewp(self):
        """
        :return:"""
        if self.arg.ewp():
            return False
        else:
            return True

    def alphabeticLength(self):
        """
        :return:"""
        return self.arg.alphabeticLength()

    def epsilonLength(self):
        """
        :return:"""
        return self.arg.epsilonLength()

    def syntacticLength(self):
        """
        :return:"""
        return 1 + self.arg.syntacticLength()

    def treeLength(self):
        """
        :return:"""
        return 1 + self.arg.treeLength()

    def setOfSymbols(self):
        """
        :return:"""
        return self.arg.setOfSymbols()

    def derivative(self, sigma):
        """
        :param sigma
        :return:"""
        der = self.arg.derivative(sigma)
        return SNot(der, self.Sigma)

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def support(self, side=True):
        """
        :return:"""
        pi_arg = self.arg.support(side)
        power_pi = powerset(pi_arg)
        s = set(next(power_pi))
        pi = _negation_set(s, self.Sigma)
        while s:
            s = set(next(power_pi))
            pi.update(_negation_set(s, self.Sigma))
        return pi

    def partialDerivatives(self, sigma):
        """
        :param sigma:
        :return:"""
        return self.arg.partialDerivativesC(sigma)

    def partialDerivativesC(self, sigma):
        """
        :param sigma:
        :return:"""
        return self.arg.partialDerivatives(sigma)

    def linearForm(self):
        """
        :return: """
        return self.arg.linearFormC()

    def linearFormC(self):
        """
        :return: """
        return self.arg.linearForm()


def to_s(r):
    """Returns a  sre from FAdo regexp.

    :arg RegExp r: the FAdo representation regexp for a regular expression.
    :rtype: RegExp"""
    if type(r) is CAtom or type(r) is CEpsilon or type(r) is CEmptySet:
        return r
    if type(r) is CDisj:
        return SDisj(frozenset(_to_sdisj(r)))
    if type(r) is CConcat:
        if hasattr(r, "arg1") and hasattr(r, "arg2"):
            if type(r.arg1) is CEpsilon and type(r.arg2) is CEpsilon:
                return CEpsilon()
            elif type(r.arg1) is CEpsilon:
                return to_s(r.arg2)
            elif type(r.arg2) is CEpsilon:
                return to_s(r.arg1)
            else:
                return SConcat(_to_sconcat(r))
        else:
            raise FAdoGeneralError
    if type(r) is CStar:
        r = to_s(r.arg)
        if type(r) == CEpsilon:
            return r
        elif type(r) == SStar:
            return r
        else:
            return SStar(r)


def _to_sdisj(r):
    """
    :param r: RegExp
    :return: RegExp """
    if hasattr(r, "arg1") and hasattr(r, "arg2"):
        if type(r.arg1) is CDisj:
            if type(r.arg2) is CDisj:
                return _to_sdisj(r.arg1) | _to_sdisj(r.arg2)
            else:
                return _to_sdisj(r.arg1) | {to_s(r.arg2)}
        else:
            if type(r.arg2) is CDisj:
                return {to_s(r.arg1)} | _to_sdisj(r.arg2)
            else:
                return {to_s(r.arg1)} | {to_s(r.arg2)}
    else:
        raise FAdoGeneralError


def _to_sconcat(r):
    """
     :param r:  RegExp
    :return:  RegExp sre"""
    if type(r.arg1) is CConcat:
        if type(r.arg2) is CConcat:
            return _to_sconcat(r.arg1) + _to_sconcat(r.arg2)
        elif type(r.arg2) is CEpsilon:
            return _to_sconcat(r.arg1)
        else:
            return _to_sconcat(r.arg1) + (to_s(r.arg2),)
    else:
        if type(r.arg2) is CConcat:
            return (to_s(r.arg1),) + _to_sconcat(r.arg2)
        elif type(r.arg2) is CEpsilon:
            return to_s(r.arg1),
        else:
            return (to_s(r.arg1),) + (to_s(r.arg2),)


def rpn2regexp(s, sigma=None, strict=False):
    """Reads a (simple) RegExp from a RPN representation

    .. productionlist:: Representation r
       r: .RR | +RR | *r | L | @
       L: [a-z] | [A-Z]

    :param s: RPN representation
    :type s: str
    :param strict: Boolean
    :type strict: bool
    :param sigma: alphabet
    :type sigma: set
    :rtype: reex.RegExp

    .. note:: This method uses python stack... thus depth limitations apply"""
    (nf, reg) = _rpn2re(re.sub("@CEpsilon", "@", s), 0, sigma)
    if sigma is not None:
        reg.setSigma(sigma, strict)
    elif strict:
        reg.setSigma(reg.setOfSymbols())
    return reg


def _rpn2re(s, i, sigma=None):
    """
    :param s:
    :param i:
    :return:
    """
    if s[i] in "+.":
        (i1, arg1) = _rpn2re(s, i + 1)
        (i2, arg2) = _rpn2re(s, i1)
        if s[i] == ".":
            return i2, CConcat(arg1, arg2, sigma)
        else:
            return i2, CDisj(arg1, arg2, sigma)
    if s[i] == "*":
        (i1, arg1) = _rpn2re(s, i + 1)
        return i1, CStar(arg1, sigma)
    if s[i] == "@":
        return i + 1, CEpsilon(sigma)
    else:
        return i + 1, CAtom(s[i], sigma)


def _concat_set(s, reg):
    """Computes a set that contains the concatenation of a regular expression with all the regular expression
        within a set.

        :arg s: set of regular expressions
        :arg reg: regular expression
        :rtype: set of regular expression"""
    if type(reg) is CEmptySet:
        return set()
    elif type(reg) is CEpsilon:
        return s
    else:
        new_set = set()
        for re_i in s:
            new_set.add(re_i._dot(reg))
        return new_set


def _intersection_set(s1, s2):
    if s1 == set() or s2 == set():
        return set()
    else:
        new_set = set()
        for r1 in s1:
            for r2 in s2:
                new_set.add(r1._inter(r2))
        return new_set


def _negation_set(s, sigma=None):
    if s == set():
        return {CSigmaS(sigma)}
    else:
        re1 = CSigmaS(sigma)
        for re_i in s:
            if sigma and re_i.Sigma:
                re1.Sigma = re1.Sigma | re_i.Sigma
            re1 = re1._inter(SNot(re_i))
        re1.setSigma(sigma)
        return {re1}


def _concat_mon(d, re1):
    """Computes a set that contains the concatenation of a regular expression with all the monomials
        within a set.

        :arg d: dict of monomials
        :arg re1: regular expression
        :rtype: dict  of monomials"""
    if type(re1) is CEmptySet:
        return dict()
    elif type(re1) is CEpsilon:
        return d
    else:
        new_dic = dict()
        for sigma in d:
            new_set = set()
            for re_i in d[sigma]:
                new_set.add(re_i._dot(re1))
            new_dic[sigma] = new_set
        return new_dic


def _intersection_mon(d1, d2):
    new_dic = dict()
    for a in d1:
        if a in d2:
            new_dic[a] = _intersection_set(d1[a], d2[a])
    return new_dic


def _negation_mon(d, sigma=None):
    new_dic = dict()
    for a in sigma:
        if a in d:
            new_dic[a] = _negation_set(d[a], sigma)
        else:
            new_dic[a] = _negation_set(set(), sigma)
    return new_dic


def _union_mon(d1, d2):
    """
    Args:
        d1 (dict) : dictionary one
        d2 (dict) : dictionary two
    """
    new_dic = dict()
    keys = set(d1.keys()) | set(d2.keys())
    for a in keys:
        if a in d1 and a in d2:
            new_dic[a] = d1[a] | d2[a]
        elif a in d1:
            new_dic[a] = d1[a]
        elif a in d2:
            new_dic[a] = d2[a]
    return new_dic


def _odotshuffle(s, r, sigma=None):
    """Shuffle of two sets of regexps
      Args:
          s (set):
          r (set):
          sigma (set): alphabet
      Returns:
            set of RegExp:
    """
    return {_dotshuffle(s1, s2, sigma) for s1 in s for s2 in r}


def _dotshuffle(s1, s2, sigma=None):
    """ Shuffle of two regexps
     Args:
          s1 (RegExp):
          s2 (RegExp):
     Returns:
           RegExp:
    """
    if s1.epsilonP():
        a = s2
    elif s2.epsilonP():
        a = s1
    elif s2 == s1:
        a = CShuffleU(s1, sigma)
    else:
        a = CShuffle(s1, s2, sigma)
    return a


def _odotconj(s, r, sigma=None):
    """Intersection of two  sets of Regexps
        Args:
            s (set):
            r (set):
        Returns:
            set:
    """
    return {CConj(s1, s2, sigma) for s1 in s for s2 in r}


def powerset(iterable):
    """ Powerset of a set.
    Args:
        iterable (list): the set
    Returns:
        itertools.chain:the powerset"""

    s = list(iterable)
    return chain.from_iterable(combinations(s, (len(s)) - r) for r in range(len(s) + 1))


def _concat(a, b):
    if a == Epsilon:
        return b
    elif b == Epsilon:
        return a
    else:
        return str(a) + str(b)


def _ifconcat(re0, re1, dd=True, both=False, sigma=None):
    """Concatenation of two res if not Epsilon
    Args:
        re0 (RegExp):
        re1 (RegExp):
        dd (bool): if True re0.re1 else r21.re0
        both (bool): if True test both
    Returns:
        RegExp: the concatenation
    """
    if re0.epsilonP():
        return re1
    if both and re1.epsilonP():
        return re0
    if dd:
        return CConcat(re0, re1, sigma)
    return CConcat(re1, re0, sigma)


class DNode(object):
    def __init__(self, op, arg1=None, arg2=None):
        self.dotl = set([])
        self.dotr = set([])
        self.star = None
        self.option = None
        self.plus = set([])
        self.shuffle_p = set([])
        self.shuffle_u = None
        self.inter = set([])
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2
        self.diff = dict()
        if op in [ID_OPTION, ID_STAR, ID_EPSILON]:
            self.ewp = True
        elif op in (ID_CONC, ID_DISJ, ID_SHUFFLE, ID_CONJ):
            pass
        else:
            self.ewp = False


class DAG(object):
    """Class to support dags representing regexps

    ...seealso: P. Flajolet, P. Sipala, J.-M. Steyaert, Analytic variations on the common subexpression problem,
            in: Automata, Languages and Programmin, LNCS, vol. 443, Springer, New York, 1990, pp. 220–234."""
    def __init__(self, reg):
        """ Args:
                reg (RegExp): regular expression"""
        self._beingDone = None
        self.table = dict()
        self.table[0] = DNode(ID_EPSILON)
        self.table[1] = DNode(ID_EMPTYSET)
        self.count = 2
        self.leafs = dict()
        self.diff2do = set()
        self.Sigma = reg.Sigma
        self.root = self.getIdx(reg)

    def __len__(self):
        return self.count - 1

    def getIdx(self, reg):
        """
        Builds dag nodes
        Args:
            reg (regexp): regular expression
        Returns:
              int: node id
        """
        if isinstance(reg, CAtom):
            return self.getAtomIdx(reg.val)
        elif isinstance(reg, CDisj):
            id1 = self.getIdx(reg.arg1)
            id2 = self.getIdx(reg.arg2)
            return self.getDisjIdx(id1, id2)
        elif isinstance(reg, CConcat):
            id1 = self.getIdx(reg.arg1)
            id2 = self.getIdx(reg.arg2)
            return self.getConcatIdx(id1, id2)
        elif isinstance(reg, CShuffle):
            id1 = self.getIdx(reg.arg1)
            id2 = self.getIdx(reg.arg2)
            return self.getShuffleIdx(id1, id2)
        elif isinstance(reg, CConj):
            id1 = self.getIdx(reg.arg1)
            id2 = self.getIdx(reg.arg2)
            return self.getConjIdx(id1, id2)
        elif isinstance(reg, CStar):
            id1 = self.getIdx(reg.arg)
            return self.getStarIdx(id1)
        elif isinstance(reg, COption):
            id1 = self.getIdx(reg.arg)
            return self.getOptionIdx(id1)
        elif isinstance(reg, CEmptySet):
            return 1
        else:  # It must be CEpsilon
            return 0

    def getAtomIdx(self, val):
        """
        Node atom
        Args:
            val (str): letter
        Returns:
              int: node id
        """
        if val in self.leafs:
            return self.leafs[val]
        else:
            id1 = self.count
            self.count += 1
            self.leafs[val] = id1
            new = DNode(ID_SYMB)
            new.diff[val] = {0}
            self.table[id1] = new
            return id1

    def getStarIdx(self, arg_id):
        if arg_id == 1 or arg_id == 0:
            return 0
        if self.table[arg_id].star is not None:
            return self.table[arg_id].star
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_STAR, arg_id)
            self.table[arg_id].star = id1
            self.table[id1] = new
            # new.arg1 = arg_id
            new.diff = self.catLF(arg_id, id1, True)
            self.doDelayed()
            return id1

    def getOptionIdx(self, arg_id):
        if self.table[arg_id].option is not None:
            return self.table[arg_id].option
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_OPTION, arg_id)
            self.table[arg_id].option = id1
            self.table[id1] = new
            # new.arg1 = arg_id
            new.diff = self.table[arg_id].diff
            return id1

    def getConcatIdx(self, arg1_id, arg2_id, delay=False):
        if arg1_id == 0:
            return arg2_id
        if arg2_id == 0:
            return arg1_id
        a = self.table[arg1_id].dotl.intersection(self.table[arg2_id].dotr)
        if len(a):
            return a.pop()
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_CONC, arg1_id, arg2_id)
            self.table[id1] = new
            #  new.arg1, new.arg2 = arg1_id, arg2_id
            self.ewpFixConc(new, arg1_id, arg2_id)
            self.table[arg1_id].dotl.add(id1)
            self.table[arg2_id].dotr.add(id1)
            if not delay:
                if self.table[arg1_id].ewp:
                    new.diff = self.plusLF(self.catLF(arg1_id, arg2_id), self.table[arg2_id].diff)
                else:
                    new.diff = self.catLF(arg1_id, arg2_id)
            else:
                self.diff2do.add(id1)
            return id1

    def ewpFixConc(self, obj, arg1_id, arg2_id):
        obj.ewp = self.table[arg1_id].ewp and self.table[arg2_id].ewp


    def getDisjIdx(self, arg1_id, arg2_id):
        if arg1_id == arg2_id:
            return arg1_id
        a = self.table[arg1_id].plus.intersection(self.table[arg2_id].plus)
        if len(a):
            return a.pop()
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_DISJ, arg1_id, arg2_id)
            self.table[id1] = new
        #    new.arg1, new.arg2 = arg1_id, arg2_id
            self.ewpFixDisj(new, arg1_id, arg2_id)
            self.table[arg1_id].plus.add(id1)
            self.table[arg2_id].plus.add(id1)
            new.diff = self.plusLF(self.table[arg1_id].diff, self.table[arg2_id].diff)
            return id1

    def ewpFixDisj(self, obj, arg1_id, arg2_id):
        obj.ewp = self.table[arg1_id].ewp or self.table[arg2_id].ewp

    def getConjIdx(self, arg1_id, arg2_id):
        if arg1_id == arg2_id:
            return arg1_id
        a = self.table[arg1_id].inter.intersection(self.table[arg2_id].inter)
        if len(a):
            return a.pop()
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_CONJ, arg1_id, arg2_id)
            self.table[id1] = new
            self.ewpFixConc(new, arg1_id, arg2_id)
            self.table[arg1_id].inter.add(id1)
            self.table[arg2_id].inter.add(id1)
            new.diff = self.interLF(self.table[arg1_id].diff, self.table[arg2_id].diff)
            return id1

    def getShuffleIdx(self, arg1_id, arg2_id):
        if arg1_id == 0:
            return arg2_id
        if arg2_id == 0:
            return arg1_id
        if arg1_id == arg2_id:
            if self.table[arg1_id].shuffle_u is not None:
                return self.table[arg1_id].shuffle_u
            else:
                id1 = self.count
                self.count += 1
                new = DNode(ID_SHUFFLE, arg1_id, arg2_id)
                self.table[id1] = new
            #    new.arg1, new.arg2 = arg1_id, arg2_id
                new.ewp = self.table[arg1_id].ewp
                self.table[arg1_id].shuffle_u = id1
                new.diff = self.shuffleLF(arg1_id, arg2_id)
                return id1
        else:
            a = self.table[arg1_id].shuffle_p.intersection(self.table[arg2_id].shuffle_p)
            if len(a):
                return a.pop()
            id1 = self.count
            self.count += 1
            new = DNode(ID_SHUFFLE, arg1_id, arg2_id)
            self.table[id1] = new
            #  new.arg1, new.arg2 = arg1_id, arg2_id
            self.ewpFixConc(new, arg1_id, arg2_id)
            self.table[arg1_id].shuffle_p.add(id1)
            self.table[arg2_id].shuffle_p.add(id1)
            new.diff = self.shuffleLF(arg1_id, arg2_id)
            return id1

    def catLF(self, idl, idr, delay=False):
        """Linear form for concatenation
        Args:
            idl (int): node
            idr (int): node
            delay (bool): if true partial derivatives are delayed
        Returns:
            dict: partial derivatives

        ..note:: both arguments are assumed to be already present in the DAG"""
        nlf = dict()
        left = self.table[idl].diff
        for c in left:
            nlf[c] = {self.getConcatIdx(x, idr, delay) if x != 0 else idr for x in left[c]}
        return nlf

    @staticmethod
    def plusLF(diff1, diff2):
        """ Union of partial derivatives

        :arg dict diff1: partial diff of the first argument
        :arg dict diff2: partial diff of the second argument
        :rtype: dict
        """
        nfl = dict(diff1)
        for c in diff2:
            nfl[c] = nfl.get(c, set([])).union(diff2[c])
        return nfl

    def interLF(self, diff1, diff2):
        """ Intersection of partial derivatives

        :arg dict diff1: partial diff of the first argument
        :arg dict diff2: partial diff of the second argument
        :rtype: dict
        """
        nfl = dict()
        for c in set(diff1.keys()) & set(diff2.keys()):
            # if (s1 != 0 or not self.table[s2].ewp)
            nfl[c] = {self.getConjIdx(s1, s2) for s1 in diff1[c] for s2 in diff2[c]}
        return nfl

    def shuffleLF(self, id1, id2):
        """ Shuffle of partial derivatives
        Args:
            id1 (int): node
            id2 (int): node
        Returns:
            dict: partial derivatives
        """
        nfl = dict()
        left = self.table[id1].diff
        for c in left:
            nfl[c] = {self.getShuffleIdx(x, id2) if x != 0 else id2 for x in left[c]}
        if id1 == id2:
            return nfl
        right = self.table[id2].diff
        for c in right:
            nfl[c] = nfl.get(c, set([])).union({self.getShuffleIdx(id1, x) if x != 0 else id1 for x in right[c]})
        return nfl

    def doDelayed0(self):
        if len(self.diff2do):
            self._beingDone = set(self.diff2do)
            self.diff2do = set()
            while self._beingDone:
                inode = self._beingDone.pop()
                node = self.table[inode]
                # assert node.op == ID_CONC
                if self.table[node.arg1].ewp:
                    node.diff = self.plusLF(self.catLF(node.arg1, node.arg2), self.table[node.arg2].diff)
                else:
                    node.diff = self.catLF(node.arg1, node.arg2)
            self.doDelayed()

    def doDelayed(self):
        while self.diff2do:
            inode = self.diff2do.pop()
            node = self.table[inode]
            assert node.op == ID_CONC
            if self.table[node.arg1].ewp:
                node.diff = self.plusLF(self.catLF(node.arg1, node.arg2), self.table[node.arg2].diff)
            else:
                node.diff = self.catLF(node.arg1, node.arg2)

    def NFA(self):
        """
        Returns:
            NFA: the partial derivative automaton
        """
        aut = fa.NFA()
        if self.Sigma is not None:
            aut.setSigma(self.Sigma)
        todo, done = {self.root}, set()
        id1 = aut.addState(self.root)
        aut.addInitial(id1)
        while len(todo):
            st = todo.pop()
            done.add(st)
            stn = self.table[st]
            sti = aut.stateIndex(st)
            if stn.ewp:
                aut.addFinal(sti)
            for c in stn.diff:
                for dest in stn.diff[c]:
                    if dest not in todo and dest not in done:
                        todo.add(dest)
                    desti = aut.stateIndex(dest, True)
                    aut.addTransition(sti, c, desti)
        aut.epsilon_transitions = False
        return aut

        def evalWordP(self, w):
            """
            Evaluation of a word using the DAG
            Args:
                w (str): a word
            Returns:
                  bool: True if w in l(reg)
            """
            todo = {self.root}
            for c in w:
                new = set()
                for id in todo:
                    new |= self.table[id].diff.get(c, set())
                todo = new
                if todo == set():
                    break
            for id in todo:
                if self.table[id].ewp:
                    return True
            return False

class DAG_I(DAG):
    """Class to support dags representing regexps that inherit from DAG
        Partial derivatives are buid incrementally
    """
    def getStarIdx(self, arg_id):
        if arg_id == 0 or arg_id == 1:
            return 0
        if self.table[arg_id].star is not None:
            return self.table[arg_id].star
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_STAR, arg_id)
            self.table[arg_id].star = id1
            self.table[id1] = new
            return id1

    def getOptionIdx(self, arg_id):
        if self.table[arg_id].option is not None:
            return self.table[arg_id].option
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_OPTION, arg_id)
            self.table[arg_id].option = id1
            self.table[id1] = new
            return id1

    def getConcatIdx(self, arg1_id, arg2_id, delay=False):
        if arg1_id == 0:
            return arg2_id
        if arg2_id == 0:
            return arg1_id
        a = self.table[arg1_id].dotl.intersection(self.table[arg2_id].dotr)
        if len(a):
            return a.pop()
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_CONC, arg1_id, arg2_id)
            self.table[id1] = new
            #  new.arg1, new.arg2 = arg1_id, arg2_id
            self.ewpFixConc(new, arg1_id, arg2_id)
            self.table[arg1_id].dotl.add(id1)
            self.table[arg2_id].dotr.add(id1)
            return id1

    def ewpFixConc(self, obj, arg1_id, arg2_id):
        obj.ewp = self.table[arg1_id].ewp and self.table[arg2_id].ewp


    def getDisjIdx(self, arg1_id, arg2_id):
        if arg1_id == arg2_id:
            return arg1_id
        a = self.table[arg1_id].plus.intersection(self.table[arg2_id].plus)
        if len(a):
            return a.pop()
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_DISJ, arg1_id, arg2_id)
            self.table[id1] = new
        #    new.arg1, new.arg2 = arg1_id, arg2_id
            self.ewpFixDisj(new, arg1_id, arg2_id)
            self.table[arg1_id].plus.add(id1)
            self.table[arg2_id].plus.add(id1)
        return id1

    def ewpFixDisj(self, obj, arg1_id, arg2_id):
        obj.ewp = self.table[arg1_id].ewp or self.table[arg2_id].ewp

    def getConjIdx(self, arg1_id, arg2_id):
        if arg1_id == arg2_id:
            return arg1_id
        a = self.table[arg1_id].inter.intersection(self.table[arg2_id].inter)
        if len(a):
            return a.pop()
        else:
            id1 = self.count
            self.count += 1
            new = DNode(ID_CONJ, arg1_id, arg2_id)
            self.table[id1] = new
            self.ewpFixConc(new, arg1_id, arg2_id)
            self.table[arg1_id].inter.add(id1)
            self.table[arg2_id].inter.add(id1)
            return id1

    def getShuffleIdx(self, arg1_id, arg2_id):
        if arg1_id == 0:
            return arg2_id
        if arg2_id == 0:
            return arg1_id
        if arg1_id == arg2_id:
            if self.table[arg1_id].shuffle_u is not None:
                return self.table[arg1_id].shuffle_u
            else:
                id1 = self.count
                self.count += 1
                new = DNode(ID_SHUFFLE, arg1_id, arg2_id)
                self.table[id1] = new
                new.ewp = self.table[arg1_id].ewp
                self.table[arg1_id].shuffle_u = id1
                return id1
        else:
            a = self.table[arg1_id].shuffle_p.intersection(self.table[arg2_id].shuffle_p)
            if len(a):
                return a.pop()
            id1 = self.count
            self.count += 1
            new = DNode(ID_SHUFFLE, arg1_id, arg2_id)
            self.table[id1] = new
            #  new.arg1, new.arg2 = arg1_id, arg2_id
            self.ewpFixConc(new, arg1_id, arg2_id)
            self.table[arg1_id].shuffle_p.add(id1)
            self.table[arg2_id].shuffle_p.add(id1)
            return id1

    def one_derivative(self, id, c):
        """
        Args:
            c (str): a symbol
            id (int): a node (representing a regexp)
        """
        if id == 0 or id == 1:
            return set([])
        if c in self.table[id].diff:
            return self.table[id].diff[c]
        else:
            if self.table[id].op == ID_SYMB:  # id != c
                self.table[id].diff[c] = set([])
                return set([])
            elif self.table[id].op == ID_DISJ:
                nfl = self.one_derivative(self.table[id].arg1, c) | self.one_derivative(self.table[id].arg2, c)
            elif self.table[id].op == ID_CONC:
                nfl = self.cat_one(self.table[id].arg1, self.table[id].arg2, c)
                if self.table[self.table[id].arg1].ewp:
                    nfl |= self.one_derivative(self.table[id].arg2, c)
            elif  self.table[id].op == ID_SHUFFLE:
                nfl = self.shuffle_one(self.table[id].arg1, self.table[id].arg2, c)
            elif  self.table[id].op == ID_CONJ:
                left =  self.one_derivative(self.table[id].arg1, c)
                right =  self.one_derivative(self.table[id].arg2, c)
                nfl = {self.getConjIdx(s1, s2) for s1 in left for s2 in right}
            elif self.table[id].op == ID_STAR:
                nfl = self.cat_one(self.table[id].arg1, id, c)
            elif self.table[id].op == ID_OPTION:
                nfl = set(self.one_derivative(self.table[id].arg1, c))
            self.table[id].diff[c] = nfl
            return nfl

    def cat_one(self, idl, idr,c):
        """Partial derivative by one symbol for concatenation
        Args:
            idl (int): node
            idr (int): node
            c (char): symbol
        Returns:
            set: partial derivatives
        """
        left = self.one_derivative(idl, c)
        return  {self.getConcatIdx(x, idr) if x != 0 else idr for x in left}

    def shuffle_one(self, id1, id2, c):
        """ Shuffle of partial derivatives
        Args:
            id1 (int): node
            id2 (int): node
            c (char): symbol
        Returns:
            set: of partial derivatives
        """
        left =  self.one_derivative(id1, c)
        nfl= {self.getShuffleIdx(x, id2) if x != 0 else id2 for x in left}
        if id1 == id2:
            return nfl
        right =  self.one_derivative(id2, c)
        nfl |= {self.getShuffleIdx(id1, x) if x != 0 else id1 for x in right}
        return nfl

    def evalWordP(self, w):
        """
        Args:
            w (str): a word
        Returns:
              bool: True if w in L(reg)
        """
        todo = {self.root}
        for c in w:
            new = set()
            for id in todo:
                new |= self.one_derivative(id, c)
            todo = new
            if todo == set():
                break
        for id in todo:
            if self.table[id].ewp:
                return True
        return False

class BuildRegexp(lark.Transformer):
    """ Semantics of the FAdo grammars' regexps
        Priorities of operators: disj > conj > shuffle > concat > not > star >= option

    """

    def __init__(self, context=None):
        super(BuildRegexp, self).__init__()
        if context is None:
            context = dict()
        self.context = context
        if "sigma" in self.context:
            self.sigma = self.context["sigma"]
        else:
            self.sigma = None

    @staticmethod
    def rege(s):
        return s[0]

    epsilon = lambda self, _: CEpsilon(self.sigma)

    emptyset = lambda self, _: CEmptySet(self.sigma)

    sigmap = lambda self, _: CSigmaP(self.sigma)

    sigmas = lambda self, _: CSigmaS(self.sigma)

    @staticmethod
    def base(s):
        return s[0]

    def symbol(self, s):
        (s,) = s
        r = CAtom(s[:], self.sigma)
        r._ewp = False
        return r

    def disj(self, s):
        (arg1, arg2) = s
        r = CDisj(arg1, arg2, self.sigma)
        r._ewp = arg1._ewp or arg2._ewp
        return r

    def concat(self, s):
        (arg1, arg2) = s
        r = CConcat(arg1, arg2, self.sigma)
        r._ewp = arg1._ewp and arg2._ewp
        return r

    def shuffle(self, s):
        (arg1, arg2) = s
        r = CShuffle(arg1, arg2, self.sigma)
        r._ewp = arg1._ewp and arg2._ewp
        return r

    def u_shuffle(self, s):
        r = CShuffleU(s[0], self.sigma)
        r._ewp = s[0]._ewp
        return r

    def conj(self, s):
        (arg1, arg2) = s
        r = CConj(arg1, arg2, self.sigma)
        r._ewp = arg1._ewp and arg2._ewp
        return r

    def option(self, s):
        r = COption(s[0], self.sigma)
        r._ewp = True
        return r

    def notn(self, s):
        arg = s[0]
        r = Compl(arg, self.sigma)
        r._ewp = not arg._ewp
        return r

    def star(self, s):
        r = CStar(s[0], self.sigma)
        r._ewp = True
        return r


class BuildRPNRegexp(BuildRegexp):
    pass


class BuildSRE(BuildRegexp):
    """Parser for sre """

    @staticmethod
    def base(s):
        return s[0]

    def symbol(self, s):
        (s,) = s
        return CAtom(s[:], self.sigma)

    def disj(self, s):
        (arg1, arg2) = s
        return arg1._plus(arg2)

    def concat(self, s):
        (arg1, arg2) = s
        return arg1._dot(arg2)

    def conj(self, s):
        (arg1, arg2) = s
        return arg1._inter(arg2)

    def star(self, s):
        return SStar(s[0], self.sigma)

    def notn(self, s):
        return SNot(s[0], self.sigma)

    def shuffle(self, s):
        return s[0]


class BuildRPNSRE(BuildSRE):
    pass


regGrammar = lark.Lark.open("regexp_grammar.lark", rel_to=__file__, start="rege", parser="lalr")
regRPNGrammar = lark.Lark.open("regexp_grammar.lark", rel_to=__file__, start="regrpn", parser="lalr")


def str2regexp(s, parser=regGrammar, sigma=None, strict=False):
    """ Reads a RegExp from string.

        :arg s:  the string representation of the regular expression
        :type s: string
        :arg parser: a parser generator for regexps
        :arg sigma: alphabet of the regular expression
        :type sigma: list or set of symbols
        :arg strict: if True tests if the symbols of the regular expression are included in sigma
        :type strict: boolean
        :rtype: reex.RegExp
        """
    tree = parser.parse(s)
    reg = RegExp()
    if parser == regGrammar:
        reg = BuildRegexp(context={"sigma": sigma}).transform(tree)
    elif parser == regRPNGrammar:
        reg = BuildRPNRegexp(context={"sigma": sigma}).transform(tree)
    if sigma is not None:
        reg.setSigma(sigma, strict)
    else:
        reg.setSigma(reg.setOfSymbols())
    return reg


def str2sre(s, parser=regGrammar, sigma=None, strict=False):
    """ Reads a sre from string. Arguments as str2regexp.

    :rtype: reex.sre"""
    tree = parser.parse(s)
    reg = RegExp()
    if parser == regGrammar:
        reg = BuildSRE(context={"sigma": sigma}).transform(tree)
    elif parser == regRPNGrammar:
        reg = BuildRPNSRE(context={"sigma": sigma}).transform(tree)
    if sigma is not None:
        reg.setSigma(sigma, strict)
    else:
        reg.setSigma(reg.setOfSymbols(), strict)
    return reg


def equivalentP(first, second):
    """Verifies if the two languages given by some representative (DFA, NFA or re) are equivalent

    :arg first: language
    :arg second: language
    :rtype: bool

    .. versionadded:: 0.9.6"""
    t1, t2 = type(first), type(second)
    if t1 == t2 or (issubclass(t1, RegExp) and issubclass(t2, RegExp)):
        return first.equivalentP(second)
    elif t1 == fa.DFA:
        return first == second.toDFA()
    elif t1 == fa.NFA:
        if t2 == fa.DFA:
            return first.toDFA() == second
        else:
            return first == second.toNFA()
    else:
        if t2 == fa.NFA:
            return first.toNFA() == second
        else:
            return first.toDFA() == second
