# -*- coding: utf-8 -*-
"""**Synchronous Kleene Algebra Support**

SKA terms manipulation.

.. versionadded:: 1.0

.. *Authors:* Rogério Reis & Nelma Moreira

.. Contributions by
    - Sabine Broda
    - Sílvia Cavadas

.. *This is part of FAdo project*   http://fado.dcc.fc.up.pt.

.. *Copyright:* 1999-2014 Rogério Reis & Nelma Moreira {rvr,nam}@dcc.fc.up.pt

.. see_also:: Synchronous Kleene Algebra

.. This program is free software; you can redistribu>te it and/or
   modify it under the terms of the GNU General Public License as published
   by the Free Software Foundation; either version 2 of the License, or
   (at your COption) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   675 Mass Ave, Cambridge, MA 02139, USA."""
# from itertools import *
from . import reex
from . import fa
from . common import *
from . cfg import *
import lark

grse_rpn = ["Tx -> {0:s} | Ti | +  Tx  Tx  | & Tx Tx | . Tx Tx  | * Tx".format(Epsilon)]

grs_rpn = ["Tx ->  Ti | +  Tx  Tx  | & Tx Tx | . Tx Tx  | * Tx"]


def powersetNotN(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def relabelStr(aut):
    """Relabel transitions """
    d = aut.dup()
    for s in aut.delta:
        for k in aut.delta[s]:
            del d.delta[s][k]
            d.delta[s][hash(k)] = aut.delta[s][k]
    return d


class CAtom(reex.CAtom):
    """ Class for symbols"""
    def setSigma(self, symbolset, strict=True):
        """
            :param symbolset:
            :param strict:
            :raise regexpInvalidSymbols:
            """
        if symbolset is not None:
            if strict and not (self.setOfSymbols() <= symbolset):
                raise regexpInvalidSymbols()
            self.Sigma = powersetNotN(symbolset)
        else:
            raise regexpInvalidSymbols()

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def nfaThompson(self):
        """Epsilon-NFA constructed with Thompson's method that accepts the regular expression's language.

            :rtype: NFA

            """
        return eliminateEpsilon(super(CAtom, self).nfaThompson())


class CDisj(reex.CDisj):
    """ Class for disjunction """
    def setSigma(self, symbolset, strict=True):
        """
            :param symbolset:
            :param strict:
            :raise regexpInvalidSymbols:
            """
        if symbolset is not None:
            if strict and not (self.setOfSymbols() <= symbolset):
                raise regexpInvalidSymbols()
            self.Sigma = powersetNotN(symbolset)
        else:
            raise regexpInvalidSymbols()

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def nfaThompson(self):
        return eliminateEpsilon(super(CDisj, self).nfaThompson())


class CConcat(reex.CConcat):
    """Class for concatenation"""

    def setSigma(self, symbolset, strict=True):
        """
            :param symbolset:
            :param strict:
            :raise regexpInvalidSymbols:
            """
        if symbolset is not None:
            if strict and not (self.setOfSymbols() <= symbolset):
                raise regexpInvalidSymbols()
            self.Sigma = powersetNotN(symbolset)
        else:
            raise regexpInvalidSymbols()

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def nfaThompson(self):
        return eliminateEpsilon(super(CConcat, self).nfaThompson())


class CStar(reex.CStar):
    """Class for CStar"""

    def setSigma(self, symbolset, strict=True):
        """
            :param symbolset:
            :param strict:
            :raise regexpInvalidSymbols:
            """
        if symbolset is not None:
            if strict and not (self.setOfSymbols() <= symbolset):
                raise regexpInvalidSymbols()
            self.Sigma = powersetNotN(symbolset)
        else:
            raise regexpInvalidSymbols()

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def nfaThompson(self):
        """ Returns a NFA that accepts the RE.
            :rtype: NFA

            """
        sun = self.arg.nfaThompson()
        au = sun.dup()
        (s0, s1) = (au.addState(), au.addState())
        if self.Sigma is not None:
                au.setSigma(self.Sigma)
        au_initial = au.Initial.pop()
        au.addTransition(s0, Epsilon, s1)
        #au.addTransition(s1, Epsilon, s0)
        au.addTransition(list(au.Final)[0], Epsilon, au_initial)
        au.addTransition(s0, Epsilon, au_initial)
        au.addTransition(list(au.Final)[0], Epsilon, s1)  # we know by contruction
        au.setInitial([s0])  # that there is only one final state,
        au.setFinal([s1])  # and only one initial state
        return eliminateEpsilon(au)


class epsilon(reex.CEpsilon):

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def nfaThompson(self):
        return eliminateEpsilon(super(epsilon, self).nfaThompson())


class emptyset(reex.CEmptySet):

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def nfaThompson(self):
        return eliminateEpsilon(super(emptyset, self).nfaThompson())


class sync(reex.Connective):
    """ Class for synchronous product

    """

    def __init__(self, arg1, arg2, sigma=None):
        super(sync, self).__init__(arg1, arg2, sigma)
        self.Power = None

    def __str__(self):
        return "{0:s} & {1:s}".format(self.arg1._strP(), self.arg2._strP())

    def _strP(self):
        return "({0:s} & {1:s})".format(self.arg1._strP(), self.arg2._strP())

    def rpn(self):
        return "&%s%s" % (self.arg1.rpn(), self.arg2.rpn())

    def ewp(self):
        return self.arg1.ewp() and self.arg2.ewp()

    def nfaPD(self, pdmethod="nfaPDNaive"):
        """
        Computes the partial derivative automaton
        """
        return self.__getattribute__(pdmethod)()

    def partialDerivatives(self, sigma):
        pdset = self.arg1.partialDerivatives(sigma)
        pdset.update(self.arg2.partialDerivatives(sigma))
        return pdset

    def linearForm(self):
        def _dotsync(s1, s2):
            if s1.epsilonP():
                a = s2
            elif s2.epsilonP():
                a = s1
            else:
                a = sync(s1, s2)
            return a

        arg1_lf = self.arg1.linearForm()
        arg2_lf = self.arg2.linearForm()
        lf = dict()
        for k3 in arg1_lf:
            for k4 in arg2_lf:
                k = tuple(set(k3).union(set(k4)))
                if len(k) == 1:
                       k = k[0]
                tails = {_dotsync(r1, r2) for r1 in arg1_lf.get(k3, set()) for r2 in arg2_lf.get(k4, set())}
                if tails != set():
                    if k in lf:
                        lf[k].update(tails)
                    else:
                        lf[k] = tails
        if self.arg1.ewp():
            for head in arg2_lf:
                if head in lf:
                    lf[head].update(arg2_lf[head])
                else:
                    lf[head] = set(arg2_lf[head])
        if self.arg2.ewp():
            for head in arg1_lf:
                if head in lf:
                    lf[head].update(arg1_lf[head])
                else:
                    lf[head] = set(arg1_lf[head])
        return lf

    def nfaThompson(self):
        """ Returns an NFA (Thompson) that accepts the RE.
        :rtype: NFA

        """

        def _sproduct(a1, a2):
            """Returns a NFA (skeletom) resulting of the simultaneous execution of two DFA.

                  :param a1,a2: the other automata
                  :type a1, a2: NFA
                  :rtype: NFA

                  .. note::
                     No final states are set.

                  .. attention::
                     - the name ``EmptySet`` is used in a unique special state name
                     - the method uses 3 internal functions for simplicity of code (really!)"""

            def _sN(a, s):
                try:
                    j = a.stateIndex(s)
                except DFAstateUnknown:
                    return None
                return j

            def _kS(a, j):
                """
                      :param a:
                      :param j:
                      :return:"""
                if j is None:
                    return set()
                try:
                    ks = list(a.delta[j].keys())
                except KeyError:
                    return set()
                return set(ks)

            def _dealT(srcI, dest):
                """
                      :param srcI: source state
                      :param dest: destination state"""
                if not (dest in done or dest in notDone):
                    ind = new.addState(dest)
                    notDone.append(dest)
                else:
                    ind = new.stateIndex(dest)
                new.addTransition(srcI, k, ind)

            new = fa.NFA()
            # new.setSigma(a1.sigma.union(a2.sigma))
            notDone = []
            done = []
            for s1 in [a1.States[x] for x in a1.Initial]:
                for s2 in [a2.States[x] for x in a2.Initial]:
                    sname = (s1, s2)
                    iN = new.addState(sname)
                    new.addInitial(iN)
                    if (s1, s2) not in notDone:
                        notDone.append((s1, s2))
            while notDone:
                state = notDone.pop()
                done.append(state)
                (s1, s2) = state
                i = new.stateIndex(state)
                (i1, i2) = (_sN(a1, s1), _sN(a2, s2))
                (k1, k2) = (_kS(a1, i1), _kS(a2, i2))
                for k3 in k1:
                    for k4 in k2:
                        for destination in [(a1.States[d1], a2.States[d2]) for d1 in a1.delta[i1][k3] for d2 in
                                            a2.delta[i2][k4]]:
                            if k3 == Epsilon:
                                k = k4
                            elif k4 == Epsilon:
                                k = k3
                            else:
                                k = tuple(set(k3).union(set(k4)))
                                if len(k) == 1:
                                   k = k[0]
                            _dealT(i, destination)
                if a1.finalP(i1) and not a2.finalP(i2):
                    for k in k2:
                        for n in a2.delta[i2][k]:
                            _dealT(i, (s1, a2.States[n]))
                elif a2.finalP(i2) and not a1.finalP(i1):
                    for k in k1:
                        for n in a1.delta[i1][k]:
                            _dealT(i, (a1.States[n], s2))
                elif a2.finalP(i2) and a1.finalP(i1):
                    new.addFinal(i)
            return new

        a1 = eliminateEpsilon(self.arg1.nfaThompson())
        a2 = eliminateEpsilon(self.arg2.nfaThompson())
        return _sproduct(a1, a2)


class BuildSKA(reex.BuildRegexp):
    """ Semantics of the SKA grammar"""

    def sync(self, s):
        (arg1, arg2) = s
        return sync(arg1, arg2, self.sigma)

    def star(self, s):
        return CStar(s[0], self.sigma)

    def disj(self, s):
        (arg1, arg2) = s
        return CDisj(arg1, arg2, self.sigma)

    def concat(self, s):
        (arg1, arg2) = s
        return CConcat(arg1, arg2, self.sigma)

    def symbol(self, s):
        (s,) = s
        return CAtom(s[:], self.sigma)

class BuildSKARPN(BuildSKA):
    pass


ParserSKARPN = lark.Lark(
    r"""
                ?rege: disj | sync | concat | star | symbol 
                | epsilon | emptyset 

                disj: "+" rege rege | "|" rege rege
                concat: "." rege rege
                star: "*" rege
                sync: "&" rege rege
                symbol: /[a-zA-Z0-9]/

                epsilon: "@epsilon"
                emptyset: "@empty_set"
    
                %ignore /[ \t\f\"]+/
                """, start="rege")

ParserSKA = lark.Lark(
    r"""
    ?rege: disjn    
            ?disjn:  syncn 
            | rege "+" syncn  -> disj

            ?syncn: concatn
            | syncn "&" concatn -> sync

            ?concatn: rep
                | concatn ["."] rep -> concat

            ?rep: base
                | rep "*" -> star
              
            ?base: "(" rege ")" | symbol | epsilon | emptyset 

                symbol: /[a-zA-Z0-9]/

                epsilon: "@epsilon"
                emptyset: "@empty_set"

                %ignore /[ \t\f\"]+/
    """, start="rege", parser="lalr")


def eliminateEpsilon(aut):
    def closeEpsilon(aut, st):
        """Add all non CEpsilon transitions from the states in the CEpsilon closure of given state to given state.

            :param int st: state index
            :param FA aut: automaton
            """
        targets = aut.epsilonClosure(st)
        targets.remove(st)
        if not targets:
            return
        for target in targets:
            if not aut.finalP(target):
                aut.delTransition(st, Epsilon, target)
        for target in targets:
            if target in aut.delta:
                eps = False
                for symbol, states in list(aut.delta[target].items()):
                    if symbol is Epsilon:
                        eps = True
                    for st1 in states:
                        if (eps and aut.finalP(st1)) or not eps:
                            aut.addTransition(st, symbol, st1)
                    eps = False
        if targets.intersection(aut.Final) and Epsilon not in aut.delta.get(st, {}):
            aut.addTransition(st, Epsilon, list(aut.Final)[0])

    for state in range(len(aut.States)):
        closeEpsilon(aut, state)
    return aut.trim()


def test_gen(n=50, k=10, amount=50, grammar=grs_rpn, eps=None, empty=None):
    """

    :rtype : None
    Uniformly Random generated SKA terms and test Thompson and Partial Derivatives Constructions. Use: test_gen()
    :param n: size of ska terms
    :param k: size of Ab
    :param amount: number of terms generated
    :param grammar: SKA grammar to be used
    :param eps: if CEpsilon is allowed not None
    :param empty: if EmptySet is allowed not None
    :return:
    """
    print("n=", n, "k = ", k, "a= ", amount)
    gen = REStringRGenerator(smallAlphabet(k), n, grammar, eps, empty)
    for i in range(amount):
        gska = gen.generate()
        ska = str2ska(gska, parser=ParserSKARPN, strict=True)
        nt = ska.nfaThompson()
        np = ska.nfaPD()
        nt.Sigma.update(np.Sigma)
        np.setSigma(nt.Sigma)
        assert np.Sigma == nt.Sigma
        print(len(nt), nt.countTransitions(), len(np), np.countTransitions(), len(np) * 0.1 / len(nt))
        dt = relabelStr(nt)
        dp = relabelStr(np)
        dt.renameStates()
        dp.renameStates()
        if np != nt:
            print(gska, ska)


def timesAllSymbols(n=3):
    """ A SKA term with all sigma symbols for AB with size n"""
    Sigma = [("a", str(i)) for i in range(n)]
    sk = epsilon()
    for s in Sigma:
        sk = sync(sk, CDisj(epsilon(), CAtom(s)))
    di = epsilon()
    for p in powersetNotN(Sigma):
        di = CDisj(di, CAtom(p))
    c = CAtom(("a", str(1)))
    for i in range(1, n):
        c = CConcat(c, sk)
    return sk


def str2ska(s, parser=ParserSKA, sigma=None, strict=False):
    """Reads a ska from string. Arguments as str2regexp.

    :rtype: reex.ska"""
    tree = parser.parse(s)
    if parser == ParserSKA:
        reg = BuildSKA(context={"sigma": sigma}).transform(tree)
    elif parser == ParserSKARPN:
        reg = BuildSKARPN(context={"sigma": sigma}).transform(tree)
    else:
        raise FAdoGeneralError
    if sigma is not None:
        reg.setSigma(sigma, strict)
    else:
        reg.setSigma(reg.setOfSymbols())

    return reg
