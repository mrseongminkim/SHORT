# coding=utf-8
"""**Set Specification Transducer supportt**


.. versionadded:: 1.4

.. *Authors:* Rogério Reis, Nelma Moreira & Stavros Konstantinidis

.. *This is part of FAdo project*   http://fado.dcc.fc.up.pt.

.. *Copyright:* 1999-2018 Rogério Reis & Nelma Moreira {rvr,nam}@dcc.fc.up.pt

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

from . import transducers
from . import fa
from . common import *
import copy


class PSP(object):
    """Relation pair of set specifications"""
    @abstractmethod
    def inverse(self):
        pass

    @abstractmethod
    def isAInvariant(self):
        pass

    @abstractmethod
    def inIntersection(self, nfa, alph):
        pass

    def left(self, _):
        return self.Arg1

    def right(self, _):
        return self.Arg2

    @abstractmethod
    def behaviour(self, alph):
        pass


class PSPVanila(PSP):
    """Relation pair of two set specifications"""
    def __init__(self, arg1, arg2):
        """
        :type arg1: SetSpec
        :type arg2: SetSpec
        """
        self.Arg1 = arg1
        self.Arg2 = arg2

    def __repr__(self):
        return "PSPVanila"+self.__str__()

    def __str__(self):
        return "("+str(self.Arg1)+", "+str(self.Arg2)+")"

    def alphabet(self):
        """ The covering alphabet of a PSP

        :rtype: set"""
        return self.Arg1.alphabet() | self.Arg2.alphabet()

    def inverse(self):
        """Inverse of a PSP

        :rtype: PSPVanila
        """
        return PSPVanila(self.Arg2, self.Arg1)

    def behaviour(self, sigma):
        """ Expansion of a PSP

        :rtype: (set, set)"""
        l = set()
        for i1 in self.Arg1.behaviour(sigma):
            for i2 in self.Arg2.behaviour(sigma):
                l.add((i1, i2))
        return l

    def isAInvariant(self):
        """ Is this an alphabet invariant PSP?

        :rtype: bool"""
        return self.Arg1.isAInvariant() and self.Arg2.isAInvariant()

    def inIntersection(self, other, alph):
        """ Evaluates the intersect on input with another Set Specification

        :param SetSpec other: the other
        :param set alph: alphabet
        :rtype: PSP"""
        a = self.Arg1.intersection(other, alph)
        if type(a) is not SSEmpty:
            return PSPVanila(a, self.Arg2)
        else:
            return None


class PSPEqual(PSP):
    """Relation pair of two set specifications (constrained by equality)"""
    def __init__(self, arg1):
        """
        :type arg1: SetSpec
        """
        self.Arg1 = arg1

    def isAInvariant(self):
        return self.Arg1.isAInvariant()

    def __repr__(self):
        return "PSPEqual(" + str(self.Arg1) + ")"

    def symList(self):
        return self.Arg1.alphabet()

    def inverse(self):
        return self

    def behaviour(self, sigma):
        l = set()
        for i in self.Arg1.behaviour(sigma):
            l.add((i, i))
        return l

    def inIntersection(self, other, alph):
        """ Evaluates the intersect on input wit anothe Set Specification

        :param SetSpec other: the other
        :param set alph: alphabet
        :rtype: PSP"""
        a = self.Arg1.intersection(other, alph)
        if type(a) is not SSEmpty:
            return PSPEqual(a)
        else:
            return None

    def right(self, _):
        return self.Arg1


class PSPDiff(PSP):
    """Relation pair of two set specifications (constrained by non equality)"""
    def __init__(self, arg1, arg2):
        """
        :type arg1: SetSpec
        :type arg2: SetSpec
        """
        self.Arg1 = arg1
        self.Arg2 = arg2

    def isAInvariant(self):
        return self.Arg1.isAInvariant() and self.Arg2.isAInvariant()

    def __repr__(self):
        return "PSPDiff(" + str(self.Arg1) + "," + str(self.Arg2) + ")"

    def symList(self):
        return self.Arg1.alphabet() | self.Arg2.alphabet()

    def inverse(self):
        return PSPDiff(self.Arg2, self.Arg1)

    def behaviour(self, sigma):
        l = set()
        for i1 in self.Arg1.behaviour(sigma):
            for i2 in self.Arg2.behaviour(sigma):
                if i1 != i2:
                    l.add((i1, i2))
        return l

    def inIntersection(self, other, alph):
        """ Evaluates the intersect on input wit anothe Set Specification

        :param SetSpec other: the other
        :param set alph: alphabet
        :rtype: PSP"""
        a = self.Arg1.intersection(other, alph)
        if type(a) is not SSEmpty and not a.equivalentP(self.Arg2, alph):
            return PSPDiff(a, self.Arg2)
        else:
            return None

    def left(self, alph):
        if self.Arg1.equivalentP(self.Arg2, alph):
            return SSEmpty()
        if isinstance(self.Arg2, SSOneOf) and len(self.Arg2.Arg1) == 1:
            if isinstance(self.Arg1, SSAnyOf):
                return SSNoneOf(self.Arg2.Arg1)
            elif isinstance(self.Arg1, SSOneOf):
                return SSOneOf(self.Arg1.Arg1 - self.Arg2.Arg1)
            elif isinstance(self.Arg1, SSNoneOf):
                return SSNoneOf(self.Arg1.Arg1 | self.Arg2.Arg1)
        elif isinstance(self.Arg2, SSNoneOf) and len(self.Arg2.Arg1) == len(alph) - 1:
            return PSPDiff(self.Arg1, SSOneOf(alph - self.Arg2.Arg1))
        else:
            return self.Arg1

    def right(self, alph):
        return PSPDiff(self.Arg2, self.Arg1).left(alph)


class SetSpec(object):
    """Set Specification labels"""
    def alphabet(self):
        return set()

    @abstractmethod
    def behaviour(self, _):
        pass

    @abstractmethod
    def intersection(self, other, alph):
        pass

    @abstractmethod
    def witness(self, _):
        pass

    def isAInvariant(self):
        return True

    def __repr__(self):
        return self.__str__()

    def equivalentP(self, other, alph):
        if len(alph) == 1:
            return True
        else:
            return False

    @abstractmethod
    def collapse(self, other):
        pass


class SSOneOf(SetSpec):
    def __init__(self, oset):
        """Set specification for 'one of...'

        :type oset: list"""
        self.Arg1 = set(oset)

    def __repr__(self):
        return "SSOneOf("+str(self.Arg1)+")"

    def alphabet(self):
        return self.Arg1

    def behaviour(self, _):
        return self.Arg1

    def intersection(self, other, _):
        if type(other) == SSEmpty or type(other) == SSEpsilon:
            return SSEmpty()
        elif type(other) == SSAnyOf:
            return self
        elif type(other) == SSOneOf:
            foo = self.Arg1 & other.Arg1
            if not len(foo):
                return SSEmpty()
            else:
                return SSOneOf(foo)
        else:
            foo = self.Arg1 - other.Arg1
            if not len(foo):
                return SSEmpty()
            else:
                return SSOneOf(foo)

    def isAInvariant(self):
        return False

    def witness(self, _):
        return getOneFromSet(self.Arg1)

    def equivalentP(self, other, alph):
        if isinstance(other, SSOneOf) and self.Arg1 == other.Arg1:
                return True
        elif isinstance(other, SSNoneOf) and len(alph) == 2 and self.Arg1 == other.Arg1:
                return True
        return False

    def collapse(self, other):
        if isinstance(other, SSAnyOf):
            return other
        elif isinstance(other, SSOneOf):
            return SSOneOf(self.Arg1 | other.Arg1)
        elif isinstance(other, SSNoneOf):
            return SSNoneOf(other.Arg1 - self.Arg1)
        else:
            assert isinstance(other, SSEpsilon)
            return self


class SSAnyOf(SetSpec):
    """Set specification for 'any'
    """
    def __str__(self):
        return "SSAnyOf()"

    def alphabet(self):
        return set()

    @staticmethod
    def behaviour(sigma):
        return sigma

    @staticmethod
    def intersection(other, _):
        if type(other) == SSEpsilon:
            return SSEmpty()
        else:
            return other

    def witness(self, alph):
        return getOneFromSet(alph)

    def collapse(self, other):
        return self


class SSNoneOf(SetSpec):
    def __init__(self, oset):
        """Set specification for 'none of...'

        :type oset: list"""
        assert len(oset) > 0
        self.Arg1 = set(oset)

    def __str__(self):
        return "SSNOneOf("+str(self.Arg1)+")"

    def behaviour(self, sigma):
        return sigma - self.Arg1

    def intersection(self, other, alphabet=None):
        if type(other) == SSEmpty or type(other) == SSEpsilon:
            return SSEmpty()
        elif type(other) == SSAnyOf:
            return self
        elif type(other) == SSOneOf:
            foo = other.Arg1 - self.Arg1
            if not len(foo):
                return SSEmpty()
            else:
                return SSOneOf(foo)
        else:
            foo = self.Arg1 | other.Arg1
            try:
                l = len(alphabet)
            except TypeError:
                raise SSMissAlphabet()
            if l == len(foo):
                return SSEmpty
            else:
                return SSNoneOf(foo)

    def isAInvariant(self):
        return False

    def witness(self, alph):
        return getOneFromSet(alph - self.Arg1)

    def equivalentP(self, other, alph):
        if isinstance(other, SSNoneOf) and self.Arg1 == other.Arg1:
                return True
        elif isinstance(other, SSOneOf) and len(alph) == 2 and self.Arg1 == other.Arg1:
                return True
        return False

    def collapse(self, other):
        if isinstance(other, SSAnyOf):
            return other
        elif isinstance(other, SSNoneOf):
            return SSConditionalNoneOf(self.Arg1 & other.Arg1)
        elif isinstance(other, SSOneOf):
            return SSConditionalNoneOf(self.Arg1 - other.Arg1)
        else:
            assert isinstance(other, SSEpsilon)
            return self


def SSConditionalNoneOf(oset):
    """Auxiliary function that coalesces an SSNoneOf into an SSAnyOf if oset is empty"""
    if len(oset):
        return SSNoneOf(oset)
    else:
        return SSAnyOf()


class SSEmpty(SetSpec):
    def __str__(self):
        return "SSEmpty"

    def behaviour(self, _):
        return set()

    def intersection(self, _a, _b):
        return self

    def witness(self, _):
        raise SSBadTransition()

    def collapse(self, other):
        raise SSError("Empty transition")


class SSEpsilon(SetSpec):
    def __str__(self):
        return "SSEpsilon"

    def behaviour(self, _):
        return {Epsilon}

    def intersection(self, other, _):
        if type(other) == SSEpsilon:
            return self
        else:
            return SSEmpty()

    def witness(self, _):
        return Epsilon

    def collapse(self, other):
        return other


class SST(transducers.SFT):
    """SFT with set specification labels

    .. inheritance-diagram:: SST"""
    def __init__(self, sigma=None):
        """

        :type sigma: list
        """
        super(SST, self).__init__()
        if sigma is None:
            sigma = []
        self.Sigma = set()
        if sigma is not None:
            for c in sigma:
                self.addToSigma(c)

    def __str__(self):
        """Return a string representing the details of the current transducer instance.

        :rtype: str"""
        return str((self.States, self.Sigma, self.Initial, self.Final, self.delta))

    def __repr__(self):
        """Return a string adding type 'Transducer'in front of the representation

        :rtype: str"""
        return 'SFT(%s)' % self.__str__()

    def addToSigma(self, sym):
        """ Adds a new symbol to the alphabet (it it is not already there)
        
        :param unicode sym: symbol to add
        :rtype: int
        :returns: the index of the new symbol
        """
        self.Sigma.add(sym)

    def addSigmaPair(self, pair):
        for sy in pair.alphabet():
            self.addToSigma(sy)

    def addTransition(self, stsrc, pair, sti2):
        """

        :type stsrc: int
        :type pair: sst.PSP
        :param sti2: int
        """
        if stsrc not in self.delta:
            self.delta[stsrc] = {pair: {sti2}}
        elif pair not in self.delta[stsrc]:
            self.delta[stsrc][pair] = {sti2}
        else:
            self.addSigmaPair(pair)
            self.delta[stsrc][pair].add(sti2)

    def toSFT(self):
        """ Expands a SST to an SFT

        :rtype: SFT"""
        new = transducers.SFT()
        new.setSigma(self.Sigma)
        new.setOutput(self.Sigma)
        new.States = copy.copy(self.States)
        for s1 in self.delta:
            for t in self.delta[s1]:
                for s2 in self.delta[s1][t]:
                    for (c1, c2) in t.behaviour(self.Sigma):
                        new.addTransition(s1, c1, c2, s2)
        new.setInitial(self.Initial)
        new.setFinal(self.Final)
        return new

    def reversal(self):
        new = SST(self.Sigma)
        new.States = copy.copy(self.States)
        for s1 in self.delta:
            for t in self.delta[s1]:
                for s2 in self.delta[s1][t]:
                    new.addTransition(s2, t.inverse(), s1)
        new.setInitial(self.Final)
        new.setFinal(self.Initial)
        return new

    def inverse(self):
        new = SST(self.Sigma)
        new.States = copy.copy(self.States)
        for s1 in self.delta:
            for t in self.delta[s1]:
                for s2 in self.delta[s1][t]:
                    new.addTransition(s1, t.inverse(), s2)
        new.setInitial(self.Initial)
        new.setFinal(self.Final)
        return new

    def productInput(self, other):
        """Returns a transducer (skeleton) resulting from the execution of the transducer with the automaton as
        filter on the input.

        .. note:: This version does not use stateIndex() with the price of generating some unreachable sates

        :param SSFA other: the automaton used as filter
        :rtype: SST

        .. versionchanged:: 1.3.3"""
        new = SST(self.Sigma.union(other.Sigma))
        notDone = set()
        done = set()
        sz2 = len(other.States)
        for _ in range(len(self.States) * sz2):
            new.addState()
        for s1 in self.Initial:
            for s2 in other.Initial:
                new.addInitial(s1*sz2+s2)
                notDone.add((s1, s2))
        while notDone:
            state = notDone.pop()
            done.add(state)
            (s1, s2) = state
            sti = s1*sz2+s2
            for t1 in self.delta.get(s1, {}):
                for t2 in other.delta.get(s2, {}):
                    a = t1.inIntersection(t2, new.Sigma)
                    if a is not None:
                        for o1 in self.delta[s1][t1]:
                            for o2 in other.delta[s2][t2]:
                                new.addTransitionProductQ(sti, o1*sz2+o2, (o1, o2), a, notDone, done)
        return new

    def inIntersection(self, other):
        """ Conjunction of transducer and automata: X & Y.

        .. note:: This is a fast version of the method that does not produce meaningfull state names.

        .. note:: The resulting transducer is not trim.

        :param DFA|NFA other: the automata needs to be operated.
        :rtype: SFT"""
        if isinstance(other, fa.DFA):
            nother = NFA2SSFA(other.toNFA().renameStates())
        elif isinstance(other, fa.NFA):
            nother = NFA2SSFA(other.renameStates())
        elif isinstance(other, SSFA):
            nother = other.renameStates()
        else:
            raise FAdoGeneralError("Incompatible objects")
        et, en = self.epsilonP(), nother.epsilonP()
        if en:
            par1 = copy.copy(self)
            par1.addEpsilonLoops()
        else:
            par1 = self
        if et:
            par2 = copy.copy(nother)
            par2.addEpsilonLoops()
        else:
            par2 = nother
        new = par1.productInput(par2)
        sz2 = len(par2.States)
        for s1 in par1.Final:
            for s2 in par2.Final:
                new.addFinal(s1*sz2+s2)
        return new

    def epsilonP(self):
        for g in self.delta.values():
            for t in g:
                if isinstance(t, PSPVanila) and isinstance(t.Arg1, SSEpsilon):
                    return True
        return False

    def epsilonOutP(self):
        for g in self.delta.values():
            for t in g:
                if isinstance(t, PSPVanila) and isinstance(t.Arg2, SSEpsilon):
                    return True
        return False

    def addTransitionProductQ(self, src, dest, ddest, sym, futQ, pastQ):
        """Add transition to the new transducer instance.

        Version for the optimized product

        :param src: source state
        :param dest: destination state
        :param ddest: destination as tuple
        :param sym: symbol
        :param set futQ: queue for later
        :param set pastQ: past queue"""
        if ddest not in pastQ:
            futQ.add(ddest)
        self.addTransition(src, sym, dest)

    def addEpsilonLoops(self):
        for i in self.stateIndexes():
            self.addTransition(i, PSPVanila(SSEpsilon(), SSEpsilon()), i)

    def outIntersection(self, other):
        return self.outIntersectionDerived(other)

    def toInSSFA(self):
        """Delete the output labels in the transducer. Translate it into an SSFA

        :rtype: SSFA"""
        return self.toXSSFA("left")

    def toInNFA(self):
        return self.toInSSFA()

    def toOutSSFA(self):
        """Returns the result of considering the output symbols of the transducer as input symbols of a SSFA (ignoring
        the input symbol, thus)

        :return: the SSFA
        :rtype: SSFA"""
        return self.toXSSFA("right")

    def toOutNFA(self):
        return self.toOutSSFA()

    def toXSSFA(self, side):
        """ Skeleton of a method that extracts both left & right language of a PSP """
        aut = SSFA(self.Sigma)
        aut.States = copy.copy(self.States)
        aut.setInitial(self.Initial)
        aut.setFinal(self.Final)
        for s in self.delta:
            aut.delta[s] = {}
            for t in self.delta[s]:
                f = getattr(t, side)
                aut.delta[s][f(self.Sigma)] = copy.copy(self.delta[s][t])
        return aut

    def nonEmptyW(self):
        """Witness of non emptyness

        :return: pair (in-word, out-word)
        :rtype: tuple"""
        done = set()
        notDone = set()
        pref = dict()
        for si in self.Initial:
            pref[si] = (Epsilon, Epsilon)
            notDone.add(si)
        while notDone:
            si = notDone.pop()
            done.add(si)
            if si in self.Final:
                return pref[si]
            for syi in self.delta.get(si, []):
                for so in self.delta[si][syi]:
                    if so in done or so in notDone:
                        continue
                    ex = getOneFromSet(syi.behaviour(self.Sigma))
                    pref[so] = transducers.concatN(pref[si], ex)
                    notDone.add(so)
        return None, None


class SSFA(fa.NFA):
    """ NFAs with Set Specifications as transition labels"""

    def __init__(self, alph):
        """

        :param alph: alphabet
        """
        super(SSFA, self).__init__()
        self.Sigma = set(list(alph))

    def addTransition(self, sti1, spec, sti2):
        """ Add af Set Specification transition

        :param int sti1: start state index
        :param int sti2: end state index
        :param SetSpec spec: symbolic spec"""
        if spec != Epsilon:
            for c in spec.alphabet():
                self.addSigma(c)
        if sti1 not in self.delta:
            self.delta[sti1] = {spec: {sti2}}
        elif spec not in self.delta[sti1]:
            self.delta[sti1][spec] = {sti2}
        else:
            self.delta[sti1][spec].add(sti2)

    def witness(self):
        """Witness of non emptyness

        :return: word
        :rtype: str"""
        done = set()
        notDone = set()
        pref = dict()
        for si in self.Initial:
            pref[si] = Epsilon
            notDone.add(si)
        while notDone:
            si = notDone.pop()
            done.add(si)
            if si in self.Final:
                return pref[si]
            for syi in self.delta.get(si, []):
                for so in self.delta[si][syi]:
                    if so in done or so in notDone:
                        continue
                    pref[so] = sConcat(pref[si], syi.witness(self.Sigma))
                    notDone.add(so)
        return None

    def addEpsilonLoops(self):
        for i in self.stateIndexes():
            self.addTransition(i, SSEpsilon(), i)

    def epsilonP(self):
        for g in self.delta.values():
            for t in g:
                if isinstance(t, SSEpsilon):
                    return True
        return False

    def toNFA(self):
        new = fa.NFA()
        new.States = copy.copy(self.States)
        new.setSigma(copy.copy(self.Sigma))
        new.setInitial(copy.copy(self.Initial))
        new.setFinal(copy.copy(self.Final))
        for s in self.delta:
            for t in self.delta[s]:
                for b in t.behaviour(self.Sigma):
                    for d in self.delta[s][t]:
                        new.addTransition(s, b, d)
        return new

    def emptyP(self):
        return self.witness() is None


def NFA2SSFA(aut):
    """ Transforms a NFA to and SSFA

    :param fa.NFA aut: NFA
    :rtype: SSFA"""
    new = SSFA(aut.Sigma)
    new.States = copy.copy(aut.States)
    new.Initial = copy.copy(aut.Initial)
    new.Final = copy.copy(aut.Final)
    for s1 in aut.delta:
        for t in aut.delta[s1]:
            if t != Epsilon:
                t1 = SSOneOf({t})
            else:
                t1 = SSEpsilon()
            for s2 in aut.delta[s1][t]:
                new.addTransition(s1, t1, s2)
    return new
