# -*- coding: utf-8 -*-
"""**Finite automata manipulation.**

Deterministic and non-deterministic automata manipulation, conversion and evaluation.

.. *Authors:* Rogério Reis & Nelma Moreira

.. *This is part of FAdo project*   https://fado.dcc.fc.up.pt.

.. *Copyright:* 1999-2022 Rogério Reis & Nelma Moreira {rogerio.reis,nelma.moreira} @ fc.up.pt

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
#  Copyright (c) 2022-2023. Rogério Reis <rogerio.reis@fc.up.pt> and Nelma Moreira <nelma.moreira@fc.up.pt>.
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
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

from copy import copy
from functools import cmp_to_key
from collections import deque
import deprecation
from .common import *
from .ssemigroup import SSemiGroup
from .unionFind import UnionFind
from . import graphs


class SemiDFA(Drawable):
    # noinspection PyUnresolvedReferences
    """Class of automata without initial or final states

    :ivar list States: set of states.
    :ivar set sigma: alphabet set.
    :ivar dict delta: the transition function."""

    def __init__(self):
        self.States = []
        self.delta = {}
        self.Sigma = set()

    def dotDrawState(self, sti: int, sep="\n") -> str:
        """Dot representation of a state

        Args:
            sti (int): state index.
            sep (:obj:`str`, optional): separator.

        Returns:
            str: line to add to the dot file."""
        return "node [shape = circle]; \"{0:s}\";{1:s}".format(graphvizTranslate(self.dotLabel(self.States[sti])), sep)

    @staticmethod
    def dotDrawTransition(st1: str, lbl1: str, st2, sep="\n") -> str:
        """Draw a transition in dot format

        Args:
            st1 (str): departing state.
            lbl1 (str): label.
            st2 (str): arriving state.
            sep (:obj:`str`, optional): separator.

        Returns:
            str: line to add to the dot file."""
        return "\"{0:s}\" -> \"{1:s}\" [label = \"{2:s}\"];{3:s} ".format(st1, st2, lbl1, sep)

    def dotFormat(self, size="20,20", filename=None, direction="LR", strict=False, maxlblsz=6, sep="\n") -> str:
        """ A dot representation

        Args:
            direction (str): direction of drawing - "LR" or "RL"
            size (str): size of image
            filename (str): Name of the output file
            sep (str): line separator
            maxlblsz (int): max size of labels before getting removed
            strict (bool): use limitations of label sizes
        Returns:
            str: the dot representation

        .. versionadded:: 0.9.6

        .. versionchanged:: 1.2.1"""
        s = "digraph finite_state_machine {{{0:s}".format(sep)
        s += "rankdir={0:s};{1:s}".format(direction, sep)
        s += "size=\"{0:s}\";{1:s}".format(size, sep)
        for si in range(len(self.States)):
            sn = self.dotLabel(self.States[si])
            s += "node [shape = point]; \"dummy{0:s}\"{1:s}".format(sn, sep)
            s += self.dotDrawState(si)
            s += "\"dummy{0:s}\" -> \"{1:s}\";{2:s}".format(sn, graphvizTranslate(sn), sep)
        for si in range(len(self.States)):
            for s1 in self.Sigma:
                s += self.dotDrawTransition(self.dotLabel(self.States[si]), str(s1),
                                            self.dotLabel(self.States[self.delta[si][s1]]))
        s += "}}{0:s}".format(sep)
        return s


# noinspection PyUnresolvedReferences
class FA(Drawable):
    """Base class for Finite Automata.
        This is just an abstract class.
        **Not to be used directly!!**

    :ivar list States: set of states.
    :ivar set sigma: alphabet set.
    :ivar int Initial: the initial state index.
    :ivar set Final: set of final states indexes.
    :ivar dict delta: the transition function.

    .. inheritance-diagram:: FA"""

    def __init__(self):
        self.States = []
        self.Sigma = set()
        self.Initial = None
        self.Final = set()
        self.delta = {}

    def __repr__(self):
        """'Official' string representation

        Returns:
            str:"""
        return 'FA({0:>s})'.format(self.__str__())

    def __str__(self):
        # noinspection PyProtectedMember
        a = self._s_States
        b = self._s_Sigma
        c = self._s_lstInitial()
        d = self._s_Final
        e = str(self._lstTransitions())
        return str((a, b, c, d, e))

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def toNFA(self):
        pass

    @abstractmethod
    def __ne__(self, other):
        pass

    @abstractmethod
    def evalSymbol(self, stil, sym):
        """Evaluation of a single symbol"""
        pass

    @abstractmethod
    def _s_lstInitial(self):
        pass

    @abstractmethod
    def _lstTransitions(self):
        pass

    @property
    def _s_States(self):
        return [str(x) for x in self.States]

    @property
    def _s_Sigma(self):
        return [str(x) for x in self.Sigma]

    @property
    def _s_Final(self):
        return [str(self.States[x]) for x in self.Final]

    @abstractmethod
    def transitions(self):
        pass

    @abstractmethod
    def transitionsA(self):
        pass

    def __len__(self):
        """Size: number of states

        :rtype: int"""
        return len(self.States)

    def stateAlphabet(self, sti: int) -> list:
        """Active alphabet for this state

        Args:
            sti (int): state
        Returns:
            list:"""
        if sti not in self.delta:
            return []
        else:
            return list(self.delta[sti].keys())

    def images(self, sti, c):
        """The set of images of a state by a symbol

        Args:
            sti (int): state
            c (object): symbol
        Returns:
            iterable:"""
        if sti not in self.delta or c not in self.delta[sti]:
            return []
        else:
            return self.delta[sti][c]

    def dotDrawState(self, sti, sep="\n", _strict=False, _maxlblsz=6):
        """ Draw a state in dot format

        Args:
            sti (int): index of the state.
            sep (:obj:`str`, optional): separator.
            _maxlblsz (:obj:`int`, optional): max size of labels before getting removed
            _strict (:obj:`bool`, optional): use limitations of label size
        Returns:
            str: string to be added to the dot file."""
        if sti in self.Final:
            return "node [shape = doublecircle]; \"{0:s}\";".format(graphvizTranslate(self.dotLabel(self.States[sti])),
                                                                    sep)
        else:
            return "node [shape = circle]; \"{0:s}\";{1:s}".format(graphvizTranslate(self.dotLabel(self.States[sti])),
                                                                   sep)

    def same_nullability(self, s1: int, s2: int) -> bool:
        """Tests if this two states have the same nullability

        Args:
            s1 (int): state index.
            s2 (int): state index.
        Returns:
            bool: have the states the same nullability?"""
        return (s1 in self.Final) is (s2 in self.Final)

    @abstractmethod
    def succintTransitions(self):
        """Collapsed transitions"""
        pass

    @staticmethod
    def dotDrawTransition(st1, label, st2, sep="\n"):
        """Draw a transition in dot format

        Args:
            st1 (str): departing state
            label (str): label
            st2 (str): arriving state
            sep (str): separator
        Returns:
            str:"""
        pass

    def initialSet(self):
        """The set of initial states

        Returns:
            set: set of States."""
        return self.Initial

    def initialP(self, state: int) -> bool:
        """ Tests if a state is initial

        Args:
            state: state index
        Returns:
            bool: is the state initial?"""
        return state in self.Initial

    def finalP(self, state: int) -> bool:
        """ Tests if a state is final

        Args:
            state (int): state index.
        Returns:
            bool: is the state final?"""
        return state in self.Final

    def finalsP(self, states: set) -> bool:
        """ Tests if al the states in a set are final

        Args:
            states (set): set of state indexes.
        Returns:
            bool: are all the states final?

        .. versionadded:: 1.0"""
        return states.issubset(self.Final)

    def _namesToString(self):
        """All state names are transformed in strings"""
        n = []
        for s in self.States:
            n.append(str(s))
        self.States = n
        return self

    def hasStateIndexP(self, st: int) -> bool:
        """Checks if a state index pertains to an FA

        Args:
            st (int): index of the state.
        Returns:
            bool:"""
        if st > (len(self.States) - 1):
            return False
        else:
            return True

    def addState(self, name=None) -> int:
        """Adds a new state to an FA. If no name is given a new name is created.

        Args:
            name (:obj:`Object`, optional): Name of the state to be added.
        Returns:
         int: Current number of states (the new state index).
        Raises:
            DuplicateName: if a state with that name already exists"""
        if name is None:
            iname = len(self.States)
            name = str(iname)
            while iname in self.States or name in self.States:
                iname += 1
                name = str(iname)
            self.States.append(name)
            return len(self.States) - 1
        elif name in self.States:
            raise DuplicateName(self.stateIndex(name))
        else:
            self.States.append(name)
            return len(self.States) - 1

    @abstractmethod
    def _deleteRefInDelta(self, j, sm, s):
        pass

    @abstractmethod
    def _deleteRefInitial(self, s):
        pass

    def stateIndexes(self):
        return list(range(len(self.States)))

    def deleteState(self, sti: int):
        """Remove the given state and the transitions related with that state.

        Args:
            sti (int): index of the state to be removed
        Raises:
            DFAstateUnknown: if state index does not exist"""
        if sti >= len(self.States):
            raise DFAstateUnknown(sti)
        else:
            if sti in list(self.delta.keys()):
                del self.delta[sti]
            for j in list(self.delta.keys()):
                for sym in list(self.delta[j].keys()):
                    self._deleteRefInDelta(j, sym, sti)
            if sti in self.Final:
                self.Final.remove(sti)
            self._deleteRefInitial(sti)
            to_add = set()
            to_del = set()
            for s in self.Final:
                if sti < s:
                    to_del.add(s)
                    to_add.add(s - 1)
            self.Final = self.Final - to_del | to_add
            for j in range(sti + 1, len(self.States)):
                if j in self.delta:
                    self.delta[j - 1] = self.delta[j]
                    del self.delta[j]
            del self.States[sti]

    def words(self, stringo=True):
        """Lexicographical word generator

        Args:
            stringo (:obj:`bool`, optional): are words strings?
                Default is True.
        Yields:
            Word: the next word generated.

        .. attention:: Does not generate the empty word

        .. versionadded:: 0.9.8"""

        def _translate(l, r, s1=True):
            if s1:
                s = ""
                for z in l:
                    s += r[z]
                return s
            else:
                return [r[y] for y in l]

        import itertools

        ss = list(self.Sigma)
        ss.sort(key=lambda x1: x1.__repr__())
        n = len(ss)
        n0 = 1
        while True:
            for x in itertools.product(list(range(n)), repeat=n0):
                yield Word(_translate(x, ss, stringo))
            n0 += 1

    def equivalentP(self, other):
        """Test equivalence between automata

        Args:
         other (FA): the other automata
        Returns:
         bool:

        .. versionadded:: 0.9.6"""
        return self == other

    def setInitial(self, stateindex):
        """Sets the initial state of a FA

        Args:
            stateindex (int): index of the initial state."""
        self.Initial = stateindex

    def setFinal(self, statelist):
        """Sets the final states of the FA

        Args:
            statelist (int|list|set): a list (or set) of final states indexes.

        .. caution::
           Erases any previous definition of the final state set."""
        self.Final = set(statelist)

    def addFinal(self, stateindex):
        """A new state is added to the already defined set of final states.

        Args:
            stateindex (int): index of the new final state."""
        self.Final.add(stateindex)

    def delFinals(self):
        """Deletes all the information about final states."""
        self.Final = set([])

    def delFinal(self, st):
        """Deletes a state from the final states list

        Args:
            st (int): state to be marked as not final."""
        self.Final -= {st}

    def setSigma(self, symbol_set):
        """Defines the alphabet for the FA.

        Args:
            symbol_set (list|set): alphabet symbols"""
        self.Sigma = set(symbol_set)

    def addSigma(self, sym):
        """Adds a new symbol to the alphabet.

        Args:
            sym (str): symbol to be added
        Raises:
            DFAepsilonRedefinition: if sym is Epsilon

        .. note::
            * There is no problem with duplicate symbols because sigma is a Set.
            * No symbol Epsilon can be added."""
        if sym == Epsilon:
            raise DFAepsilonRedefinition()
        self.Sigma.add(sym)

    def stateIndex(self, name, auto_create=False):
        """Index of given state name.

        Args:
            name (object): name of the state.
            auto_create (:obj:`bool`, optional): flag to create state if not already done.
        Returns:
            int: state index
        Raises:
            DFAstateUnknown: if the state name is unknown and autoCreate==False

        .. note::
           Replaces stateName

        .. note::
           If the state name is not known and flag is set creates it on the fly

        .. versionadded:: 1.0"""
        if name in self.States:
            return self.States.index(name)
        else:
            if auto_create:
                return self.addState(name)
            else:
                raise DFAstateUnknown(name)

    @deprecation.deprecated(deprecated_in="1.0",
                            current_version=FAdoVersion,
                            details="Use the stateIndex() function instead")
    def stateName(self, name, auto_create=False):
        """Index of given state name.

        Args:
            name (object): name of the state
            auto_create (:obj:`bool`, optional): flag to create state if not already done
        Returns:
            int: state index
        Raises:
            DFAstateUnknown: if the state name is unknown and autoCreate==False

        .. deprecated:: 1.0
           Use: :func:`stateIndex` instead"""
        return self.stateIndex(name, auto_create)

    def indexList(self, lstn):
        """Converts a list of stateNames into a set of stateIndexes.

        Args:
            lstn (list): list of names
        Returns:
            set: the list of state indexes
        Raises:
            DFAstateUnknown: if a state name is unknown"""
        lst = set()
        for s in lstn:
            lst.add(self.stateIndex(s))
        return lst

    @abstractmethod
    def star(self, _):
        pass

    @abstractmethod
    def __or__(self, _):
        pass

    @abstractmethod
    def __and__(self, _):
        pass

    def plus(self):
        """Plus of a FA (star without the adding of epsilon)

        .. versionadded:: 0.9.6"""
        return self.star(True)

    def disjunction(self, other):
        """A simple literate invocation of __or__

        Args:
            other (FA): the other FA
        Returns:
            FA: Union of self and other."""
        return self.__or__(other)

    def disj(self, other):
        """Another simple literate invocation of __or__

        Args:
             other (FA): the other FA.
        Returns:
            FA: Union of self and other.

        .. versionadded:: 0.9.6"""
        return self.__or__(other)

    def union(self, other):
        """A simple literate invocation of __or__

        Args:
             other (FA): right-hand operand.
        Returns:
            FA: Union of self and other."""
        return self.__or__(other)

    def conjunction(self, other):
        """A simple literate invocation of __and__

        Args:
             other (FA): right-hand operand.
        Returns:
            FA: Intersection of self and other.

        .. versionadded:: 0.9.6"""
        return self.__and__(other)

    def renameState(self, st, name):
        """Rename a given state.

        Args:
            st (int): state index.
            name (object): name.
        Returns:
            FA: self.

        .. note::
            Deals gracefully both with int and str names in the case of name collision.

        .. attention::
           the object is modified in place"""
        if name != self.States[st]:
            if name in self.States:
                if isinstance(name, int):
                    while name in self.States:
                        name += name + 1
                elif isinstance(name, str):
                    while name in self.States:
                        name += "+"
                else:
                    raise DuplicateName
            self.States[st] = name
        return self

    def renameStates(self, name_list=None):
        """Renames all states using a new list of names.

        Args:
            name_list (list): list of new names.
        Returns:
            FA: self.
        Raises:
            DFAerror: if provided list is too short.

        .. note::
           If no list of names is given, state indexes are used.

        .. attention::
           the object is modified in place"""
        if name_list is None:
            self.States = list(range(len(self.States)))
        else:
            if len(name_list) < len(self.States):
                raise DFAerror
            else:
                for i in range(len(self.States)):
                    self.renameState(i, name_list[i])
        return self

    def eliminateDeadName(self):
        """Eliminates dead state name (common.DeadName) renaming the state

        Returns:
            DFA: self
        Attention:
           works inplace

        .. versionadded:: 1.2"""
        try:
            i = self.stateIndex(DeadName)
        except DFAstateUnknown:
            return self
        self.renameState(i, str(len(self.States)))
        return self

    def noBlankNames(self):
        """Eliminates blank names

        Returns:
            FA: self

        .. attention::
           in place transformation"""
        for i in range(len(self.States)):
            if self.States[i] == "":
                self.States[i] = str(i)
        return self

    @abstractmethod
    def reverseTransitions(self, _):
        pass

    def reversal(self):
        """Returns a NFA that recognizes the reversal of the language

        Returns:
            NFA: NFA recognizing reversal language
        """
        rev = NFA()
        rev.setSigma(self.Sigma)
        rev.States = list(self.States)
        self.reverseTransitions(rev)
        rev.setFinal([self.Initial])
        rev.setInitial(self.Final)
        return rev

    def countTransitions(self):
        """Evaluates the size of FA transitionwise

        Returns:
            int: the number of transitions

        .. versionchanged:: 1.0"""
        return sum([len(self.delta[i]) for i in self.delta])

    def inputS(self, i):
        """Input labels coming out of state i

        Args:
            i (int): state
        Returns:
            set of str: set of input labels

        .. versionadded:: 1.0"""
        return set(self.delta.get(i, {}))

    @abstractmethod
    def dup(self):
        pass

    def dotFormat(self, size="20,20", filename=None, direction="LR", strict=False, maxlblsz=6, sep="\n") -> str:
        """ A dot representation

        Args:
            direction (str): direction of drawing - "LR" or "RL"
            size (str): size of image
            filename (str): output file name
            sep (str): line separator
            maxlblsz (int): max size of labels before getting removed
            strict (bool): use limitations of label sizes
        Returns:
            str: the dot representation

        .. versionadded:: 0.9.6

        .. versionchanged:: 1.2.1"""
        if not strict and max([len(str(name)) for name in self.States]) > maxlblsz:
            o = self.dup()
            o.renameStates()
        else:
            o = self
        s = "digraph finite_state_machine {{{0:s}".format(sep)
        s += "rankdir={0:s};{1:s}".format(direction, sep)
        s += "size=\"{0:s}\";{1:s}".format(size, sep)
        s += "node [shape = point]; dummy{0:s}".format(sep)
        ni_states = [i for i in range(len(o.States)) if i != o.Initial]
        s += o.dotDrawState(o.Initial)
        s += "dummy -> \"{0:s}\"{1:s}".format(graphvizTranslate(o.dotLabel(o.States[o.Initial])), sep)
        for sti in ni_states:
            s += o.dotDrawState(sti)
        for si in o.succintTransitions():
            s += o.dotDrawTransition(si[0], si[1], si[2], sep)
        s += "}}{0:s}".format(sep)
        return s


class OFA(FA):
    """ Base class for one-way automata

    :ivar list States: set of states.
    :ivar set sigma: alphabet set.
    :ivar int Initial: the initial state index.
    :ivar set Final: set of final states indexes.
    :ivar dict delta: the transition function.

    .. inheritance-diagram:: OFA"""

    @abstractmethod
    def succintTransitions(self):
        """Collapsed transitions"""
        pass

    @abstractmethod
    def evalSymbol(self, stil, sym):
        """Eval symbol"""
        pass

    @abstractmethod
    def addTransition(self, st1, sym, st2):
        """Add transition

        Args:
            st1 (int): departing state
            sym (str): label
            st2 (int): arriving state"""
        pass

    @abstractmethod
    def __ne__(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def stateChildren(self, _state, _strict=None):
        """ To be implemented below

        Args:
            _state (state):
            _strict (int): state id queried
        Returns:
            list:"""
        pass

    def _deleteRefTo(self, src, sym, dest):
        """Delete transition

        Args:
            src (int): source state
            sym (str): label
            dest (int): target state"""
        if self.delta.get(src, {}).get(sym, None) == dest:
            del (self.delta[src][sym])

    def dotDrawTransition(self, st1, label, st2, sep="\n"):
        """Draw a transition in dot format

        Args:
            st1 (str): departing state
            label (str): symbol
            st2 (str): arriving state
            sep (str): separator
        Returns:
            str:"""
        return "\"{0:s}\" -> \"{1:s}\" [label = \"{2:s}\"];{3:s} ".format(graphvizTranslate(st1),
                                                                          graphvizTranslate(st2),
                                                                          label, sep)

    @abstractmethod
    def initialComp(self):
        pass

    @abstractmethod
    def _getTags(self):
        pass

    @abstractmethod
    def usefulStates(self):
        pass

    @abstractmethod
    def deleteStates(self, del_states):
        pass

    @abstractmethod
    def finalCompP(self, s):
        pass

    @abstractmethod
    def uniqueRepr(self):
        pass

    def emptyP(self):
        """ Tests if the automaton accepts an empty language

        Returns:
            bool:

        .. versionadded:: 1.0"""
        a = self.initialComp()
        for s in a:
            if self.finalP(s):
                return False
        return True

    def dump(self):
        """Returns a python representation of the object

        Returns:
            tuple:the python representation (Tags,States,sigma,delta,Initial,Final)"""
        tags = self._getTags()
        sig = list(self.Sigma)
        initial = [i for i in forceIterable(self.Initial)]
        final = [i for i in self.Final]
        dt = []
        for i in range(self.__len__()):
            for c in self.delta.get(i, []):
                if c == Epsilon:
                    ci = -1
                else:
                    ci = sig.index(c)
                for j in forceIterable(self.delta[i][c]):
                    dt.append((i, ci, j))
        return tags, self.States, sig, dt, initial, final

    def trim(self):
        """Removes the states that do not lead to a final state, or, inclusively,
            that can't be reached from the initial state. Only useful states
            remain.

        Returns:
            FA:

        .. attention:
            only applies to non-empty languages

        .. attention::
           in place transformation"""
        useful = self.usefulStates()
        del_states = [s for s in range(len(self.States)) if s not in useful]
        if del_states:
            self.deleteStates(del_states)
        return self

    def trimP(self):
        """Tests if the FA is trim: initially connected and co-accessible

        Returns:
            bool:"""
        for s, _ in enumerate(self.States):
            if not self.finalCompP(s):
                return False
        return len(self.States) == len(self.initialComp())

    def minimalBrzozowski(self):
        """Constructs the equivalent minimal DFA using Brzozowski's algorithm

        Returns:
            DFA: equivalent minimal DFA"""
        return self.reversal().toDFA().reversal().toDFA()

    def minimalBrzozowskiP(self):
        """Tests if the FA is minimal using Brzozowski's algorithm

        Returns:
            bool:"""
        x = self.minimalBrzozowski()
        x.complete()
        return self.uniqueRepr() == x.uniqueRepr()

    def _isAcyclic(self, s, visited, strict):
        """Determines if from state s a cycle is reached

        Args:
            s (int): state
            visited (dict): marks visited states
            strict (bool): if not True loops are allowed
        Returns:
            bool:"""
        if s not in visited:
            visited[s] = 1
            if s in self.delta:
                for dest in self.stateChildren(s, strict):
                    acyclic = self._isAcyclic(dest, visited, strict)
                    if not acyclic:
                        return False
                visited[s] = 2
            else:
                visited[s] = 2
                return True
        else:
            if visited[s] == 1:
                return False
        return True

    def acyclicP(self, strict=True):
        """ Checks if the FA is acyclic

        Args:
            strict (bool): if not True loops are allowed
        Returns: True if the FA is acyclic
            bool: True if the FA is acyclic"""
        visited = {}
        for s in range(len(self.States)):
            acyclic = self._isAcyclic(s, visited, strict)
            if not acyclic:
                return False
        return True

    def _topoSort(self, s, visited, lst):
        """Auxiliar for topological order"""
        if s not in visited:
            visited.append(s)
            if s in self.delta:
                # noinspection PyTypeChecker
                for dest in self.stateChildren(s, True):
                    self._topoSort(dest, visited, lst)
            lst.insert(0, s)

    def topoSort(self):
        """Topological order for the FA

        Returns:
            list: List of state indexes

        .. note::
           self loops are taken in consideration"""
        visited = []
        lst = []
        for s in range(len(self.States)):
            self._topoSort(s, visited, lst)
        return lst

    def eliminateStout(self, st):
        """Eliminate all transitions outgoing from a given state

        Args:
            st (int): the state index to lose all outgoing transitions

        .. attention::
           performs in place alteration of the automata

        .. versionadded:: 0.9.6"""
        if st in list(self.delta.keys()):
            del (self.delta[st])
        return self

    def dup(self):
        """ Duplicate OFA

        Returns:
            OFA: duplicate object"""
        return deepcopy(self)


class NFA(OFA):
    """Class for Non-deterministic Finite Automata (epsilon-transitions allowed).

    :ivar list States: set of states.
    :ivar set sigma: alphabet set.
    :ivar set Initial: initial state indexes.
    :ivar set Final: set of final states indexes.
    :ivar dict delta: the transition function.

    .. inheritance-diagram:: NFA"""

    def uniqueRepr(self):
        """Dummy representation. Used DFA.uniqueRepr()

        Returns:
            tuple:"""
        return self.toDFA().uniqueRepr()

    def __init__(self):
        FA.__init__(self)
        self.Initial = set()
        self.epsilon_transitions = None

    def __repr__(self):
        return "NFA({0:>s})".format(self.__str__())

    def _lstTransitions(self):
        l = []
        for x in self.delta:
            for k in self.delta[x]:
                for y in self.delta[x][k]:
                    l.append((self.States[x], k, self.States[y]))
        return l

    def _s_lstInitial(self):
        return [str(self.States[x]) for x in self.Initial]

    def _lstInitial(self):
        if self.Initial is None:
            raise DFAnoInitial()
        else:
            return [self.States[i] for i in self.Initial]

    @staticmethod
    def _vDescription():
        """Generation of Verso interface description

        Returns:
            list:

        .. versionadded:: 0.9.5"""
        return [("NFA", "Nondeterministic Finite Automata"),
                [("NFAFAdo", lambda x: saveToString(x), "FAdo"),
                 ("NFAdot", lambda x: x.dotFormat("&"), "dot")],
                ("NFA-to-DFA", ("NFA to DFA", "NFA to DFA"), 1, "NFA", "DFA", lambda *x: x[0].toDFA()),
                ("NFA-reversal", ("Reversal language NFA", "Reversal language NFA"), 1, "NFA", "NFA",
                 lambda *x: x[0].reversal())]

    def __or__(self, other):
        """ Disjunction of automata:  X | Y.

        Args:
         other (FA) : the right-hand operand
        Raises:
             FAdoGeneralError: if any operand is not an NFA

        .. versionchanged:: 1.2"""
        if isinstance(other, DFA):
            par2 = other.toNFA()
        elif not isinstance(other, NFA):
            raise FAdoGeneralError("Incompatible objects")
        else:
            par2 = other
        new = self._copySkell(par2)
        ini = new.addState()
        new.Sigma = new.Sigma.union(other.Sigma)
        new.addInitial(ini)
        for s in self.Initial:
            si = new.stateIndex((0, s))
            new.addTransition(ini, Epsilon, si)
        for s in par2.Initial:
            si = new.stateIndex((1, s))
            new.addTransition(ini, Epsilon, si)
        fin = new.addState()
        new.addFinal(fin)
        for s in self.Final:
            si = new.stateIndex((0, s))
            new.addTransition(si, Epsilon, fin)
        for s in par2.Final:
            si = new.stateIndex((1, s))
            new.addTransition(si, Epsilon, fin)
        return new

    def __and__(self, other):
        """Conjunction of automata

        Args:
            other (FA): the right-hand operand
        Returns:
            NFA:
        Raises:
             FAdoGeneralError: if any operand is not an NFA"""
        if isinstance(other, DFA):
            par2 = other.toNFA()
        elif not isinstance(other, NFA):
            raise FAdoGeneralError("Incompatible objects")
        else:
            par2 = other
        new = self.product(par2)
        for x in [(self.States[a], par2.States[b]) for a in self.Final for b in other.Final]:
            if x in new.States:
                new.addFinal(new.stateIndex(x))
        return new._namesToString()

    def __invert__(self):
        """Complement of the NFA (through conversion to DFA)

        Returns:
            NFA:"""
        foo = self.toDFA()
        return foo.__invert__().toNFA()

    def _getTags(self):
        """returns Tags for dump

        Returns:
            list:"""
        return ["NFA"]

    def concat(self, other, middle="middle"):
        """Concatenation of NFA

        Args:
            middle (str): glue state name
            other (FA): the other NFA
        Returns:
            NFA: the result of the concatenation"""
        if isinstance(other, DFA):
            par2 = other.toNFA()
        else:
            par2 = other
        new = self._copySkell(par2)
        for i in self.Initial:
            new.addInitial(new.stateIndex((0, i)))
        m = new.addState(middle)
        for i in self.Final:
            new.addTransition(new.stateIndex((0, i)), Epsilon, m)
        for i in par2.Initial:
            new.addTransition(m, Epsilon, new.stateIndex((1, i)))
        for i in par2.Final:
            new.addFinal(new.stateIndex((1, i)))
        return new

    def computeFollowNames(self):
        """ Computes the follow set to use in names

        Returns:
             list:"""
        l = []
        for i in range(len(self.States)):
            l1 = []
            for c in self.delta.get(i, []):
                for j in self.delta[i][c]:
                    k = self.States[j]
                    if k not in l1:
                        l1.append(k)
            l.append((sorted(l1), i in self.Final))
        return l

    def renameStatesFromPosition(self):
        """ Rename states of a Glushkov automaton using the positions of the marked RE

        Returns:
            NFA:"""
        new = self.dup()
        l = []
        for i in new.States:
            if i == "Initial":
                l.append(0)
            else:
                (_, j) = eval(i)
                l.append(j)
        new.States = l
        return new

    def followFromPosition(self):
        """ computes follow automaton from a Position automaton

        Returns:
            NFA:"""
        fl = self.renameStatesFromPosition().computeFollowNames()
        new = NFA()
        fu = unique(fl)
        for x in fu:
            new.addState(x)
        for i in self.Initial:
            new.addInitial(fu.index(fl[i]))
        for i in self.delta:
            for c in self.delta.get(i, []):
                for j in self.delta[i][c]:
                    new.addTransition(fu.index(fl[i]), c, fu.index(fl[j]))
        for i in self.Final:
            new.addFinal(fu.index(fl[i]))
        return new

    def detSet(self, generic=False):
        """ Computes the determination uppon a followFromPosition result

        Returns:
            NFA:"""
        # TODO: write better code for this method
        if generic:
            n = len(self.States)
            self.States = [i for i in range(n)]
            nn = self.computeFollowNames()
            self.States = nn
        l = [set(i) for (i, _) in self.States]
        l1 = []
        for i in l:
            if i == set():
                l1.append({'@@'})
            else:
                l1.append(i)
        l = l1
        new = DFA()
        assert len(self.Initial) == 1
        for i in self.Initial:
            if i in self.Final:
                shit = set(self.States[i][0])
                shit.add("@")
                foo = new.addState(shit)
            else:
                foo = new.addState(set(self.States[i][0]))
            new.setInitial(foo)
            if i in self.Final:
                new.addFinal(foo)
        done, todo = set(), {0}
        while todo:
            s = todo.pop()
            ls = [(i, ii) for (i, ii) in enumerate(l) if new.States[s].issuperset(ii)]
            for c in self.Sigma:
                ss, sf = set(), False
                f = False
                for i, ii in ls:
                    for j in self.delta.get(i, {}).get(c, set()):
                        ss, sf = ss.union(l[j]), True
                        if j in self.Final:
                            f = True
                ss.discard("@@")
                if sf:
                    if f:
                        ss.add("@")
                    if ss not in new.States:
                        n = new.addState(ss)
                        todo.add(n)
                    else:
                        n = new.stateIndex(ss)
                    new.addTransition(s, c, n)
                    if f:
                        new.addFinal(n)
        return new

    def witness(self):
        """Witness of non emptiness

        Returns:
            str: word"""
        done = set()
        not_done = set()
        pref = dict()
        for si in self.Initial:
            pref[si] = Epsilon
            not_done.add(si)
        while not_done:
            si = not_done.pop()
            done.add(si)
            if si in self.Final:
                return pref[si]
            for syi in self.delta.get(si, []):
                for so in self.delta[si][syi]:
                    if so in done or so in not_done:
                        continue
                    pref[so] = sConcat(pref[si], syi)
                    not_done.add(so)
        return None

    # noinspection PyUnresolvedReferences
    def shuffle(self, other):
        """Shuffle of a NFA

        Args:
            other (FA): an FA
        Returns:
            NFA: the resulting NFA"""
        if len(self.Initial) > 1:
            d1 = self._toNFASingleInitial().elimEpsilon()
        else:
            d1 = self
        if type(other) == NFA:
            if len(other.Initial) > 1:
                d2 = self._toNFASingleInitial().elimEpsilon()
            else:
                d2 = other
        else:
            d2 = other.toNFA()
        c = NFA()
        n_sigma = d1.Sigma.union(d2.Sigma)
        c.setSigma(n_sigma)
        c.States = [(i, j) for i in range(len(d1.States)) for j in range(len(d2.States))]
        c.addInitial(c.stateIndex((list(d1.Initial)[0], list(d2.Initial)[0])))
        for st in c.States:
            si = c.stateIndex(st)
            if d1.finalP(st[0]) and d2.finalP(st[1]):
                c.addFinal(si)
            for sym in c.Sigma:
                try:
                    lq = d1.evalSymbol([st[0]], sym)
                    for q in lq:
                        c.addTransition(si, sym, c.stateIndex((q, st[1])))
                except (DFAstopped, DFAsymbolUnknown):
                    pass
                try:
                    lq = d2.evalSymbol([st[1]], sym)
                    for q in lq:
                        c.addTransition(si, sym, c.stateIndex((st[0], q)))
                except (DFAstopped, DFAsymbolUnknown):
                    pass
        return c

    def star(self, flag=False):
        """Kleene star of a NFA

        Args:
            flag (bool): plus instead of star?
        Returns:
            NFA: the resulting NFA"""
        new = self.dup()
        ini = copy(new.Initial)
        fin = copy(new.Final)
        nf = new.addState()
        new.addFinal(nf)
        if not flag:
            ni = new.addState()
            new.setInitial([ni])
            new.addTransition(ni, Epsilon, nf)
        else:
            ni = new.addState()
            nni = new.addState()
            new.setInitial([nni])
            new.addTransition(nni, Epsilon, ni)
        new.addTransition(nf, Epsilon, ni)
        for i in ini:
            new.addTransition(ni, Epsilon, i)
        for i in fin:
            new.addTransition(i, Epsilon, nf)
        return new

    def __eq__(self, other):
        return self.toDFA() == other.toDFA()

    def __ne__(self, other):
        return not self == other

    def _copySkell(self, other):
        """Creates a new NFA with the skells of both NFAs

        Each state is named with its previous index is inscribed in a tuple (0,_) and (1,_) respectively

        :param NFA other: the other NFA
        :rtype: NFA

        .. attention::
           No initial and final states are assigned in the resulting NFA."""
        new = NFA()
        s = len(self.States)
        for i in range(s):
            new.addState((0, i))
        for i in self.delta:
            for c in self.delta[i]:
                for j in self.delta[i][c]:
                    new.addTransition(new.stateIndex((0, i)), c, new.stateIndex((0, j)))
        s = len(other.States)
        for i in range(s):
            new.addState((1, i))
        for i in other.delta:
            for c in other.delta[i]:
                for j in other.delta[i][c]:
                    new.addTransition(new.stateIndex((1, i)), c, new.stateIndex((1, j)))
        return new

    def setInitial(self, statelist):
        """Sets the initial states of an NFA

        Args:
            statelist (set|list|int): an iterable of initial state indexes"""
        self.Initial = set(statelist)

    def addInitial(self, stateindex):
        """Add a new state to the set of initial states.

        Args:
            stateindex (int): index of new initial state"""
        self.Initial.add(stateindex)

    def succintTransitions(self):
        """ Collects the transition information in a compact way suitable for graphical representation.
        Returns:
            list:

        .. note:
            tupples in the list are stateout, label, statein
        """
        foo = dict()
        for s in self.delta:
            for c in self.delta[s]:
                for s1 in self.delta[s][c]:
                    k = (s, s1)
                    if k not in foo:
                        foo[k] = []
                    foo[k].append(c)
        l = []
        for k in foo:
            cs = foo[k]
            s = "%s" % graphvizTranslate(str(cs[0]))
            for c in cs[1:]:
                s += ", %s" % graphvizTranslate(str(c))
            l.append((self.dotLabel(self.States[k[0]]), s, self.dotLabel(self.States[k[1]])))
        return l

    def deleteStates(self, del_states):
        """Delete given iterable collection of states from the automaton.

        Args:
            del_states (set|list): collection of int representing states

        .. note::
           delta function will always be rebuilt, regardless of whether the states list to remove is a suffix,
           or a sublist, of the automaton's states list."""
        rename_map = {}
        new_delta = {}
        new_final = set()
        new_states = []
        for state in del_states:
            if state in self.Initial:
                self.Initial.remove(state)
        for state in range(len(self.States)):
            if state not in del_states:
                rename_map[state] = len(new_states)
                new_states.append(self.States[state])
        for state in rename_map:
            if state in self.Final:
                new_final.add(rename_map[state])
            if state not in self.delta:
                continue
            if not len(self.delta[state]) == 0:
                new_delta[rename_map[state]] = {}
            for symbol in self.delta[state]:
                new_targets = set([rename_map[s] for s in self.delta[state][symbol]
                                   if s in rename_map])
                if new_targets:
                    new_delta[rename_map[state]][symbol] = new_targets
        self.States = new_states
        self.delta = new_delta
        self.Final = new_final
        self.Initial = set([rename_map.get(x, x) for x in self.Initial])

    def addTransition(self, sti1, sym, sti2):
        """Adds a new transition. Transition is from ``sti1`` to ``sti2`` consuming symbol ``sym``. ``sti2`` is a
        unique state, not a set of them.

        Args:
            sti1 (int): state index of departure
            sti2 (int): state index of arrival
            sym (str): symbol consumed"""
        if sym != Epsilon:
            self.Sigma.add(sym)
        if sti1 not in self.delta:
            self.delta[sti1] = {sym: {sti2}}
        elif sym not in self.delta[sti1]:
            self.delta[sti1][sym] = {sti2}
        else:
            self.delta[sti1][sym].add(sti2)

    def addTransitionStar(self, sti1, sti2, exception=[]):
        """Adds a new transition from sti1 to sti2 consuming any symbol

        Args:
            sti1 (int): state index of departure
            sti2 (int): state index of arrival
            exception (list): letters to excluded from the pattern

        .. versionadded:: 2.1"""
        for c in self.Sigma:
            if c not in exception:
                self.addTransition(sti1, c, sti2)

    def addEpsilonLoops(self):
        """Add epsilon loops to every state

        .. attention:: in-place modification

        .. versionadded:: 1.0"""
        for i in range(len(self.States)):
            self.addTransition(i, Epsilon, i)
        return self

    def addTransitionQ(self, srci, dest, symb, qfuture, qpast):
        """Add transition to the new transducer instance.

        Args:
            qpast (set): past queue
            qfuture (set): future queue
            symb: symbol
            dest(int): destination state
            srci (int): source state

        .. versionadded:: 1.0"""
        if dest not in qpast:
            qfuture.add(dest)
        i = self.stateIndex(dest, True)
        self.addTransition(srci, symb, i)

    def delTransition(self, sti1, sym, sti2, _no_check=False):
        """Remove a transition if existing and perform cleanup on the transition function's internal data structure.

        Args:
            sti1 (int): state index of departure
            sti2 (int): state index of arrival
            sym: symbol consumed
            _no_check (bool): dismiss secure code

        .. note::
           unused alphabet symbols will be discarded from sigma."""
        if not _no_check and (sti1 not in self.delta or sym not in self.delta[sti1]):
            return
        self.delta[sti1][sym].discard(sti2)
        if not self.delta[sti1][sym]:
            del self.delta[sti1][sym]
            if all([sym not in x for x in iter(self.delta.values())]):
                self.Sigma.discard(sym)
            if not self.delta[sti1]:
                del self.delta[sti1]

    def reversal(self):
        """Returns a NFA that recognizes the reversal of the language

        Returns:
            NFA: NFA recognizing reversal language"""
        rev = NFA()
        rev.setSigma(self.Sigma)
        rev.States = self.States[:]
        self.reverseTransitions(rev)
        rev.setFinal(self.Initial)
        rev.setInitial(self.Final)
        return rev

    def reorder(self, dicti):
        """Reorder states indexes according to given dictionary.

        Args:
            dicti (dict): state name reorder

        .. attention:: in-place modification

        .. note::
           dictionary does not have to be complete"""
        if len(list(dicti.keys())) != len(self.States):
            for i in range(len(self.States)):
                if i not in dicti:
                    dicti[i] = i
        delta = {}
        for s in self.delta:
            delta[dicti[s]] = {}
            for c in self.delta[s]:
                delta[dicti[s]][c] = set()
                for st in self.delta[s][c]:
                    delta[dicti[s]][c].add(dicti[st])
        self.delta = delta
        self.setInitial([dicti[x] for x in self.Initial])
        final = set()
        for i in self.Final:
            final.add(dicti[i])
        self.Final = final
        states = list(range(len(self.States)))
        for i in range(len(self.States)):
            states[dicti[i]] = self.States[i]
        self.States = states

    def universalP(self):
        """ Whether this NFA  is universal (accetps all words) """
        foo = self.toDFA()
        return foo.universalP()

    def epsilonP(self):
        """Whether this NFA has epsilon-transitions

        Returns:
            bool:"""
        return any([Epsilon in x for x in iter(self.delta.values())])

    def epsilonClosure(self, st):
        """Returns the set of states epsilon-connected to from given state or set of states.

        Args:
            st (int|set): state index or set of state indexes
        Returns:
            set: the list of state indexes epsilon connected to ``st``

        .. attention::
           ``st`` must exist beforehand."""
        if type(st) is set:
            s2 = set(st)
        else:
            s2 = {st}
        s1 = set()
        while s2:
            s = s2.pop()
            s1.add(s)
            s2.update(self.delta.get(s, {}).get(Epsilon, set()) - s1)
        return s1

    def closeEpsilon(self, st):
        """Add all non epsilon transitions from the states in the epsilon closure of given state to given state.

        Args:
            st (int): state index

        .. attention:: in-place modification"""
        targets = self.epsilonClosure(st)
        targets.remove(st)
        if not targets:
            return
        for target in targets:
            self.delTransition(st, Epsilon, target)
        not_final = st not in self.Final
        for target in targets:
            if target in self.delta:
                for symbol, states in list(self.delta[target].items()):
                    if symbol is Epsilon:
                        continue
                    for state in states:
                        self.addTransition(st, symbol, state)
            if not_final and target in self.Final:
                self.addFinal(st)
                not_final = False

    def eliminateTSymbol(self, symbol):
        """Delete all trasitions through a given symbol

        Args:
            symbol (str): the symbol to be excluded from delta

        .. attention:: in-place modification

        .. versionadded:: 0.9.6"""
        for s in self.delta:
            if symbol in self.delta[s]:
                del (self.delta[s][symbol])
            if not self.delta[s]:
                del self.delta[s]

    def elimEpsilon(self):
        """Eliminate epsilon-transitions from this automaton.

        :rtype : NFA

        .. attention::
           performs in place modification of automaton

        .. versionchanged:: 1.1.1"""
        for state in range(len(self.States)):
            self.closeEpsilon(state)
        self.epsilon_transitions = False
        return self

    def evalWordP(self, word):
        """Verify if the NFA recognises given word.

        Args:
            word (str): word to be recognised
        Returns:
            bool:"""
        if self.epsilon_transitions or self.epsilon_transitions is None:
            ilist = self.epsilonClosure(self.Initial)
            for c in word:
                ilist = self.evalSymbol(ilist, c)
                if not ilist:
                    return False
            for f in self.Final:
                if f in ilist:
                    return True
            return False
        else:
            foo = self.Initial
            for c in word:
                bar = set()
                for si in foo:
                    bar = bar.union(self.delta.get(si, {}).get(c, set()))
                foo = bar
            return len(self.Final.intersection(foo)) != 0

    def evalSymbol(self, stil, sym):
        """Set of states reacheable from given states through given symbol and epsilon closure.

        Args:
            stil (set|list): set of current states
            sym (str): symbol to be consumed
        Returns:
            set: set of reached state indexes
        Raises:
            DFAsymbolUnknown: if symbol is not in alphabet"""
        if sym not in self.Sigma:
            raise DFAsymbolUnknown(sym)
        res = set()
        for s in stil:
            try:
                ls = self.delta[s][sym]
            except KeyError:
                ls = set()
            except NameError:
                ls = set()
            for t in ls:
                res.update(self.epsilonClosure(t))
        return res

    def minimal(self):
        """Evaluates the equivalent minimal DFA

        Returns:
            DFA: equivalent minimal DFA"""
        return self.minimalDFA()

    def minimalDFA(self):
        """Evaluates the equivalent minimal complete DFA

        Returns:
            DFA: equivalent minimal DFA"""
        return self.minimalBrzozowski()

    def dup(self):
        """Duplicate the basic structure into a new NFA. Basically a copy.deep.

        Returns:
            NFA: """
        new = NFA()
        new.setSigma(self.Sigma)
        new.States = self.States[:]
        new.Initial = self.Initial.copy()
        new.Final = self.Final.copy()
        for s in self.delta:
            new.delta[s] = {}
            for c in self.delta[s]:
                new.delta[s][c] = self.delta[s][c].copy()
        return new

    def _inc(self, fa):
        """Combine self with given FA with a single final state.

        Args:
            fa (FA): FA to be included
        Returns:
            tupple: a pair of state indexes (initial and final of the resulting NFA)

        .. note::
           State names are not preserved."""
        for s in range(len(self.States)):
            self.States[s] = (0, self.States[s])
        for c in fa.Sigma:
            self.addSigma(c)
        for s in range(len(fa.States)):
            self.addState((1, s))
        for s in fa.delta:
            for c in fa.delta[s]:
                for t in fa.delta[s][c]:
                    self.addTransition(self.stateIndex((1, s)), c, self.stateIndex((1, t)))
        return (self.stateIndex((1, uSet(fa.Initial))),
                self.stateIndex((1, uSet(fa.Final))))

    def reverseTransitions(self, rev):
        """Evaluate reverse transition function.

        Args:
            rev (NFA): NFA in which the reverse function will be stored"""
        for s in self.delta:
            for a in self.delta[s]:
                for s1 in self.delta[s][a]:
                    rev.addTransition(s1, a, s)

    def initialComp(self):
        """Evaluate the connected component starting at the initial state.

        Returns:
            list of int: list of state indexes in the component"""
        lst = list(self.Initial)
        i = 0
        while True:
            try:
                foo = list(self.delta[lst[i]].keys())
            except KeyError:
                foo = []
            for c in foo:
                for _ in self.delta[lst[i]]:
                    for s in self.delta[lst[i]][c]:
                        if s not in lst:
                            lst.append(s)
            i += 1
            if i >= len(lst):
                return lst

    def finalCompP(self, s):
        """Verify whether there is a final state in strongly connected component containing given state.

        Args:
            s(int): state index
        Returns:
            bool:"""
        if s in self.Final:
            return True
        lst = [s]
        i = 0
        while True:
            try:
                foo = list(self.delta[lst[i]].keys())
            except KeyError:
                foo = []
            for c in foo:
                for s in self.delta[lst[i]][c]:
                    if s not in lst:
                        if s in self.Final:
                            return True
                        lst.append(s)
            i += 1
            if i >= len(lst):
                return False

    def deterministicP(self):
        """Verify whether this NFA is actually deterministic

        Returns:
            bool:"""
        if len(self.Initial) != 1:
            return False
        for st in self.delta:
            for sy in self.delta[st]:
                if sy == Epsilon or len(self.delta[st][sy]) > 1:
                    return False
        return True

    def _toDFAd(self):
        """Transforms into a DFA assuming it is deterministic

        Returns:
            DFA:the FA in a DFA structure"""
        # The subset construction will consider only accessible states
        new = DFA()
        new.Sigma = self.Sigma
        # self must be trim
        old = self.dup()
        old.trim()
        s = len(old.States)
        for i in range(s):
            new.addState(str(i))
        for i in old.delta:
            for c in old.delta[i]:
                new.addTransition(new.stateIndex(str(i)), c, new.stateIndex("{0:d}".format(uSet(old.delta[i][c]))))
        new.setInitial(new.stateIndex(str(uSet(old.Initial))))
        for i in old.Final:
            new.addFinal(new.stateIndex(str(i)))
        return new

    def homogenousP(self, x):
        """Whether this NFA is homogenous; that is, for all states, whether all incoming transitions to that state
        are through the same symbol.

        Args:
            x: dummy parameter to agree with the method in DFAr
        Returns:
            bool:"""
        return self.toNFAr().homogenousP(True)

    def stronglyConnectedComponents(self):
        """Strong components

        Returns:
            list:

        .. versionadded:: 1.0"""

        def _strongConnect(st):
            # todo This bombs out for a large automaton (loop exceed) Fixe it!
            indices[st] = index[0]
            lowlink[st] = index[0]
            index[0] += 1
            s.append(st)
            in_indices[st] = True
            in_s[st] = True
            links = [x for k in self.delta.get(st, {}) for x in self.delta[st][k]]
            # links = [self.delta[state][k] for k in self.delta.get(state, {})]
            for l in links:
                if not in_indices[l]:
                    _strongConnect(l)
                    lowlink[st] = min(lowlink[st], lowlink[l])
                elif in_s[l]:
                    lowlink[st] = min(lowlink[st], indices[l])
            if lowlink[st] == indices[st]:
                component = []
                while True:
                    l = s.pop()
                    in_s[l] = False
                    component.append(l)
                    if l == st:
                        break
                result.append(component)

        index = [0]
        indices = []
        lowlink = []
        s = []
        result = []
        in_indices = []
        in_s = []
        for _ in self.States:
            in_indices.append(False)
            in_s.append(False)
            indices.append(-1)
            lowlink.append(-1)
        for state in self.delta:
            if not in_indices[state]:
                _strongConnect(state)
        return result

    def dotFormat(self, size="20,20", filename=None, direction="LR", strict=False, maxlblsz=6, sep="\n") -> str:
        """ A dot representation

        Args:
            direction (str): direction of drawing - "LR" or "RL"
            size (str): size of image
            filename (str): output file name
            sep (str): line separator
            maxlblsz (int): max size of labels before getting removed
            strict (bool): use limitations of label sizes
        Returns:
            str: the dot representation

        .. versionadded:: 0.9.6

        .. versionchanged:: 1.2.1"""
        if not strict and max([len(str(name)) for name in self.States]) > maxlblsz:
            o = self.dup()
            o.renameStates()
        else:
            o = self
        s = "digraph finite_state_machine {{{0:s}".format(sep)
        s += "rankdir={0:s};{1:s}".format(direction, sep)
        s += "size=\"{0:s}\";{1:s}".format(size, sep)
        for si in o.Initial:
            sn = o.dotLabel(o.States[si])
            s += "node [shape = point]; \"dummy{0:s}\"{1:s}".format(sn, sep)
            s += o.dotDrawState(si)
            s += "\"dummy{0:s}\" -> \"{1:s}\";{2:s}".format(sn, graphvizTranslate(sn), sep)
        ni_states = [i for i in range(len(o.States)) if i not in o.Initial]
        for sti in ni_states:
            s += o.dotDrawState(sti)
        for si in o.succintTransitions():
            s += o.dotDrawTransition(si[0], si[1], si[2])
        s += "}}{0:s}".format(sep)
        return s

    def wordImage(self, word, ist=None):
        """Evaluates the set of states reached consuming given word

        Args:
            word: the word
            ist (int) : starting state index (or set of)
        Returns:
            set of int: the set of ending states"""
        if not ist:
            ist = self.Initial
        ilist = self.epsilonClosure(ist)
        for c in word:
            ilist = self.evalSymbol(ilist, c)
            if not ilist:
                return []
        return ilist

    def transitionsA(self):
        for si in self.delta:
            for c in self.delta[si]:
                yield si, c, self.delta[si][c]

    def transitions(self):
        for si in self.delta:
            for c1 in self.delta[si]:
                for s1 in self.delta[si][c1]:
                    yield si, c1, s1

    # todo: change product to avoid use of stateIndex()
    def product(self, other):
        """Returns a NFA (skeletom) resulting of the simultaneous execution of two DFA.

        Args:
            other (NFA): the other automata
        Returns:
            NFA:
        Raises:
            NFAerror: if any argument  has epsilon-transitions

        .. note::
           No final states are set.

        .. attention::
           - operands cannot have epsilon-transitions
           - the name ``EmptySet`` is used in a unique special state name
           - the method uses 3 internal functions for simplicity of code (really!)"""

        def _sN(a, s: int):
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

        def _dealT(srci, dest):
            """

            Args:
                srci (int): source state
                dest (int): destination state"""
            if not (dest in done or dest in not_done):
                i_n = new.addState(dest)
                not_done.append(dest)
            else:
                i_n = new.stateIndex(dest)
            new.addTransition(srci, k, i_n)

        if self.epsilonP() or other.epsilonP():
            raise NFAerror("Automata cannot have epsilon-transitions")
        new = NFA()
        new.setSigma(self.Sigma.union(other.Sigma))
        not_done = []
        done = []
        for s1 in [self.States[x] for x in self.Initial]:
            for s2 in [other.States[x] for x in other.Initial]:
                sname = (s1, s2)
                new.addState(sname)
                new.addInitial(new.stateIndex(sname))
                if (s1, s2) not in not_done:
                    not_done.append((s1, s2))
        while not_done:
            state = not_done.pop()
            done.append(state)
            (s1, s2) = state
            i = new.stateIndex(state)
            (i1, i2) = (_sN(self, s1), _sN(other, s2))
            (k1, k2) = (_kS(self, i1), _kS(other, i2))
            for k in k1.intersection(k2):
                for destination in [(self.States[d1], other.States[d2]) for d1 in self.delta[i1][k] for d2 in
                                    other.delta[i2][k]]:
                    _dealT(i, destination)
            for k in k1 - k2:
                for n in self.delta[i1][k]:
                    _dealT(i, (self.States[n], EmptySet))
            for k in k2 - k1:
                for n in other.delta[i2][k]:
                    _dealT(i, (EmptySet, other.States[n]))
        return new

    def _toNFASingleInitial(self):
        """Construct an equivalent NFA with only one initial state

        :rtype: NFA"""
        aut = self.dup()
        initial = aut.addState()
        aut.delta[initial] = {}
        aut.delta[initial][Epsilon] = aut.Initial
        aut.setInitial([initial])
        return aut

    def _deleteRefInDelta(self, src, sym, dest):
        """Deletion of a reference in Delta

        Args:
            src (int): source state
            sym (int): symbol
            dest (int): destination state"""
        if dest in self.delta[src][sym]:
            self.delta[src][sym].remove(dest)
        for k in range(dest + 1, len(self.States)):
            if k in self.delta[dest][sym]:
                self.delta[src][sym].remove(k)
                self.delta[src][sym].add(k - 1)
        if not len(self.delta[src][sym]):
            del self.delta[src][sym]
            if not len(self.delta[src]):
                del self.delta[src]

    def _deleteRefInitial(self, sti):
        """Deletes a state from the set of initial states.  The other states are renumbered.

        Args:
            sti (int): state index"""
        if sti in self.Initial:
            self.Initial.remove(sti)
        for s in self.Initial:
            if sti < s:
                self.Initial.remove(s)
                self.Initial.add(s - 1)

    def toNFA(self):
        """ Dummy identity function

        Returns:
            NFA:"""
        return self

    def toDFA(self):
        """Construct a DFA equivalent to this NFA, by the subset construction method.

        Returns:
            DFA:

        .. note::
           valid to epsilon-NFA"""
        if self.deterministicP():
            return self._toDFAd()
        dfa = DFA()
        l_states = []
        stl = self.epsilonClosure(self.Initial)
        l_states.append(stl)
        dfa.setInitial(dfa.addState(stl))
        dfa.setSigma(self.Sigma)
        for f in self.Final:
            if f in stl:
                dfa.addFinal(0)
                break
        index = 0
        while True:
            slist = l_states[index]
            si = dfa.stateIndex(slist)
            for s in self.Sigma:
                stl = self.evalSymbol(slist, s)
                if not stl:
                    continue
                if stl not in l_states:
                    l_states.append(stl)
                    foo = dfa.addState(stl)
                    for f in self.Final:
                        if f in stl:
                            dfa.addFinal(foo)
                            break
                else:
                    foo = dfa.stateIndex(stl)
                dfa.addTransition(si, s, foo)
            if index == len(l_states) - 1:
                break
            else:
                index += 1
        return dfa

    def hasTransitionP(self, state, symbol=None, target=None):
        """Whether there's a transition from given state, optionally through given symbol,
        and optionally to a specific target.

        Args:
            state (int): source state
            symbol (str): (optional) transition symbol
            target (int): (optional) target state
        Returns:
            bool: if there is a transition"""
        if state not in self.delta:
            return False
        if symbol is None:
            return True
        if symbol not in self.delta[state]:
            return False
        if target is None:
            return self.delta[state][symbol] != set()
        else:
            return target in self.delta[state][symbol]

    def usefulStates(self, initial_states=None):
        """Set of states reacheable from the given initial state(s) that have a path to a final state.

        Args:
            initial_states (set): set of initial states
        Returns:
            set:set of state indexes"""
        if initial_states is None:
            initial_states = self.Initial
        useful = set([s for s in initial_states
                      if s in self.Final])
        stack = list(initial_states)
        preceding = {}
        for i in stack:
            preceding[i] = []
        while stack:
            state = stack.pop()
            if state not in self.delta:
                continue
            for symbol in self.delta[state]:
                for adjacent in self.delta[state][symbol]:
                    is_useful = adjacent in useful
                    if adjacent in self.Final or is_useful:
                        useful.add(state)
                        if not is_useful:
                            useful.add(adjacent)
                            preceding[adjacent] = []
                            stack.append(adjacent)
                        inpath_stack = [p for p in preceding[state] if p not in useful]
                        preceding[state] = []
                        while inpath_stack:
                            previous = inpath_stack.pop()
                            useful.add(previous)
                            inpath_stack += [p for p in preceding[previous] if p not in useful]
                            preceding[previous] = []
                        continue
                    if adjacent not in preceding:
                        preceding[adjacent] = [state]
                        stack.append(adjacent)
                    else:
                        preceding[adjacent].append(state)
        if not useful and self.Initial:
            useful.add(min(self.Initial))
        return useful

    def eliminateEpsilonTransitions(self):
        """Eliminates all epslilon-transitions with no state addition

        .. attention::
           in-place modification"""
        for s in range(len(self.States)):
            if Epsilon in self.delta.get(s, []):
                for s1 in self.epsilonClosure(self.delta[s][Epsilon]):
                    if s1 in self.Final:
                        self.addFinal(s)
                    for a in self.delta.get(s1, []):
                        if a != Epsilon:
                            for s2 in self.delta[s1][a]:
                                self.addTransition(s, a, s2)
                foo = copy(self.delta[s][Epsilon])
                for s1 in foo:
                    self.delTransition(s, Epsilon, s1)
        self.trim()
        return self

    def HKeqP(self, other, strict=True):
        """
        Test NFA equivalence with extended Hopcroft-Karp method

        Args:
            other (NFA):
            strict (bool): if True checks for same alphabets
        Returns:
            bool:

        .. seealso::
            J. E. Hopcroft and r. M. Karp. A Linear Algorithm for Testing Equivalence of Finite Automata.TR 71--114. U.
            California. 1971"""
        if strict and self.Sigma != other.Sigma:
            return False
        n = len(self.States)
        if n == 0 or len(other.States) == 0:
            raise NFAEmpty
        i1 = frozenset(self.Initial)
        i2 = frozenset([i + n for i in other.Initial])
        s = UnionFind(auto_create=True)
        s.union(i1, i2)
        stack = [(i1, i2)]
        while stack:
            (p, q) = stack.pop()
            # test if p is in self
            lp = list(p)
            on_other = False
            if len(lp) > 0 and lp[0] >= n:
                on_other = True
            if on_other:
                if other.Final.isdisjoint(set([i - n for i in p])) != \
                        other.Final.isdisjoint(set([i - n for i in q])):
                    return False
            elif self.Final.isdisjoint(p) != other.Final.isdisjoint(set([i - n for i in q])):
                return False
            for sigma in self.Sigma:
                if on_other:
                    p1 = s.find(frozenset([j + n for j in other.evalSymbol(frozenset([i - n for i in p]), sigma)]))
                else:
                    p1 = s.find(frozenset(self.evalSymbol(p, sigma)))
                q1 = s.find(frozenset([j + n for j in other.evalSymbol(frozenset([i - n for i in q]), sigma)]))
                if p1 != q1:
                    s.union(p1, q1)
                    stack.append((p1, q1))
        return True

    def autobisimulation(self):
        """Largest right invariant equivalence between states of the NFA

        Returns:
            set: Incomplete equivalence relation (transitivity, and reflexivity not calculated) as a set of
            unordered pairs of states

        .. seealso:: L. Ilie and S. Yu, Follow automata Inf. Comput. 186 - 1, pp 140-162, 2003"""
        n_states = len(self.States)
        undecided_pairs = set([frozenset((i, j))
                               for i in range(n_states)
                               for j in range(i + 1, n_states)])
        marked = set()
        for pair in undecided_pairs:
            a, b = pair
            if (a in self.Final) != (b in self.Final):
                marked.add(pair)

        def _desc_marked(d_p, sym, q1, mrkd):
            for d_q in self.delta[q1][sym]:
                yield frozenset((d_p, d_q)) in mrkd

        changed_marked = True
        while changed_marked:
            changed_marked = False
            undecided_pairs.difference_update(marked)
            for pair in undecided_pairs:
                p, q = pair
                if p in self.delta:
                    if q not in self.delta or (set(self.delta[p].keys()) != set(self.delta[q].keys())):
                        marked.add(pair)
                        changed_marked = True
                    else:
                        for symbol in self.delta[p]:
                            for desc_p in self.delta[p][symbol]:
                                if all(_desc_marked(desc_p, symbol, q, marked)):
                                    marked.add(pair)
                                    changed_marked = True
                                    break
                            if pair in marked:
                                break
                if pair not in marked and q in self.delta:
                    if p not in self.delta:
                        marked.add(pair)
                        changed_marked = True
                    else:
                        for symbol in self.delta[q]:
                            for desc_q in self.delta[q][symbol]:
                                if all(_desc_marked(desc_q, symbol, p, marked)):
                                    marked.add(pair)
                                    changed_marked = True
                                    break
                            if pair in marked:
                                break
        undecided_pairs.difference_update(marked)
        return undecided_pairs

    # noinspection PyUnusedLocal
    def autobisimulation2(self):
        """Alternative space-efficient definition of NFA.autobisimulation.

        Returns:
            list: Incomplete equivalence relation (reflexivity, symmetry, and transitivity not calculated) as a set of
            pairs of states"""
        n_states = len(self.States)
        marked = set()
        for i in range(n_states):
            for j in range(i + 1, n_states):
                if (i in self.Final) != (j in self.Final):
                    marked.add((i, j))

        def _all_desc_marked(p, q, mrkd):
            for s in self.delta[p]:
                for desc_p in self.delta[p][s]:
                    all_marked = True
                    for desc_q in self.delta[q][s]:
                        if (desc_p, desc_q) not in mrkd and (desc_q, desc_p) not in mrkd:
                            all_marked = False
                            break
                    yield all_marked

        changed_marked = True
        while changed_marked:
            # noinspection PyUnusedLocal
            changed_marked = False
            for i in range(n_states):
                for j in range(i + 1, n_states):
                    if (i, j) in marked:
                        continue
                    if i not in self.delta and j not in self.delta:
                        continue
                    if set(self.delta.get(i, {}).keys()) != set(self.delta.get(j, {}).keys()):
                        marked.add((i, j))
                        changed_marked = True
                        continue
                    if any(_all_desc_marked(i, j, marked)) or any(_all_desc_marked(j, i, marked)):
                        marked.add((i, j))
                        changed_marked = True
                        continue
        return [(i, j)
                for i in range(n_states)
                for j in range(i + 1, n_states)
                if (i, j) not in marked]

    def equivReduced(self, equiv_classes):
        """Equivalent NFA reduced according to given equivalence classes.

        Args:
            equiv_classes (UnionFind): Equivalence classes
        Returns:
            NFA: Equivalent NFA"""
        nfa = NFA()
        nfa.setSigma(self.Sigma)
        rename_map = {}
        for istate in self.Initial:
            equiv_istate = equiv_classes.find(istate)
            equiv_istate_renamed = nfa.addState(equiv_istate)
            rename_map[equiv_istate] = equiv_istate_renamed
            nfa.addInitial(equiv_istate_renamed)
        for state in self.delta:
            equiv_state = equiv_classes.find(state)
            if equiv_state not in rename_map:
                equiv_state_renamed = nfa.addState(equiv_state)
                rename_map[equiv_state] = equiv_state_renamed
            else:
                equiv_state_renamed = rename_map[equiv_state]
            for symbol in self.delta[state]:
                for target in self.delta[state][symbol]:
                    equiv_target = equiv_classes.find(target)
                    if equiv_target not in rename_map:
                        equiv_target_renamed = nfa.addState(equiv_target)
                        rename_map[equiv_target] = equiv_target_renamed
                    else:
                        equiv_target_renamed = rename_map[equiv_target]
                    nfa.addTransition(equiv_state_renamed, symbol, equiv_target_renamed)
        for state in self.Final:
            equiv_state = equiv_classes.find(state)
            if equiv_state not in rename_map:
                rename_map[equiv_state] = nfa.addState(equiv_state)
            nfa.addFinal(rename_map[equiv_state])
        return nfa

    def rEquivNFA(self):
        """Equivalent NFA obtained from merging equivalent states from autobisimulation of this NFA.

        :rtype: NFA

        .. note::
           returns copy of self if autobisimulation renders no equivalent states."""
        autobisimulation = self.autobisimulation()
        if not autobisimulation:
            return self.dup()
        equiv_classes = UnionFind(auto_create=True)
        for i in range(len(self.States)):
            equiv_classes.make_set(i)
        for i, j in autobisimulation:
            equiv_classes.union(i, j)
        return self.equivReduced(equiv_classes)

    def lEquivNFA(self):
        """Equivalent NFA obtained from merging equivalent states from autobisimulation of this NFA's reversal.

        Returns:
            NFA:

        .. note::
           returns copy of self if autobisimulation renders no equivalent states."""
        autobisimulation = self.reversal().autobisimulation()
        if not autobisimulation:
            return self.dup()
        equiv_classes = UnionFind(auto_create=True)
        for i in range(len(self.States)):
            equiv_classes.make_set(i)
        for i, j in autobisimulation:
            equiv_classes.union(i, j)
        return self.equivReduced(equiv_classes)

    def lrEquivNFA(self):
        """Equivalent NFA obtained from merging equivalent states from autobisimulation of this NFA,
        and from autobisimulation of its reversal; i.e., merges all states that are equivalent w.r.t. the largest
        right invariant and largest left invariant equivalence relations.

        Returns:
            NFA:

        .. note::
           returns copy of self if autobisimulations render no equivalent states."""
        l_nfa = self.lEquivNFA()
        lr_nfa = l_nfa.rEquivNFA()
        del l_nfa
        return lr_nfa

    def epsilonPaths(self, start, end):
        """All states in all paths (DFS) through empty words from a given starting state to a given ending state.

        Args:
            start (int): start state
            end (int): end state
        Returns:
            set: states in epsilon paths from start to end"""
        inpaths = set()
        stack = [start]
        preceding = {start: []}
        while stack:
            state = stack.pop()
            if self.hasTransitionP(state, Epsilon):
                for adjacent in self.delta[state][Epsilon]:
                    if adjacent is end or adjacent in inpaths:
                        inpaths.add(state)
                        inpath_stack = [p for p in preceding[state] if p not in inpaths]
                        preceding[state] = []
                        while inpath_stack:
                            previous = inpath_stack.pop()
                            inpaths.add(previous)
                            inpath_stack += [p for p in preceding[previous] if p not in inpaths]
                            preceding[previous] = []
                        continue
                    if adjacent not in preceding:
                        preceding[adjacent] = [state]
                        stack.append(adjacent)
                    else:
                        preceding[adjacent].append(state)
        return inpaths

    def toNFAr(self):
        """NFA with the reverse mapping of the delta function.

        Returns:
            NFAr: shallow copy with reverse delta function added"""
        nfa_r = NFAr()
        nfa_r.setInitial(self.Initial)
        nfa_r.setFinal(self.Final)
        nfa_r.setSigma(self.Sigma)
        nfa_r.States = list(self.States)
        for source in self.delta:
            for symbol in self.delta[source]:
                for target in self.delta[source][symbol]:
                    nfa_r.addTransition(source, symbol, target)
        return nfa_r

    def homogeneousFinalityP(self):
        """ Tests if states have incoming transitions froms states with different finalities

        Returns:
            bool:"""
        sr = self.toNFAr()
        for i in sr.stateIndexes():
            l = []
            for c in sr.deltaReverse.get(i, []):
                for j in sr.deltaReverse[i][c]:
                    l.append(j in sr.Final)
            if not homogeneousP(l):
                return False
        return True

    def countTransitions(self):
        """Count the number of transitions of a NFA

        Returns:
            int:"""
        return sum([sum(map(len, iter(self.delta[t].values())))
                    for t in self.delta])

    def stateChildren(self, state, strict=False):
        """Set of children of a state

        Args:
            state (int): state id queried
            strict (bool): if not strict a state is never its own child even if a self loop is in place
        Returns:
            set: children states"""
        l = set([])
        if state not in list(self.delta.keys()):
            return l
        for c in self.Sigma:
            if c in self.delta[state]:
                l += self.delta[state][c]
        if not strict:
            if state in l:
                l.remove(state)
        return l

    def half(self):
        """Half operation

        Returns:
            NFA:

        .. versionadded:: 0.9.6"""
        a1 = self.dup()
        a1.renameStates()
        a2 = a1.dup().reversal()
        a4 = a2._starTransitions()
        a3 = a1.product(a4)
        l = []
        for n1, n2 in a3.States:
            if n1.__str__() == "@empty_set" or n2.__str__() == "@empty_set":
                l.append((n1, n2))
            if n1.__str__() == n2.__str__():
                a3.addFinal(a3.stateIndex((n1, n2)))
        a3.deleteStates(list(map(a3.stateIndex, l)))
        return a3

    def _starTransitions(self):
        new = NFA()
        for _ in self.States:
            new.addState()
        for s in self.delta:
            for c in self.delta[s]:
                for s1 in self.delta[s][c]:
                    for c1 in self.Sigma:
                        new.addTransition(s, c1, s1)
        for s in self.Initial:
            new.addInitial(s)
        for s in self.Final:
            new.addFinal(s)
        return new

    def subword(self):
        """NFA that recognizes subword(L(self))

        Returns:
            NFA:"""
        c = self.dup()
        c.trim()
        for s in c.delta:
            ss = set([])
            for sym in c.delta[s]:
                ss.update(c.delta[s][sym])
            if Epsilon not in c.delta[s]:
                c.delta[s][Epsilon] = set([])
            c.delta[s][Epsilon].update(ss)
        return c

    def enumNFA(self, n=None):
        """The set of words of length up to n accepted by self

        Args:
            n (int): highest lenght or all words if finite

        Returns:
            list: list of strings or None

        .. note: use with care because the number of words can be huge"""
        d = self.dup()
        d.elimEpsilon()
        e = EnumNFA(d)
        if n is None:
            return None
        words = []
        for i in range(n + 1):
            e.enumCrossSection(i)
            words += e.Words
        return words


# noinspection PyTypeChecker
class NFAr(NFA):
    """Class for Non-deterministic Finite Automata with reverse delta function added by construction.
            **Includes efficient methods for merging states.**

    .. inheritance-diagram:: NFAr"""

    def __init__(self):
        super(NFAr, self).__init__()
        self.deltaReverse = {}

    def addTransition(self, sti1, sym, sti2):
        """Adds a new transition. Transition is from ``sti1`` to ``sti2`` consuming symbol ``sym``. ``sti2`` is a
        unique state, not a set of them. Reversed transition function  is also computed

        Args:
            sti1 (int): state index of departure
            sti2 (int): state index of arrival
            sym (str): symbol consumed"""
        super(NFAr, self).addTransition(sti1, sym, sti2)
        if sti2 not in self.deltaReverse:
            self.deltaReverse[sti2] = {sym: {sti1}}
        elif sym not in self.deltaReverse[sti2]:
            self.deltaReverse[sti2][sym] = {sti1}
        else:
            self.deltaReverse[sti2][sym].add(sti1)

    def delTransition(self, sti1, sym, sti2, _no_check=False):
        """Remove a transition if existing and perform cleanup on the transition function's internal data structure
        and in the reversal transition function

        Args:
            sti1 (int): state index of departure
            sti2 (int): state index of arrival
            sym (str): symbol consumed
            _no_check (bool): (optional) dismiss secure code"""
        super(NFAr, self).delTransition(sti1, sym, sti2, _no_check)
        if not _no_check and (sti2 not in self.deltaReverse or sym not in self.deltaReverse[sti2]):
            return
        self.deltaReverse[sti2][sym].discard(sti1)
        if not self.deltaReverse[sti2][sym]:
            del self.deltaReverse[sti2][sym]
            if not self.deltaReverse[sti2]:
                del self.deltaReverse[sti2]

    def deleteStates(self, del_states):
        """Delete given iterable collection of states from the automaton. Performe deletion in the transition
        function and its reversal.

        Args:
            del_states (set|list): collection of states indexes"""
        super(NFAr, self).deleteStates(del_states)
        new_delta_reverse = {}
        for target in self.delta:
            for symbol in self.delta[target]:
                for source in self.delta[target][symbol]:
                    if source not in new_delta_reverse:
                        new_delta_reverse[source] = {}
                    if symbol not in new_delta_reverse[source]:
                        new_delta_reverse[source][symbol] = set()
                    new_delta_reverse[source][symbol].add(target)
        self.deltaReverse = new_delta_reverse

    def mergeStates(self, f, t):
        """Merge the first given state into the second. If first state is an initial or final state,
        the second becomes respectively an initial or final state.

        Args:
            f (int): index of state to be absorbed
            t (int): index of remaining state

        .. attention::
           It is up to the caller to remove the disconnected state. This can be achieved with ```trim()``."""
        if f is t:
            return
        if f in self.delta:
            for symbol in self.delta[f]:
                for state in self.delta[f][symbol]:
                    self.deltaReverse[state][symbol].remove(f)
                    if state is f:
                        state = t
                    if state is t and symbol is Epsilon:
                        continue
                    self.addTransition(t, symbol, state)
                    if not self.deltaReverse[state][symbol]:
                        del (self.deltaReverse[state][symbol])
            del (self.delta[f])
        if f in self.deltaReverse:
            for symbol in self.deltaReverse[f]:
                for state in self.deltaReverse[f][symbol]:
                    if state is f:
                        state = t
                    else:
                        self.delta[state][symbol].remove(f)
                    if state is t and symbol is Epsilon:
                        continue
                    self.addTransition(state, symbol, t)
                    if not self.delta[state][symbol]:
                        del (self.delta[state][symbol])
            del (self.deltaReverse[f])
        if f in self.Initial:
            self.Initial.remove(f)
            self.addInitial(t)
        if f in self.Final:
            self.Final.remove(f)
            self.addFinal(t)

    def mergeStatesSet(self, tomerge, target=None):
        """Merge a set of states with a target merge state. If the states in the set have transitions among them,
        those transitions will be directly merged into the target state.

        Args:
            tomerge (set): set of states to merge with target
            target (int): optional target state

        .. note::
           if target state is not given, the minimal index with be considered.

        .. attention::
           The states of the list will become unreacheable, but won't be removed. It is up to the caller to remove
           them. That can be achieved with ``trim()``."""
        if not tomerge:
            return
        if not target:
            target = min(tomerge)
        # noinspection PyUnresolvedReferences
        tomerge.discard(target)
        for state in tomerge:
            if state in self.delta:
                for symbol in self.delta[state]:
                    for s in self.delta[state][symbol]:
                        self.deltaReverse[s][symbol].discard(state)
                        if s in tomerge:
                            s = target
                        if symbol is Epsilon and s is target:
                            continue
                        self.addTransition(target, symbol, s)
            if state in self.deltaReverse:
                for symbol in self.deltaReverse[state]:
                    for s in self.deltaReverse[state][symbol]:
                        self.delta[s][symbol].discard(state)
                        if s in tomerge:
                            s = target
                        if symbol is Epsilon and s is target:
                            continue
                        self.addTransition(s, symbol, target)
                del (self.deltaReverse[state])
            if state in self.delta:
                del (self.delta[state])
        if target in self.delta:
            for symbol in self.delta[target]:
                for state in self.delta[target][symbol].copy():
                    if state in tomerge:
                        self.delta[target][symbol].discard(state)
                        if symbol is Epsilon:
                            continue
                        self.delta[target][symbol].add(target)
        if self.Initial.intersection(tomerge):
            self.addInitial(target)
        if self.Final.intersection(tomerge):
            self.addFinal(target)
        return target

    def homogenousP(self, inplace=False):
        """Checks is the automaton is homogenous, i.e.the transitions that reaches a state have all the same label.

        Args:
            inplace (bool): if True performs Epsilon transitions elimination
        Return:
            bool: True if homogenous"""
        nfa = self
        if self.epsilonP():
            if inplace:
                self.elimEpsilon()
            else:
                nfa = self.dup()
                nfa.elimEpsilon()
        return all([len(m) == 1 for m in nfa.deltaReverse.values()])

    def elimEpsilonO(self):
        """Eliminate epsilon-transitions from this automaton, with reduction of states through elimination of
        Epsilon-cycles, and single epsilon-transition cases.

        Returns:
            NFAr:

        .. attention::
           performs inplace modification of automaton"""
        for state in self.delta:
            if state not in self.delta:
                continue
            merge_states = self.epsilonPaths(state, state)
            merge_states.add(self.unlinkSoleOutgoing(state))
            merge_states.add(self.unlinkSoleIncoming(state))
            merge_states.discard(None)
            if merge_states:
                if len(merge_states) == 1:
                    self.mergeStates(state, merge_states.pop())
                else:
                    self.mergeStatesSet(merge_states)
        super(NFAr, self).elimEpsilon()
        self.trim()
        return self

    def unlinkSoleIncoming(self, state):
        """If given state has only one incoming transition (indegree is one), and it's through epsilon,
        then remove such transition and return the source state.

        Args:
            state (int): state to check
        Returns:
            int | None: source state

        .. note::
           if conditions aren't met, returned source state is None, and automaton remains unmodified."""
        if not len(self.deltaReverse.get(state, [])) == 1 or not len(self.deltaReverse[state].get(Epsilon, [])) == 1:
            return None
        source_state = self.deltaReverse[state][Epsilon].pop()
        self.delTransition(source_state, Epsilon, state, True)
        return source_state

    def unlinkSoleOutgoing(self, state):
        """If given state has only one outgoing transition (outdegree is one), and it's through epsilon,
        then remove such transition and return the target state.

        Args:
            state (int): state to check
        Returns:
            int | None: target state

        .. note::
           if conditions aren't met, returned target state is None, and automaton remains unmodified."""
        if not len(self.delta.get(state, [])) == 1 or not len(self.delta[state].get(Epsilon, [])) == 1:
            return None
        target_state = self.delta[state][Epsilon].pop()
        self.delTransition(state, Epsilon, target_state, True)
        return target_state

    def toNFA(self):
        """Turn into an instance of NFA, and remove the reverse mapping of the delta function.

        Returns:
            NFA: shallow copy without reverse delta function"""
        nfa = NFA()
        nfa.Initial = self.Initial
        nfa.Final = self.Final
        nfa.delta = self.delta
        nfa.Sigma = self.Sigma
        nfa.States = self.States
        return nfa


# noinspection PyTypeChecker
class DFA(OFA):
    """ Class for Deterministic Finite Automata.

    :ivar list States: set of states.
    :ivar set sigma: alphabet set.
    :ivar int Initial: the initial state index.
    :ivar set Final: set of final states indexes.
    :ivar dict delta: the transition function.
    :ivar dict delta_inv: possible inverse transition map
    :ivar bool i: is inverse map computed?

    .. inheritance-diagram:: DFA"""

    def __init__(self):
        super(DFA, self).__init__()
        self.delta_inv = None
        self.i = None

    @staticmethod
    def _vDescription():
        """Generation of Verso interface description

        .. versionadded:: 0.9.5

        Returns:
            str: the interface list"""
        return [("DFA", "Deterministic Finite Automata"),
                [("DFAFAdo", lambda x: saveToString(x), "FAdo"),
                 ("DFAdot", lambda x: x.dotFormat("&"), "dot")],
                ("DFA-complete-minimal", ("Complete minimal automata", "Complete minimal automata"), 1, "DFA", "DFA",
                 lambda *x: x[0].completeMinimal()),
                ("DFA-concatenation", ("Concatenate two DFAs", "Concatenate two DFAs"), 2, "DFA", "DFA", "DFA",
                 lambda *x: x[0].concat(x[1])),
                ("DFA-conjunction", ("Intersection of DFAs", "Intersection of DFAs"), 2, "DFA", "DFA", "DFA",
                 lambda *x: x[0].conjunction(x[1])),
                ("DFA-disjunction", ("Disjunction of DFAs", "Disjunction of DFAs"), 2, "DFA", "DFA", "DFA",
                 lambda *x: x[0].disjunction(x[1])),
                ("DFA-to-NFA", ("Convert to NFA", "Convert to NFA"), 1, "DFA", "NFA", lambda *x: x[0].toNFA()),
                ("DFA-acyclicP", ("Test if automata is acyclic", "Test if automata is acyclic"), 1, "DFA", "Bool",
                 lambda *x: x[0].acyclicP()),
                ("DFA-trim", ("Trim automata", "Trim automata"), 1, "DFA", None, lambda *x: x[0].trim()),
                ("DFA-trimP", ("Test if automata is trim", "Test if automata is trim"), 1, "DFA", "Bool",
                 lambda *x: x[0].trimP()),
                ("DFA-to-reversal-NFA", ("Reversal NFA", "Reversal NFA"), 1, "DFA", "NFA", lambda *x: x[0].reversal()),
                ("DFA-minimal-Brzozowski", ("Minimal (Brzozowski)", "Minimal (Brzozowski)"), 1, "DFA", "DFA",
                 lambda *x: x[0].minimalBrzozowski()),
                ("DFA-minimalP-Brzozowski", ("Test minimality (Brzozowski)", "Test minimality (Brzozowski)"), 1, "DFA",
                 "Bool",
                 lambda *x: x[0].minimalBrzozowskiP()),
                ("DFA-RegExp-SE", ("Convert to RE", "Convert to RE by state elimination"), 1, "DFA", "RE",
                 lambda *x: x[0].regexpSE()),
                ("DFA-dump", ("dump", "dump"), 1, "DFA", "str", lambda *x: saveToString(x[0]))]

    def __repr__(self):
        return 'DFA({0:>s})'.format(self.__str__())

    @staticmethod
    def deterministicP():
        """Yes it is deterministic!

        Returns:
            bool:"""
        return True

    def succintTransitions(self):
        """ Collects the transition information in a compact way suitable for graphical representation.

        Returns:
            list: list of tupples

        .. note:
            tupples in the list are stateout, label, statein

        .. versionadded:: 0.9.8"""
        foo = dict()
        for s in self.delta:
            for c in self.delta[s]:
                k = (s, self.delta[s][c])
                if k not in foo:
                    foo[k] = []
                foo[k].append(c)
        lst = []
        for k in foo:
            cs = foo[k]
            s = "%s" % str(cs[0])
            for c in cs[1:]:
                s += ", %s" % str(c)
            lst.append((self.dotLabel(self.States[k[0]]), s, self.dotLabel(self.States[k[1]])))
        return lst

    def initialP(self, state):
        """ Tests if a state is initial

        Args:
            state (int): state index
        Returns:
            bool:"""
        return self.Initial == state

    def _getTags(self):
        return ["DFA"]

    def initialSet(self):
        """The set of initial states

        Returns:
            set: the set of the initial states"""
        return {self.Initial}

    def Delta(self, state, symbol):
        """Evaluates the action of a symbol over a state

        Args:
            state (int): state index
            symbol (Any): symbol
        Returns:
            int: the action of symbol over state"""
        try:
            r = self.delta[state][symbol]
        except KeyError:
            r = None
        return r

    def _deleteRefInDelta(self, src, sym, dest):
        old = self.delta.get(src, {}).get(sym, -1)
        if dest == old:
            del self.delta[src][sym]
        elif old > dest:
            self.delta[src][sym] = old - 1
        if not len(self.delta[src]):
            del self.delta[src]

    def _deleteRefInitial(self, sti):
        if sti < self.Initial:
            self.Initial -= 1
        if sti == self.Initial:
            self.Initial = None

    def deleteStates(self, del_states):
        """Delete given iterable collection of states from the automaton.

        Args:
            del_states: collection of state indexes

        .. note:: in-place action

        .. note::
           delta function will always be rebuilt, regardless of whether the states list to remove is a suffix,
           or a sublist, of the automaton's states list."""
        if not del_states:
            return
        rename_map = {}
        old_delta = self.delta
        self.delta = {}
        new_final = set()
        new_states = []
        for state in del_states:
            if self.initialP(state):
                self.Initial = None
        for state in range(len(self.States)):
            if state not in del_states:
                rename_map[state] = len(new_states)
                new_states.append(self.States[state])
        for state in rename_map:
            state_renamed = rename_map[state]
            if state in self.Final:
                new_final.add(state_renamed)
            if state not in old_delta:
                continue
            for symbol, target in old_delta[state].items():
                if target in rename_map:
                    self.addTransition(state_renamed, symbol, rename_map[target])
        self.States = new_states
        self.Final = new_final
        if self.Initial is not None:
            # noinspection PyNoneFunctionAssignment
            self.Initial = rename_map.get(self.Initial, None)

    def addTransition(self, sti1, sym, sti2):
        """Adds a new transition from sti1 to sti2 consuming symbol sym.

        Args:
            sti1 (int): state index of departure
            sti2 (int): state index of arrival
            sym (Any): symbol consumed
        Raises:
            DFAnotNFA: if one tries to add a non-deterministic transition"""
        if sym == Epsilon:
            raise DFAnotNFA("Invalid Epsilon transition from {0:>s} to {1:>s}.".format(str(sti1), str(sti2)))
        self.Sigma.add(sym)
        if sti1 not in self.delta:
            self.delta[sti1] = {sym: sti2}
        else:
            if sym in self.delta[sti1] and self.delta[sti1][sym] is not sti2:
                raise DFAnotNFA("extra transition from ({0:>s}, {1:>s})".format(str(sti1), sym))
            self.delta[sti1][sym] = sti2

    def delTransition(self, sti1, sym, sti2, _no_check=False):
        """Remove a transition if existing and perform cleanup on the transition function's internal data structure.

        Args:
            sti1 (int): state index of departure
            sym (Any): symbol consumed
            sti2 (int): state index of arrival
            _no_check (bool): use unsecure code?

        .. note::
           Unused alphabet symbols will be discarded from sigma."""
        if not _no_check and (sti1 not in self.delta or sym not in self.delta[sti1]):
            return
        if self.delta[sti1][sym] is not sti2:
            return
        del self.delta[sti1][sym]
        if all([sym not in x for x in iter(self.delta.values())]):
            self.Sigma.discard(sym)
        if not self.delta[sti1]:
            del self.delta[sti1]

    def inDegree(self, st):
        """Returns the in-degree of a given state in an FA

        Args:
            st (int): index of the state
        Returns:
            int:"""
        in_deg = 0
        for s in range(len(self.States)):
            for a in self.Sigma:
                try:
                    if self.delta[s][a] == st:
                        in_deg += 1
                except KeyError:
                    pass
        return in_deg

    def syncPower(self):
        """Evaluates the Power automata for the action of each symbol

        Returns:
            DFA: The Power automata being the set of all states the initial state and all singleton states final"""
        new = DFA()
        new.setSigma(self.Sigma)
        a = set(range((len(self.States))))
        tbd = [a]
        done = []
        ia = new.addState(a)
        new.setInitial(ia)
        while tbd:
            a = tbd.pop()
            ia = new.stateIndex(a)
            done.append(a)
            for sy in new.Sigma:
                b = set([self.Delta(s, sy) for s in a])
                b.discard(None)
                if b not in done:
                    if b not in tbd:
                        tbd.append(b)
                        ib = new.addState(b)
                        if len(b) == 1:
                            new.addFinal(ib)
                    else:
                        ib = new.stateIndex(b)
                    new.addTransition(ia, sy, ib)
                else:
                    new.addTransition(ia, sy, new.stateIndex(b))
        return new

    def pairGraph(self):
        """Returns pair graph

        Returns:
            DiGraphVM:

        .. seealso::
           A graph theoretic apeoach to automata minimality. Antonio Restivo and Roberto Vaglica. Theoretical
           Computer Science, 429 (2012) 282-291. doi:10.1016/j.tcs.2011.12.049 Theoretical Computer Science,
           2012 vol. 429 (C) pp. 282-291. http://dx.doi.org/10.1016/j.tcs.2011.12.049"""
        g = graphs.DiGraphVm()
        for s1 in range(len(self.States)):
            for s2 in range(s1, len(self.States)):
                i1 = g.vertexIndex((self.States[s1], self.States[s2]), True)
                for sy in self.delta[s1]:
                    if sy in self.delta[s2]:
                        foo = [self.delta[s1][sy], self.delta[s2][sy]]
                        foo.sort()
                        i2 = g.vertexIndex((self.States[foo[0]], self.States[1]), True)
                        g.addEdge(i1, i2)
        return g

    def subword(self):
        """A dfa that recognizes subword(L(self))

        Returns:
            DFA:

        .. versionadded:: 1.1"""

        if not self.hasTrapStateP():
            return sigmaStarDFA(self.Sigma)
        return self.toNFA().subword().toDFA()

    def prefix_free_p(self):
        """ Checks is a DFA is prefix-free
        Returns:
            bool:

        .. versionadded:: 2.0.3"""
        new = self.dup().trim()
        for i in new.Final:
            if i in new.delta:
                return False
        return True

    def make_prefix_free(self):
        """ Turns a DFA in a prefix-free automaton deleting all outgoing transitions from final states

        Returns:
            DFA:

        .. versionadded:: 2.0.3"""
        new = self.dup()
        for i in new.Final:
            try:
                del(new.delta[i])
            except KeyError:
                pass
        return new

    def make_bifix_free(self):
        new = self.dup()
        new = new.make_prefix_free()
        new = new.reversal().toDFA()
        new = new.make_prefix_free()
        return new

    def pref(self):
        """Returns a dfa that recognizes pref(L(self))

        Returns:
            DFA:

        .. versionadded:: 1.1
        """
        foo = self.dup()
        foo.trim()
        if foo.emptyP():
            return foo
        foo.setFinal(list(range(len(foo.States))))
        return foo

    def suff(self):
        """ Returns a dfa that recognizes suff(L(self))

        Returns:
            DFA:

        .. versionadded:: 0.9.8"""
        d = DFA()
        d.setSigma(self.Sigma)
        ini = self.usefulStates()
        l_states = []
        d.setInitial(d.addState(ini))
        l_states.append(ini)
        if not self.Final.isdisjoint(ini):
            d.addFinal(0)
        index = 0
        while True:
            slist = l_states[index]
            si = d.stateIndex(slist)
            for s in self.Sigma:
                stl = set([self.evalSymbol(s1, s) for s1 in slist if s in self.delta[s1]])
                if not stl:
                    continue
                if stl not in l_states:
                    l_states.append(stl)
                    foo = d.addState(stl)
                    if not self.Final.isdisjoint(stl):
                        d.addFinal(foo)
                else:
                    foo = d.stateIndex(stl)
                d.addTransition(si, s, foo)
            if index == len(l_states) - 1:
                break
            else:
                index += 1
        return d

    def infix(self):
        """ Returns a dfa that recognizes infix(L(a))

        Returns:
            DFA:"""
        m = self.minimal()
        m.complete()
        trap = None
        for i in range(len(m.States)):
            if m.finalP(i):
                continue
            f = 0
            for c in m.delta[i]:
                if m.delta[i][c] != i:
                    f = 1
                    break
            if f == 0:
                trap = i
                break
        if trap is None:
            return sigmaStarDFA(self.Sigma)
        else:
            d = DFA()
            d.setSigma(m.Sigma)
            ini = set(range(len(m.States))).difference({trap})
            d.setInitial(d.addState(ini))
            l_states = [ini]
            d.addFinal(0)
            index = 0
            while True:
                slist = l_states[index]
                si = d.stateIndex(slist, auto_create=True)
                for s in m.Sigma:
                    stl = set([m.evalSymbol(s1, s) for s1 in slist if s in m.delta[s1]])
                    if not stl:
                        continue
                    if stl not in l_states:
                        l_states.append(stl)
                        foo = d.addState(stl)
                        if stl != {trap}:
                            d.addFinal(foo)
                    else:
                        foo = d.stateIndex(stl, auto_create=True)
                    d.addTransition(si, s, foo)
                if index == len(l_states) - 1:
                    break
                else:
                    index += 1
        return d

    def hasTrapStateP(self):
        """ Tests if the automaton has a dead trap state

        Returns:
            bool:

        .. versionadded:: 1.1"""
        foo = self.minimal()
        if not foo.completeP():
            return True
        for i in range(len(foo.States)):
            if foo.finalP(i):
                continue
            f = 0
            for c in foo.delta[i]:
                if foo.delta[i][c] != i:
                    f = 1
                    break
            if f == 0:
                return True
        return False

    def _xA(self):
        """ Computes the minimal words that reach each state of DFA

        Returns:
            dict: dictionary with words"""
        x_list = dict()
        todo = [i for i in range(len(self.States))]
        if isinstance(self.Initial, set):
            rank = self.Initial
        else:
            rank = {self.Initial}

        for i in rank:
            x_list[i] = Epsilon
            todo.remove(i)
        while todo:
            nrank = set()
            for sym in self.Sigma:
                for i in rank:
                    if i in self.delta and sym in self.delta[i]:
                        ss = self.delta[i][sym]
                        if isinstance(ss, set):
                            for q in self.delta[i][sym]:
                                if q in todo:
                                    x_list[q] = sConcat(x_list[i], sym)
                                    todo.remove(q)
                                    nrank.add(q)
                        else:
                            q = ss
                            if q in todo:
                                x_list[q] = sConcat(x_list[i], sym)
                                todo.remove(q)
                                nrank.add(q)
            rank = nrank
        return x_list

    def sop(self, other):
        """ Strange operation

        Args:
            other (DFA): the other automaton
        Returns:
            DFA:

        .. seealso:: Nelma Moreira, Giovanni Pighizzini, and Rogério Reis. Universal disjunctive concatenation
            and star. In Jeffrey Shallit and Alexander Okhotin, editors, Proceedings of the 17th Int. Workshop on
            Descriptional Complexity of Formal Systems (DCFS15), number 9118 in LNCS, pages 197--208. Springer, 2015.

        .. versionadded:: 1.2b2"""
        a = self.dup()
        b = other.dup()
        if not a.completeP() or not a.completeP() or a.Sigma != b.Sigma:
            raise DFAnotComplete()
        aux = NFA()
        idx = aux.addState((a.Initial, b.Initial, 0))
        aux.addInitial(idx)
        aux.setSigma(a.Sigma)
        pool, done = _initPool()
        _addPool(pool, done, idx)
        while pool:
            idx = pool.pop()
            done.add(idx)
            t = aux.States[idx]
            for c in a.Sigma:
                if t[2] == 0:
                    nt = (a.delta[t[0]][c], t[1], 0)
                    i = aux.stateIndex(nt, True)
                    _addPool(pool, done, i)
                    aux.addTransition(idx, c, i)
                    nt = (t[0], b.delta[t[1]][c], 1)
                    i = aux.stateIndex(nt, True)
                    _addPool(pool, done, i)
                    aux.addTransition(idx, c, i)
                else:
                    nt = (t[0], b.delta[t[1]][c], 1)
                    i = aux.stateIndex(nt, True)
                    _addPool(pool, done, i)
                    aux.addTransition(idx, c, i)
        new = DFA()
        t = set()
        for idx in aux.Initial:
            t.add(aux.States[idx])
        idx = new.addState(t)
        new.setInitial(idx)
        pool, done = _initPool()
        _addPool(pool, done, idx)
        while pool:
            idx = pool.pop()
            done.add(idx)
            t = new.States[idx]
            for c in aux.Sigma:
                dest = set()
                for s in t:
                    for j in aux.delta[aux.stateIndex(s)].get(c, set()):
                        dest.add(aux.States[j])
                i = new.stateIndex(dest, True)
                _addPool(pool, done, i)
                new.addTransition(idx, c, i)
        for t in new.States:
            final = True
            for s in t:
                final = final and (s[0] in a.Final or s[1] in b.Final)
            if final:
                new.addFinal(new.stateIndex(t))
        return new

    def dist(self):
        """Evaluate the distinguishability language for a DFA

        Returns:
            DFA:

        .. seealso::
           Cezar Câmpeanu, Nelma Moreira, Rogério Reis:
           The distinguishability operation on regular languages. NCMA 2014: 85-100

        .. versionadded:: 0.9.8"""
        d = DFA()
        if not self.completeP():
            foo = self.dup()
            foo.complete()
        else:
            foo = self
        d.setSigma(foo.Sigma)
        ini = set(range(len(foo.States)))
        l_states = []
        d.setInitial(d.addState(ini))
        l_states.append(ini)
        if not foo.Final.isdisjoint(ini) and not ini.issubset(foo.Final):
            d.addFinal(0)
        index = 0
        while True:
            slist = l_states[index]
            si = d.stateIndex(slist)
            for s in foo.Sigma:
                stl = set([foo.evalSymbol(s1, s) for s1 in slist if s in foo.delta[s1]])
                if not stl:
                    continue
                if stl not in l_states:
                    l_states.append(stl)
                    new = d.addState(stl)
                    if not foo.Final.isdisjoint(stl) and not stl.issubset(foo.Final):
                        d.addFinal(new)
                else:
                    new = d.stateIndex(stl)
                d.addTransition(si, s, new)
            if index == len(l_states) - 1:
                break
            else:
                index += 1
        return d

    def distMin(self):
        """ Evaluates the list of minimal words that distinguish each pair of states

        Returns:
            set of minimal distinguishing words (FL):

        .. versionadded:: 0.9.8

        .. attention::
            If the DFA is not minimal, the method loops forever"""
        from . import fl

        sz = len(self.States)
        if sz == 1:
            return fl.FL()
        dist_list = fl.FL(Sigma=self.Sigma)
        todo = [(s, s1) for s in range(sz) for s1 in range(s + 1, sz)]
        wrds = self.words()
        l = []
        for (i, j) in todo:
            if (i in self.Final) ^ (j in self.Final):
                l.append((i, j))
                dist_list.addWord(Word(Epsilon))
        delFromList(todo, l)
        while True:
            for w in wrds:
                l = []
                for (i, j) in todo:
                    if self.evalWordP(w, i) ^ self.evalWordP(w, j):
                        l.append((i, j))
                        dist_list.addWord(w)
                delFromList(todo, l)
                if not todo:
                    return dist_list

    def distR(self):
        """Evaluate the right distinguishability language for a DFA

        Returns:
            DFA:

        ..seealso:: Cezar Câmpeanu, Nelma Moreira, Rogério Reis:
           The distinguishability operation on regular languages. NCMA 2014: 85-100"""
        foo = self.minimal()
        foo.complete()
        foo.delFinals()
        for i in range(len(foo.States)):
            f = 0
            for c in foo.delta[i]:
                if foo.delta[i][c] != i:
                    f = 1
                    break
            if f == 1:
                foo.addFinal(i)
        return foo

    def distTS(self):
        """Evaluate the two-sided distinguishability language for a DFA

        Returns:
            DFA:

        ..seealso:: Cezar Câmpeanu, Nelma Moreira, Rogério Reis:
           The distinguishability operation on regular languages. NCMA 2014: 85-100"""
        m = self.minimal()
        m.complete()
        trap = set([])
        for i in range(len(m.States)):
            f = 0
            for c in m.delta[i]:
                if m.delta[i][c] != i:
                    f = 1
                    break
            if f == 0:
                trap.add(i)
        if trap == set([]) or len(trap) == 2:
            return sigmaStarDFA(self.Sigma)
        else:
            d = DFA()
            d.setSigma(m.Sigma)
            ini = set(range(len(m.States))).difference(trap)
            d.setInitial(d.addState(ini))
            l_states = [ini]
            d.addFinal(0)
            index = 0
            while True:
                slist = l_states[index]
                si = d.stateIndex(slist, auto_create=True)
                for s in m.Sigma:
                    stl = set([m.evalSymbol(s1, s) for s1 in slist if s in m.delta[s1]])
                    if not stl:
                        continue
                    if stl not in l_states:
                        l_states.append(stl)
                        foo = d.addState(stl)
                        if stl != trap:
                            d.addFinal(foo)
                    else:
                        foo = d.stateIndex(stl, auto_create=True)
                    d.addTransition(si, s, foo)
                if index == len(l_states) - 1:
                    break
                else:
                    index += 1
        return d

    def distRMin(self):
        """Compute distRMin for DFA

        Returns:
            FL:

        ..seealso:: Cezar Câmpeanu, Nelma Moreira, Rogério Reis:
           The distinguishability operation on regular languages. NCMA 2014: 85-100"""

        def _epstr(d):
            if d == Epsilon:
                return ''
            return d

        def _strep(d):
            if d == '':
                return Epsilon
            return d

        from . import fl

        m = self.minimal()
        rev = m.reversal().toDFA()
        rev.complete()
        sz = len(rev.States)
        if sz == 1:
            return fl.FL()
        dpre_list = set()
        xlist = m._xA()
        for i in xlist:
            xlist[i] = _epstr(xlist[i])
        todo = [(s, s1) for s in range(sz) for s1 in range(s + 1, sz)]
        for (i, j) in todo:
            s1 = rev.States[i]
            s2 = rev.States[j]
            if s1 == DeadName:
                if s2 != DeadName:
                    md = min({xlist[k] for k in s2})
                    dpre_list.add(md)
            elif s2 == DeadName:
                md = min({xlist[k] for k in s1})
                dpre_list.add(md)
            else:
                d12 = s1 ^ s2
                md = min({xlist[k] for k in d12})
                dpre_list.add(md)
            todo.remove((i, j))

        return fl.FL({_strep(i) for i in dpre_list}, self.Sigma)

    def completeProduct(self, other):
        """Product structure

        Args:
            other (DFA): the other DFA
        Returns:
            DFA:"""
        n = SemiDFA()
        n.States = set([(x, y) for x in self.States for y in other.States])
        n.Sigma = copy(self.Sigma)
        for (x, y) in n.States:
            for s in n.Sigma:
                if (x, y) not in n.delta:
                    n.delta[(x, y)] = {}
                n.delta[(x, y)][s] = (self.delta[x][s], other.delta[y][s])
        return n

    def evalWordP(self, word, initial=None):
        """Verifies if the DFA recognises a given word

        :param word: word to be recognised
        :type word: list of symbols.
        :param int initial: starting state index
        :rtype: bool"""
        if initial is None:
            state = self.Initial
        else:
            state = initial
        for c in word:
            try:
                state = self.evalSymbol(state, c)
            except DFAstopped:
                return False
        if state in self.Final:
            return True
        else:
            return False

    def evalWord(self, wrd):
        """Evaluates a word

        :param Word wrd: word
        :returns: final state or None
        :rtype: int | None

        .. versionadded:: 1.3.3"""
        s = self.Initial
        for c in wrd:
            if c not in self.delta.get(s, {}):
                return None
            else:
                s = self.delta[s][c]
        return s

    def evalSymbol(self, init, sym):
        """Returns the  state reached from given state through a given symbol.

        :param int init: set of current states indexes
        :param str sym: symbol to be consumed
        :returns: reached state
        :rtype: int
        :raises DFAsymbolUnknown: if symbol not in alphabet
        :raises DFAstopped: if transition function is not defined for the given input"""
        if sym not in self.Sigma:
            raise DFAsymbolUnknown(sym)
        try:
            next_s = self.delta[init][sym]
        except KeyError:
            raise DFAstopped()
        except NameError:
            raise DFAstopped()
        return next_s

    def evalSymbolL(self, ls, sym):
        """Returns the set of states reached from a given set of states through a given symbol

        :param ls: set of states indexes
        :type ls: set of int
        :param str sym: symbol to be read
        :returns: set of reached states
        :rtype: set of int"""
        return set([self.evalSymbol(s, sym) for s in ls])

    def reverseTransitions(self, rev):
        """Evaluate reverse transition function.

        :param DFA rev: DFA in which the reverse function will be stored"""
        for s in self.delta:
            for a in self.delta[s]:
                rev.addTransition(self.delta[s][a], a, s)

    def initialComp(self):
        """Evaluates the connected component starting at the initial state.

        :returns: list of state indexes in the component
        :rtype: list of int"""
        lst = [self.Initial]
        i = 0
        while True:
            try:
                foo = list(self.delta[lst[i]].keys())
            except KeyError:
                foo = []
            for c in foo:
                s = self.delta[lst[i]][c]
                if s not in lst:
                    lst.append(s)
            i += 1
            if i >= len(lst):
                return lst

    def minimal(self, method="minimalMooreSq", complete=True):
        """Evaluates the equivalent minimal complete DFA

        :param method: method to use in the minimization
        :param bool complete: should the result be completed?
        :returns: equivalent minimal DFA
        :rtype: DFA"""
        if complete:
            foo = self.__getattribute__(method)()
            foo.completeMinimal()
            return foo
        else:
            return self.__getattribute__(method)()

    def minimalP(self, method="minimalMooreSq"):
        """Tests if the DFA is minimal

        :param method: the minimization algorithm to be used
        :rtype: bool

        ..note: if DFA non-complete test if  complete minimal has   one more state"""
        foo = self.minimal(method)

        if self.completeP():
            foo.completeMinimal()
        else:
            if foo.completeP():
                return len(foo) - 1 == len(self)
        return len(foo) == len(self)

    def minimalMoore(self):
        """Evaluates the equivalent minimal automata with Moore's algorithm

        .. seealso::
           John E. Hopcroft and Jeffrey D. Ullman, Introduction to Automata Theory, Languages, and Computation, AW,
           1979

        :returns: minimal complete DFA
        :rtype: DFA"""
        trash_idx = None  # just to satisfy the checker
        scc = set(self.initialComp())
        new = DFA()
        new.setSigma(self.Sigma)
        if (len(self.Final & scc) == 0) or (len(self.Final) == 0):
            s = new.addState()
            new.setInitial(s)
            return new
        equiv = set()
        for i in [x for x in range(len(self.States)) if x in scc]:
            for j in [x for x in range(i) if x in scc]:
                if ((i in self.Final and j in self.Final) or
                        (i not in self.Final and j not in self.Final)):
                    equiv.add((i, j))
        if not self.completeP():
            complete = False
            for i in [x for x in range(len(self.States))
                      if (x in scc and x not in self.Final)]:
                equiv.add((None, i))
        else:
            complete = True
        stable = False
        while not stable:
            stable = True
            nequiv = set()
            for (i, j) in equiv:
                for c in self.Sigma:
                    if i is None:
                        xi = None
                    else:
                        xi = self.delta.get(i, {}).get(c, None)
                    xj = self.delta.get(j, {}).get(c, None)
                    p = _sortWithNone(xi, xj)
                    if xi != xj and p not in equiv:
                        stable = False
                        nequiv.add((i, j))
                        break
            equiv -= nequiv
        n_stat_equiv = {}
        n_names = {}
        foo = list(equiv)
        foo.sort(key=lambda x: (x[1], x[0]))
        for (i, j) in foo:
            r = _deref(n_stat_equiv, j)
            n_stat_equiv[i] = r
            n_names[r] = n_names.get(r, [r]) + [i]
        removed = list(n_stat_equiv.keys())
        for i in [x for x in range(len(self.States)) if x not in removed]:
            new.addState(i)
        if complete:
            for i in [x for x in range(len(self.States)) if x not in removed]:
                for c in self.Sigma:
                    xi = new.stateIndex(i)
                    j = self.delta[i][c]
                    xj = new.stateIndex(n_stat_equiv.get(j, j))
                    new.addTransition(xi, c, xj)
        else:
            if None not in removed:
                trash_idx = new.addState()
                for c in self.Sigma:
                    new.addTransition(trash_idx, c, trash_idx)
            for i in [x for x in range(len(self.States)) if x not in removed]:
                xi = new.stateIndex(i)
                for c in self.Sigma:
                    # noinspection PyNoneFunctionAssignment
                    j = self.delta.get(i, {}).get(c, None)
                    if j is not None:
                        xj = new.stateIndex(n_stat_equiv.get(j, j))
                        new.addTransition(xi, c, xj)
                    else:
                        new.addTransition(xi, c, trash_idx)
        for i in self.Final:
            if i not in removed:
                xi = new.stateIndex(n_stat_equiv.get(i, i))
                new.addFinal(xi)
        new.setInitial(n_stat_equiv.get(self.Initial, self.Initial))
        new.renameStates([n_names.get(x, x) for x in range(len(new.States) - 1)] + ["Dead"])
        return new

    def minimalNCompleteP(self):
        """Tests if a non necessarely complete DFA is minimal, i.e., if the DFA is non-complete,
        if the minimal complete has only one more state.

        :returns: True if not minimal
        :rtype: bool

        .. attention::
            obsolete: use minimalP"""
        foo = self.minimal()
        foo.complete()
        if self.completeP():
            return len(foo) == len(self)
        else:
            return len(foo) == (len(self) + 1)

    def completeMinimal(self):
        """Completes a DFA assuming it is a minimal and avoiding de destruction of its minimality If the automaton is
        not complete, all the non-final states are checked to see if tey are not already a dead state. Only in the
        negative case a new (dead) state is added to the automaton.

        :rtype: DFA

        .. attention::
           The object is modified in place. If the alphabet is empty nothing is done"""
        if not self.Sigma:
            return
        self.trim()
        dead_s = None
        complete = True
        for s in range(len(self.States)):
            if s not in self.delta:
                complete = False
                if s not in self.Final:
                    dead_s = s
                    break
            else:
                foo = True
                for d in self.Sigma:
                    if d not in self.delta[s]:
                        complete = False
                        if s in self.Final:
                            foo = False
                    else:
                        if self.delta[s][d] != s or s in self.Final:
                            foo = False
                if foo:
                    dead_s = s
        if not complete:
            if dead_s is None:
                dead_s = self.addState("dead")
            for s in range(len(self.States)):
                for d in self.Sigma:
                    if s not in self.delta or d not in self.delta[s]:
                        self.addTransition(s, d, dead_s)
        return self

    def minimalMooreSq(self):
        """Evaluates the equivalent minimal complete DFA using Moore's (quadratic) algorithm

        .. seealso::
           John E. Hopcroft and Jeffrey D. Ullman, Introduction to Automata Theory, Languages, and Computation, AW,
           1979

        :returns: equivalent minimal DFA
        :rtype: DFA"""
        duped = self.dup()
        duped.complete()
        n_states = len(duped.States)
        duped._mooreMarked = {}
        duped._moorePairList = {}
        for p in range(n_states):
            for q in range(n_states):
                duped._moorePairList[(p, q)] = []
                duped._mooreMarked[(p, q)] = False
                if (p in duped.Final) ^ (q in duped.Final):
                    duped._mooreMarked[(p, q)] = True
        for p in range(n_states):
            for q in range(n_states):
                if not ((p in duped.Final) ^ (q in duped.Final)):
                    exists_marked = False
                    for a in duped.Sigma:
                        foo = (duped.delta[p][a], duped.delta[q][a])
                        if duped._mooreMarked[foo]:
                            exists_marked = True
                            break
                    if exists_marked:
                        duped._mooreMarked[(p, q)] = True
                        duped._mooreMarkList(p, q)
                    else:
                        for a in duped.Sigma:
                            if duped.delta[p][a] != duped.delta[q][a]:
                                pair = (duped.delta[p][a], duped.delta[q][a])
                                duped._moorePairList[pair].append((p, q))
        eqstates = duped._mooreEquivClasses()
        duped.joinStates(eqstates)
        return duped

    def minimalMooreSqP(self):
        """Tests if a DFA is minimal using the quadratic version of Moore's algorithm

        :rtype: bool"""
        foo = self.minimalMooreSq()
        foo.complete()
        return self.uniqueRepr() == foo.uniqueRepr()

    # noinspection PyUnresolvedReferences
    def _mooreMarkList(self, p, q):
        """ Marks pairs of states already known to be not non-equivalent

        :param p:
        :param q:"""
        for (p_, q_) in self._moorePairList[(p, q)]:
            if not self._mooreMarked[(p_, q_)]:
                self._mooreMarked[(p_, q_)] = True
                self._mooreMarkList(p_, q_)

    # noinspection PyUnresolvedReferences
    def _mooreEquivClasses(self):
        """Returns equivalence classes

        :returns: list of equivalence classes
        :rtype:list"""
        uf = UnionFind(auto_create=True)
        # eqstates = []
        for p in range(len(self.States)):
            for q in range(p + 1, len(self.States)):
                if not self._mooreMarked[(p, q)]:
                    a = uf.find(p)
                    b = uf.find(q)
                    uf.union(a, b)
        classes = {}
        for p in range(len(self.States)):
            lider = uf.find(p)
            if lider in classes:
                classes[lider].append(p)
            else:
                classes[lider] = [p]
        return list(classes.values())

    def _compute_delta_inv(self):
        """Adds a delta_inv feature. Used by minimalHopcroft."""
        self.delta_inv = {}
        for s in range(len(self.States)):
            self.delta_inv[s] = dict([(a, []) for a in self.Sigma])
        for s1, to in list(self.delta.items()):
            for a, s2 in list(to.items()):
                self.delta_inv[s2][a].append(s1)

    def _undelta(self, states, x):
        """Traverses Automata backwards

        :param states: destination
        :param x: symbol
        :return: list of states"""
        lst = set([])
        for s in states:
            lst.update(self.delta_inv[s][x])
        return lst

    # noinspection PyUnresolvedReferences
    def _split(self, b, c, a):
        """Split classes in Hopcroft algorithm

        :param b:
        :param c:
        :param a:
        :return:"""
        foo = frozenset(self._undelta(c, a))
        bar = frozenset(self.states - foo)
        return b & foo, b & bar

    def MyhillNerodePartition(self):
        """Myhill-Nerode partition, Moore's way

        .. versionadded:: 1.3.5

        .. attention::
            No state should be named with DeadName. This states is removed from the obtained partition.

        .. seealso::
           F.Bassino, J.David and C.Nicaud, On the Average Complexity of Moores's State Minimization Algorihm,
           Symposium on Theoretical Aspects of Computer Science"""
        if len(self.Final) == 0:
            return emptyDFA(self.Sigma)
        elif len(self.Final) == len(self) and self.completeP():
            return sigmaStarDFA(self.Sigma)
        else:
            aut = self.dup()
            aut.complete()
            p = dict()
            for s in range(len(aut.States)):
                if s in aut.Final:
                    p[s] = 1
                else:
                    p[s] = 0
            while True:
                p1 = aut._refinePartitionMN(p)
                if p == p1:
                    break
                else:
                    p = p1
            t = dict()
            for i in p:
                if aut.States[i] != DeadName:
                    t[p[i]] = t.get(p[i], set()) | {i}
            return t

    def _refinePartitionMN(self, pi):
        """refine partition one more step: used by minimalMooreN

        :param dict pi: dict representing partition
        :rtype: dict
        :return: new partition"""
        p = []
        for s in range(len(self)):
            r = [pi[s]]
            for c in self.Sigma:
                r.append(pi[self.delta[s][c]])
            p.append((s, r))
        p.sort(key=lambda x: x[1])
        pi1 = dict()
        i = 0
        s0, l0 = p.pop(0)
        pi1[s0] = i
        for s1, l1 in p:
            if l1 != l0:
                i += 1
                s0, l0 = s1, l1
            pi1[s1] = i
        return pi1

    def minimalHopcroft(self):
        """Evaluates the equivalent minimal complete DFA using Hopcroft algorithm

        :returns: equivalent minimal DFA
        :rtype: DFA

        .. seealso::
           John Hopcroft,An n log{n} algorithm for minimizing states in a  finite automaton.The Theory of Machines
           and Computations.AP. 1971"""
        duped = self.dup()
        duped.complete()
        duped._compute_delta_inv()
        duped.states = frozenset(range(len(duped.States)))
        final = frozenset(duped.Final)
        not_final = duped.states - final
        l = set([])
        if len(final) < len(not_final):
            p = {not_final, final}
            l.add(final)
        else:
            p = {final, not_final}
            l.add(not_final)
        while len(l):
            c = l.pop()
            for a in duped.Sigma:
                foo = copy(p)
                for b in foo:
                    (b1, b2) = duped._split(b, c, a)
                    p.remove(b)
                    if b1:
                        p.add(b1)
                    if b2:
                        p.add(b2)
                    if len(b1) < len(b2):
                        if b1:
                            l.add(b1)
                    else:
                        if b2:
                            l.add(b2)
        eqstates = []
        for i in p:
            if len(i) != 1:
                eqstates.append(list(i))
        duped.joinStates(eqstates)
        return duped

    def minimalHopcroftP(self):
        """Tests if a DFA is minimal

        :rtype: bool"""
        foo = self.minimalHopcroft()
        foo.complete()
        return self.uniqueRepr() == foo.uniqueRepr()

    def minimalNotEquivP(self):
        """Tests if the DFA is minimal by computing the set of distinguishable (not equivalent) pairs of states

        :rtype: bool"""
        all_pairs = set()
        for i in self.States:
            for j in range(i + 1, len(self.States)):
                all_pairs.add((i, j))
        not_final = set(self.States) - self.Final
        neq = set()
        for i in not_final:
            for j in self.Final:
                pair = _normalizePair(i, j)
                neq.add(pair)
        source = neq.copy()
        self._compute_delta_inv()
        while source:
            (p, q) = source.pop()
            for a in self.Sigma:
                p_ = self.delta_inv[p][a]
                q_ = self.delta_inv[q][a]
                for x in p_:
                    for y in q_:
                        pair = _normalizePair(x, y)
                        if pair not in neq:
                            neq.add(pair)
                            source.add(pair)
        equiv = all_pairs - neq
        return not equiv

    def _minimalHKP(self):
        """Tests the DFA's minimality using Hopcroft and Karp's state equivalence algorithm

        :returns: bool

        .. seealso::
           J. E. Hopcroft and r. M. Karp.A Linear Algorithm for Testing Equivalence of Finite Automata.TR 71--114. U.
           California. 1971

        .. attention::
           The automaton must be complete."""
        pairs = set()
        for i in range(len(self.States)):
            for j in range(i + 1, len(self.States)):
                pairs.add((i, j))
        while pairs:
            equiv = True
            (p0, q0) = pairs.pop()
            sets = UnionFind(auto_create=True)
            sets.union(p0, q0)
            stack = [(p0, q0)]
            while stack:
                (p, q) = stack.pop()
                if (p in self.Final) ^ (q in self.Final):
                    equiv = False
                    break
                for a in self.Sigma:
                    r1 = sets.find(self.delta[p][a])
                    r2 = sets.find(self.delta[q][a])
                    if r1 != r2:
                        sets.union(r1, r2)
                        stack.append((r1, r2))
            if equiv:
                return False
        return True

    def minimalIncremental(self, minimal_test=False):
        """Minimizes the DFA with an incremental method using the Union-Find algorithm and memoized non-equivalence
        intermediate results

        :param bool minimal_test: starts by verifying that the automaton is not minimal?
        :returns: equivalent minimal DFA
        :rtype: DFA

        .. seealso::
           M. Almeida and N. Moreira and and r. Reis.Incremental DFA minimisation. CIAA 2010. LNCS 6482.
           pp 39-48. 2010"""
        duped = self.dup()
        duped.complete()
        duped.minimalIncr_neq = set()
        n_states = len(duped.States)
        for p in duped.Final:
            for q in range(n_states):
                if q not in duped.Final:
                    duped.minimalIncr_neq.add(_normalizePair(p, q))
        duped.minimalIncr_uf = UnionFind(auto_create=True)
        for p in range(n_states):
            for q in range(p + 1, n_states):
                if (p, q) in duped.minimalIncr_neq:
                    continue
                if duped.minimalIncr_uf.find(p) == duped.minimalIncr_uf.find(q):
                    continue
                duped.equiv = set()
                duped.path = set()
                if duped._minimalIncrCheckEquiv(p, q):
                    # when we are only interested in testing
                    # minimality, return None to signal a pair of
                    # equivalent states
                    if minimal_test:
                        return None
                    for (x, y) in duped.equiv:
                        duped.minimalIncr_uf.union(x, y)
                else:
                    duped.minimalIncr_neq |= duped.path
        classes = {}
        for p in range(n_states):
            lider = duped.minimalIncr_uf.find(p)
            if lider in classes:
                classes[lider].append(p)
            else:
                classes[lider] = [p]
        duped.joinStates(list(classes.values()))
        return duped

    # noinspection PyUnresolvedReferences
    def _minimalIncrCheckEquiv(self, p, q, rec_level=1):
        # p == q is a useless test; union-find offers this for free
        # because p == q => find(p) == find(q) and the recursive call
        # only happens when find(p) != find(q)

        # if p_ in self.Final ^ q_ in self.Final => (p_,
        # q_) are already on self.minimalIncr_neq
        # (initialization)
        if (p, q) in self.minimalIncr_neq:
            return False
            # cycle detected; the states must be equivalent
        if (p, q) in self.path:
            return True
        self.path.add((p, q))
        for a in self.Sigma:
            (p_, q_) = _normalizePair(self.minimalIncr_uf.find(self.delta[p][a]),
                                      self.minimalIncr_uf.find(self.delta[q][a]))
            if p_ != q_ and ((p_, q_) not in self.equiv):
                self.equiv.add((p_, q_))
                if not self._minimalIncrCheckEquiv(p_, q_, rec_level + 1):
                    return False
                else:
                    # if the states are equivalent, the 'path' doesn't
                    # really interest; by removing the states here, we
                    # can make 'path' a global variable and avoid any
                    # copy() operations. removing the last inserted
                    # item is necessary when the states are equivalent
                    # because the next recursive call (next symbol)
                    # needs an "empty path", ie, the states reached by
                    # the previous symbol cannot be considered
                    self.path.discard((p_, q_))
        self.equiv.add((p, q))
        return True

    def minimalIncrementalP(self):
        """Tests if a DFA is minimal

        :rtype: bool"""
        foo = self.minimalIncremental(minimal_test=True)
        if foo is None:
            return False
        return True

    def minimalWatson(self, test_only=False):
        """Evaluates the equivalent minimal complete DFA using Waton's incremental algorithm

        :param bool test_only: is it only to test minimality
        :returns: equivalent minimal DFA
        :rtype: DFA

        :raises  DFAnotComplete: if automaton is not complete

        ..attention::
          automaton must be complete"""
        duped = self.dup()
        duped.complete()
        duped.Equiv = UnionFind(auto_create=True)
        duped.Dist = set()
        nstates = len(self.States)
        max_depth = max(0, nstates - 2)
        for p in range(nstates):
            for q in range(p + 1, nstates):
                if duped.Equiv.find(p) == duped.Equiv.find(q):
                    continue
                if (p, q) in duped.Dist:
                    continue
                duped.minimalWatson_stack = set([])
                if duped._watson_equivP(p, q, max_depth):
                    # when we are only interested in testing
                    # minimality, return None to signal a pair of
                    # equivalent states
                    if test_only:
                        return None
                    duped.Equiv.union(p, q)
                else:
                    duped.Dist.add((p, q))
        classes = {}
        for p in range(len(duped.States)):
            lider = duped.Equiv.find(p)
            if lider in classes:
                classes[lider].append(p)
            else:
                classes[lider] = [p]
        duped.joinStates(list(classes.values()))
        return duped

    # noinspection PyUnresolvedReferences
    def _watson_equivP(self, p, q, k):
        if not k:
            eq = not ((p in self.Final) ^ (q in self.Final))
        elif (p, q) in self.minimalWatson_stack:
            eq = True
        else:
            eq = not ((p in self.Final) ^ (q in self.Final))
            self.minimalWatson_stack.add((p, q))
            for a in self.Sigma:
                if not eq:
                    return eq
                try:
                    eq = eq and self._watson_equivP(self.delta[p][a], self.delta[q][a], k - 1)
                except KeyError:
                    raise DFAnotComplete
            self.minimalWatson_stack.remove((p, q))
        return eq

    def minimalWatsonP(self):
        """Tests if a DFA is minimal using Watson's incremental algorithm

        :rtype: bool"""
        foo = self.minimalWatson(test_only=True)
        if foo is None:
            return False
        return True

    def markNonEquivalent(self, s1, s2, data):
        """Mark states with indexes s1 and s2 in given map as non-equivalent states. If any back-effects exist,
        apply them.

        :param int s1: one state's index
        :param int s2: the other state's index
        :param data: the matrix relating s1 and s2"""
        try:
            del (data[s1][0][data[s1][0].index(s2)])
        except ValueError:
            pass
        try:
            back_effects = data[s1][1][s2]
        except KeyError:
            back_effects = []
        for (sb1, sb2) in back_effects:
            del (data[s1][1][s2][data[s1][1][s2].index((sb1, sb2))])
            self.markNonEquivalent(sb1, sb2, data)

    def print_data(self, data):
        """Prints table of compatibility (in the context of the minimalization algorithm).

        :param data: data to print"""
        for s in range(len(self.States)):
            for s1 in range(0, s):
                if s1 in data[s]:
                    print("_ ", end=' ')
                else:
                    print("X ", end=' ')
            print(s)

    def joinStates(self, lst):
        """Merge a list of states.

        :param lst: set of equivalent states
        :type lst: iterable of state indexes."""
        lst.sort()
        subst = {}
        for sl in lst:
            sl.sort()
            if self.Initial in sl[1:]:
                self.setInitial(sl[0])
            for s in sl[1:]:
                subst[s] = sl[0]
        for s in self.delta:
            for c in self.delta[s]:
                if self.delta[s][c] in subst:
                    self.delta[s][c] = subst[self.delta[s][c]]
        for sl in lst:
            for s in sl[1:]:
                try:
                    foo = list(self.delta[s].keys())
                    for c in foo:
                        if c not in self.delta[subst[s]]:
                            if self.delta[s][c] in subst:
                                self.delta[subst[s]][c] = subst[self.delta[s][c]]
                            else:
                                self.delta[subst[s]][c] = self.delta[s][c]
                    del (self.delta[s])
                except KeyError:
                    pass
                if s in self.Final:
                    self.Final.remove(s)
        self.trim()

    def HKeqP(self, other, strict=True):
        """Tests the DFA's equivalence using Hopcroft and Karp's state equivalence algorithm

        :param other:
        :param strict:
        :returns: bool

        .. seealso::
           J. E. Hopcroft and r. M. Karp.A Linear Algorithm for Testing Equivalence of Finite Automata.TR 71--114. U.
           California. 1971

        .. attention::
           The automaton must be complete."""
        if strict and self.Sigma != other.Sigma:
            return False
        if not isinstance(other, DFA):
            raise FAdoError
        n = len(self.States)
        if n == 0 or len(other.States) == 0:
            raise NFAEmpty
        i1 = self.Initial
        i2 = self.Initial + n
        s = UnionFind(auto_create=True)
        s.union(i1, i2)
        stack = [(i1, i2)]
        while stack:
            (p, q) = stack.pop()
            # test if p is in self
            on_other = False
            if p >= n:
                on_other = True
            if on_other:
                if other.finalP(p - n) ^ other.finalP(q - n):
                    return False
            elif self.finalP(p) != other.finalP(q - n):
                return False
            for sigma in self.Sigma:
                if on_other:
                    p1 = s.find(n + other.evalSymbol(p - n, sigma))
                else:
                    p1 = s.find(self.evalSymbol(p, sigma))
                q1 = s.find(n + other.evalSymbol(q - n, sigma))
                if p1 != q1:
                    s.union(p1, q1)
                    stack.append((p1, q1))
        return True

    def compat(self, s1, s2, data):
        """Tests compatibility between two states.

        :param data:
        :param int s1: state index
        :param int s2: state index
        :rtype: bool"""
        if s1 == s2:
            return False
        if s1 not in data[s2][0] or s2 not in data[s1][0]:
            return False
        if s1 in self.Final != s2 in self.Final:
            del (data[s1][data[s1].index(s2)])
            del (data[s2][data[s2].index(s1)])
            return True
        for s in self.Sigma:
            next1 = self.delta[s1][s]
            next2 = self.delta[s2][s]
            if (next1 not in data[next2]) or (next2 not in data[next1]):
                del (data[s1][data[s1].index(s2)])
                del (data[s2][data[s2].index(s1)])
                return True
        return False

    def dup(self):
        """Duplicate the basic structure into a new DFA. Basically a copy.deep.

        :rtype: DFA"""
        new = DFA()
        new.setSigma(self.Sigma)
        new.States = self.States[:]
        new.Initial = self.Initial
        new.Final = self.Final.copy()
        for s in list(self.delta.keys()):
            new.delta[s] = {}
            for c in self.delta[s]:
                new.delta[s][c] = self.delta[s][c]
        return new

    def equal(self, other):
        """Verify if the two automata are equivalent. Both are verified to be minimum and complete,
        and then one is matched against the other... Doesn't destroy either dfa...

        :param DFA other: the other DFA
        :rtype: bool"""
        return self.__eq__(other)

    def __eq__(self, other):
        """Tests equivalence of DFAs

        :param DFA other: the other DFA
        :return: bool"""
        dfa1, dfa2 = self.dup(), other.dup()
        dfa1 = dfa1.minimal()
        dfa2 = dfa2.minimal()
        dfa1.completeMinimal()
        dfa2.completeMinimal()
        # if ((len(dfa1.States) != len(dfa2.States)) or (len(dfa1.Final) != len(dfa2.Final)) or
        #        (dfa1._uniqueStr() != dfa2._uniqueStr())):
        if ((dfa1.Sigma != dfa2.Sigma and (len(dfa1.Sigma) != 0 or len(dfa2.Sigma) != 0)) or
                (len(dfa1.States) != len(dfa2.States)) or (len(dfa1.Final) != len(dfa2.Final)) or
                (dfa1._uniqueStr() != dfa2._uniqueStr())):
            return False
        else:
            return True

    def _lstTransitions(self):
        l = []
        for x in self.delta:
            for k in self.delta[x]:
                l.append((self.States[x], k, self.States[self.delta[x][k]]))
        return l

    def _lstInitial(self):
        """
        :return:
        :raise: DFAnoInitial if no initial state is defined """
        if self.Initial is None:
            raise DFAnoInitial()
        else:
            return self.States[self.Initial]

    def _s_lstInitial(self):
        return str(self._lstInitial())

    def notequal(self, other):
        """ Test non  equivalence of two DFAs

        :param DFA other: the other DFA
        :rtype: bool"""
        return self.__ne__(other)

    def __ne__(self, other):
        """ Tests non-equivalence of two DFAs

        :param DFA other: the other DFA
        :rtype: bool"""
        return not self == other

    def hyperMinimal(self, strict=False):
        """ Hyperminization of a minimal DFA

        :param bool strict: if strict=True it first minimizes the DFA
        :returns: an hyperminimal DFA
        :rtype: DFA

        .. seealso::
           M. Holzer and A. Maletti, An nlogn Algorithm for Hyper-Minimizing a (Minimized) Deterministic Automata,
           TCS 411(38-39): 3404-3413 (2010)

        .. note:: if strict=False minimality is assumed"""
        if strict:
            m = self.minimal()
        else:
            m = self.dup()
        comp, center, mark = m.computeKernel()
        ker = set([m.States[s] for s in mark])
        m._mergeStatesKernel(ker, m.aEquiv())
        return m

    def _mergeStatesKernel(self, ker, aequiv):
        """ Merge states of almost equivalent partition. Used by hyperMinimal.

        :param ker:
        :param aequiv: partition of almost equivalence"""
        for b in aequiv:
            try:
                q = (aequiv[b] & ker).pop()
            except KeyError:
                q = aequiv[b].pop()
            for p in aequiv[b] - ker:
                self.mergeStates(self.stateIndex(p), self.stateIndex(q))

    def computeKernel(self):
        """ The Kernel of a ICDFA is the set of states that accept  a non-finite language.

        :returns: triple (comp, center , mark) where comp are the strongly connected components,
                  center the set of center states and mark the kernel states
        :rtype: tuple

        .. note:
           DFA must be initially connected

        .. seealso:
           Holzer and A. Maletti, An nlogn Algorithm for Hyper-Minimizing a (Minimized) Deterministic Automata,
           TCS 411(38-39): 3404-3413 (2010)"""

        def _SCC(t):
            ind[t] = self.i
            low[t] = self.i
            self.i += 1
            stack.append(t)
            for b in self.Sigma:
                # noinspection PyNoneFunctionAssignment
                t1 = self.delta.get(t, {}).get(b, None)
                if t1 is not None and t1 not in ind:
                    _SCC(t1)
                    low[t] = min([low[t], low[t1]])
                else:
                    if t1 in stack:
                        low[t] = min([low[t], ind[t1]])
            if low[t] == ind[t]:
                comp[t] = [t]
                p = stack.pop()
                while p != t:
                    comp[t].append(p)
                    p = stack.pop()

        def _DFS(t):
            mark[t] = 1
            for a1 in self.Sigma:
                # noinspection PyNoneFunctionAssignment
                t1 = self.delta.get(t, {}).get(a1, None)
                if t1 is not None and t1 not in mark:
                    _DFS(t1)

        ind = {}
        low = {}
        stack = []
        self.i = 0
        comp = {}
        center = set([s for s in self.delta for a in self.delta[s] if self.delta[s][a] == s])
        _SCC(self.Initial)
        for s in comp:
            if len(comp[s]) > 1:
                center.update(comp[s])
        mark = {}
        for s in center:
            mark[s] = 1
        for s in center:
            for a in self.Sigma:
                # noinspection PyNoneFunctionAssignment
                s1 = self.delta.get(s, {}).get(a, None)
                if s1 is not None and s1 not in mark:
                    _DFS(s1)
        del self.i
        return comp, center, mark

    def aEquiv(self):
        """ Computes almost equivalence, used by hyperMinimial

        Returns:
            dict: partition of states

        .. note::
           may be optimized to avoid dupped"""
        pi = {}
        dupped = self.dup()
        for q in dupped.States:
            pi[q] = {q}
        h = {}
        i = set(dupped.States)
        p1 = set(dupped.States)
        dupped._compute_delta_inv()
        while i != set([]):
            q = i.pop()
            succ = tuple([dupped.States[dupped.delta[dupped.stateIndex(q)][a]] for a in dupped.Sigma
                          if dupped.stateIndex(q) in dupped.delta and a in dupped.delta[dupped.stateIndex(q)]])
            if succ in h:
                p = h[succ]
                if len(pi[p]) >= len(pi[q]):
                    p, q = q, p
                p1.remove(p)
                i.update([r for r in p1 for a in dupped.Sigma
                          if dupped.stateIndex(r) in dupped.delta_inv[dupped.stateIndex(p)][a]])
                dupped.mergeStates(dupped.stateIndex(p), dupped.stateIndex(q))
                dupped._compute_delta_inv()
                pi[q] = pi[q].union(pi[p])
                del (pi[p])
            h[succ] = q
        return pi

    def mergeStates(self, f, t):
        """Merge the first given state into the second. If the first state is an initial state the second becomes the
        initial state.

        :param int f: index of state to be absorbed
        :param int t: index of remaining state

        .. attention::
           It is up to the caller to remove the disconnected state. This can be achieved with ```trim()``."""
        if f is not t:
            for state, to in list(self.delta.items()):
                for a, s in list(to.items()):
                    if f == s:
                        self.delta[state][a] = t
            if self.initialP(f):
                self.setInitial(t)
            self.deleteStates([f])

    def toADFA(self):
        """ Try to convert DFA to ADFA

        :return: the same automaton as a ADFA
        :rtype: ADFA
        :raises notAcyclic: if this is not an acyclic DFA

        .. versionadded:: 1.2

        .. versionchanged:: 1.2.1"""
        from . import fl

        foo = self.dup().trim()
        if not foo.acyclicP():
            raise notAcyclic()
        else:
            new = fl.ADFA()
            new.Initial = foo.Initial
            new.States = deepcopy(foo.States)
            new.Sigma = deepcopy(foo.Sigma)
            new.Final = deepcopy(foo.Final)
            new.delta = deepcopy(foo.delta)
            return new

    def stronglyConnectedComponents(self):
        """Dummy method that uses the NFA conterpart

         .. versionadded:: 1.3.3

         :rtype: list"""
        return self.toNFA().stronglyConnectedComponents()

    def orderedStrConnComponents(self):
        """Topological ordered list of strong components

        .. versionadded:: 1.3.3

        :rtype: list"""

        def _topOrder(i1, j1):
            if l1[str(j1)] in l[l1[str(i1)]]:
                return -1
            elif l1[str(i1)] in l[l1[str(j1)]]:
                return 1
            else:
                return 0

        comp = self.stronglyConnectedComponents()
        l = [set([]) for _ in comp]
        l1 = dict()
        c = dict()
        for i, x in enumerate(comp):
            l1[str(x)] = i
            for s in x:
                c[s] = i
        for s in range(len(self.States)):
            for ch in self.delta.get(s, {}):
                d = c[self.delta[s][ch]]
                if d != c[s]:
                    l[c[s]].add(d)
        comp.sort(key=cmp_to_key(_topOrder))
        return comp

    def reversibleP(self):
        """Test if an automaton is reversible

        :rtype: bool"""
        self._compute_delta_inv()
        for s in self.delta_inv:
            for c in self.delta_inv[s]:
                if len(self.delta_inv[s][c]) > 1:
                    return False
        return True

    def makeReversible(self):
        """Make a DFA reversible (if possible)

        .. seealso:: M.Holzer, s. Jakobi, M. Kutrib 'Minimal Reversible Deterministic Finite Automata'

        :rtype: DFA"""
        not_done = True
        aut = self.dup()
        while not_done:
            not_done = False
            comp = aut.orderedStrConnComponents()
            aut._compute_delta_inv()
            for cp in comp:
                for s in cp:
                    l, ch = aut._notReversiblePoint(s)
                    if ch is not None:
                        ls = aut.delta_inv[s][ch][1:]
                        for i in range(l - 1):
                            sn = aut._dupSubAut(cp, s)
                            sto = ls.pop()
                            aut.delTransition(sto, ch, s)
                            aut.addTransition(sto, ch, sn)
                        not_done = True
                        break
                if not_done:
                    break
        return aut

    def _dupSubAut(self, ss, s):
        """Duplicates a set of states and identifies the copy of a given state

        :param int s: state to indentify
        :param lst ss: list of states
        :rtype: int"""
        nm = dict()
        for i in ss:
            nm[i] = self.addState()
        for i in ss:
            if i in self.Final:
                self.addFinal(nm[i])
            for c in self.delta.get(i, {}):
                j = self.delta[i][c]
                if j in ss:
                    self.addTransition(nm[i], c, nm[j])
                else:
                    self.addTransition(nm[i], c, j)
        return nm[s]

    def _possibleToMakeReversible(self, st):
        """Test id a state is a forbidden state for reversability

        :param int st: state
        :rtype: bool"""
        if self.delta_inv is None:
            self._compute_delta_inv()
        for c in self.delta_inv.get(st, {}):
            l = self.delta_inv[st][c]
            if len(l) > 1:
                todo = [st]
                done = set([])
                while todo:
                    s = todo.pop()
                    done.add(s)
                    for d in self.delta.get(s, {}):
                        j = self.delta[s][d]
                        if j in l:
                            return False
                        if j not in done:
                            todo.append(j)
                return True
        return True

    def possibleToReverse(self):
        """Tests if language is reversible

        .. versionadded:: 1.3.3"""
        for i in range(len(self.States)):
            if not self._possibleToMakeReversible(i):
                return False
        return True

    def _notReversiblePoint(self, st):
        """Checks if the state is reversible

        :param int st: state  index
        :rtype: tuple"""
        for c in self.delta_inv.get(st, {}):
            l = len(self.delta_inv[st][c])
            if l > 1:
                return l, c
        return None, None

    def toDFA(self):
        """Dummy function. It is already a DFA

        :returns: a self deep copy
        :rtype: DFA"""
        return self.dup()

    def _uniqueStr(self):
        """ Returns a canonical representation of the automaton.

        :returns: canonical representation of the skeleton and the list of final states, in a pair
        :rtype: pair of lists of int

        .. note:
           Automata is supposed to be a icdfa. It, now, should cope with non-complete automata"""
        s_sigma = list(self.Sigma)
        s_sigma.sort(key=lambda x: x.__repr__())
        tf, tr = {}, {}
        string = []
        i, j = 0, 0
        tf[self.Initial], tr[0] = 0, self.Initial
        while i <= j:
            lst = []
            for c in s_sigma:
                foo = self.delta.get(tr[i], {}).get(c, None)
                # foo = self.delta[tr[i]][c]
                if foo is None:
                    lst.append(-1)
                else:
                    if foo not in tf:
                        j += 1
                        tf[foo], tr[j] = j, foo
                    lst.append(tf[foo])
            string.append(lst)
            i += 1
        lst = []
        for s in self.Final:
            lst.append(tf[s])
        lst.sort(key=lambda x: x.__repr__())
        return string, lst

    def uniqueRepr(self):
        """Normalise unique string for the string icdfa's representation.
        .. seealso::
        TCS 387(2):93-102, 2007 https://www.dcc.fc.up.pt/~nam/publica/tcsamr06.pdf

        :returns: normalised representation
        :rtype: list

        :raises DFAnotComplete: if DFA is not complete"""
        try:
            (a, b) = self._uniqueStr()
            n = len(a)
            finals = [0] * n
            for i in b:
                finals[i] = 1
            return [j for i in a for j in i], finals, n, len(self.Sigma)
        except KeyError:
            raise DFAnotComplete

    def __invert__(self):
        """ Returns a DFA that recognises the complementary language:  ~X. Basically change all non-final states to
        final and vice-versa. After ensuring that it is complete.

        :rtype: DFA"""
        fa = self.dup()
        fa.eliminateDeadName()
        fa.complete()
        fa.setFinal([])
        for s in range(len(fa.States)):
            if s not in self.Final:
                fa.addFinal(s)
        return fa

    def __or__(self, other, complete=True, trim=True):
        """ Union of two automata

        :param DFA other: the other automaton
        :param bool complete: should the result be complete (default True)
        :param bool trim: should the result be trom (default True)
        :rtype: DFA

        .. versionchanged:: 1.3.4"""
        if type(other) != type(self):
            raise FAdoGeneralError("Incompatible objects")
        fa = self.product(other)
        sz1, sz2 = len(self.States), len(other.States)
        for s1 in self.Final:
            for s2 in range(sz2 + 1):
                fa.addFinal(s1 * (sz2 + 1) + s2)
        for s2 in other.Final:
            for s1 in range(sz1 + 1):
                fa.addFinal(s1 * (sz2 + 1) + s2)
        if trim:
            fa.trim()
        if complete:
            fa.complete()
        return fa._namesToString()

    def __sub__(self, other):
        return self & (~other)

    def simDiff(self, other):
        """Symetrical difference

        :param other:
        :return:"""
        # noinspection PyUnresolvedReferences
        return (self - other) | (other - self)

    def andSlow(self, other, complete=True):
        if not isinstance(other, DFA):
            raise FAdoGeneralError("Incompatible objects")
        fa = self.productSlow(other, complete)
        for i in range(len(fa.States)):
            (i1, i2) = fa.States[i]
            if i1 in self.Final and i2 in other.Final:
                fa.addFinal(i)
        return fa._namesToString()

    def __and__(self, other, complete=False, trim=True):
        """ Intersection automaton of two automata

        :param DFA other: the other automaton
        :param bool complete: should the result be complete (defaut False)
        :param bool trim: should the result be trim (default True)
        :rtype: DFA

        .. note:: This version does not use the product method

        .. versionchanged:: 1.3.4"""
        if not isinstance(other, DFA):
            raise FAdoGeneralError("Incompatible objects")
        new = DFA()
        n_sigma = self.Sigma.union(other.Sigma)
        new.setSigma(n_sigma)
        sz1, sz2 = len(self.States), len(other.States)
        for _ in range(sz1 * sz2):
            new.addState()
        new.setInitial(self.Initial * sz2 + other.Initial)
        if not complete:
            for s1 in range(sz1):
                for s2 in range(sz2):
                    sti = s1 * sz2 + s2
                    for c in self.delta.get(s1, {}):
                        if c in other.delta.get(s2, {}):
                            new.addTransition(sti, c, self.delta[s1][c] * sz2 + other.delta[s2][c])
        else:
            last = new.addState()
            for s1 in range(sz1):
                for s2 in range(sz2):
                    sti = s1 * sz2 + s2
                    for c in n_sigma:
                        if c in other.delta.get(s2, {}) and c in self.delta.get(s1, {}):
                            new.addTransition(sti, c, self.delta[s1][c] * sz2 + other.delta[s2][c])
                        else:
                            new.addTransition(sti, c, last)
            for c in n_sigma:
                new.addTransition(last, c, last)
        for s1 in self.Final:
            for s2 in other.Final:
                new.addFinal(s1 * sz2 + s2)
        if trim:
            new.trim()
        return new

    def productSlow(self, other, complete=True):
        """ Returns a DFA resulting of the simultaneous execution of two DFA. No final states set.

        .. note:: this is a slow implementation for those that need meaningfull state names

        .. versionadded:: 1.3.3

        :param other: the other DFA
        :param bool complete: evaluate product as a complete DFA
        :rtype: DFA"""
        n_sigma = self.Sigma.union(other.Sigma)
        fa1, fa2 = self.dup(), other.dup()
        fa1.setSigma(n_sigma)
        fa2.setSigma(n_sigma)
        fa1.complete()
        fa2.complete()
        fa = DFA()
        fa.setSigma(n_sigma)
        s = fa.addState((fa1.Initial, fa2.Initial))
        fa.setInitial(s)
        i = 0
        while True:
            i1, i2 = fa.States[i]
            for c in fa.Sigma:
                new = (fa1.delta[i1][c], fa2.delta[i2][c])
                foo = fa.stateIndex(new, True)
                fa.addTransition(i, c, foo)
            i += 1
            if i == len(fa.States):
                break
        if not complete:
            d1 = fa1.stateIndex(DeadName)
            d2 = fa2.stateIndex(DeadName)
            try:
                d = fa.stateIndex((d1, d2))
            except DFAstateUnknown:
                pass
            else:
                fa.deleteState(d)
        return fa

    def product(self, other):
        """ Returns a DFA resulting of the simultaneous execution of two DFA. No final states set.

        .. note:: this is a fast version of the method. The resulting state names are not meaningfull.

        .. versionchanged: 1.3.3

        :param other: the other DFA
        :rtype: DFA"""
        n_sigma = self.Sigma.union(other.Sigma)
        sz1, sz2 = len(self.States), len(other.States)
        sz2c = sz2 + 1
        new = DFA()
        for _ in range((sz1 + 1) * (sz2 + 1)):
            new.addState()
        new.setInitial(self.Initial * sz2c + other.Initial)
        _last = (sz1 + 1) * (sz2 + 1) - 1
        for s1 in range(sz1):
            for s2 in range(sz2):
                sti = s1 * sz2c + s2
                for c in n_sigma:
                    if c in self.delta.get(s1, {}):
                        if c in other.delta.get(s2, {}):
                            new.addTransition(sti, c, self.delta[s1][c] * sz2c + other.delta[s2][c])
                        else:
                            new.addTransition(sti, c, self.delta[s1][c] * sz2c + sz2)
                    else:
                        if c in other.delta.get(s2, {}):
                            new.addTransition(sti, c, sz1 * sz2c + other.delta[s2][c])
        for s1 in range(sz1):
            sti = s1 * sz2c + sz2
            for c in n_sigma:
                if c in self.delta.get(s1, {}):
                    new.addTransition(sti, c, self.delta.get(s1, {}).get(c, sz2) * sz2c + sz2)
        for s2 in range(sz2):
            sti = sz1 * sz2c + s2
            for c in n_sigma:
                if c in other.delta.get(s2, {}):
                    new.addTransition(sti, c, sz1 * sz2c + other.delta.get(s2, {}).get(c, sz2))
        return new

    def witness(self):
        """Witness of non emptyness

        :return: word
        :rtype: str"""
        done = set()
        not_done = set()
        pref = dict()
        si = self.Initial
        pref[si] = Epsilon
        not_done.add(si)
        while not_done:
            si = not_done.pop()
            done.add(si)
            if si in self.Final:
                return pref[si]
            for syi in self.delta.get(si, []):
                so = self.delta[si][syi]
                if so in done or so in not_done:
                    continue
                pref[so] = sConcat(pref[si], syi)
                not_done.add(so)
        return None

    def concat(self, fa2, strict=False):
        """Concatenation of two DFAs. If DFAs are not complete, they are completed.

        :param bool strict: should alphabets be checked?
        :param DFA fa2: the second DFA
        :returns: the result of the concatenation
        :rtype: DFA
        :raises DFAdifferentSigma: if alphabet are not equal"""
        if strict and self.Sigma != fa2.Sigma:
            raise DFAdifferentSigma
        n_sigma = self.Sigma.union(fa2.Sigma)
        d1, d2 = self.dup(), fa2.dup()
        d1.setSigma(n_sigma)
        d2.setSigma(n_sigma)
        d1.complete()
        d2.complete()
        if len(d1.States) == 0 or len(d1.Final) == 0:
            return d1
        if len(d2.States) <= 1:
            if not len(d2.Final):
                return d2
            else:
                new = DFA()
                new.setSigma(d1.Sigma)
                new.States = d1.States[:]
                new.Initial = d1.Initial
                new.Final = d1.Final.copy()
                for s in d1.delta:
                    new.delta[s] = {}
                    if new.finalP(s):
                        for c in d1.delta[s]:
                            new.delta[s][c] = s
                    else:
                        for c in d1.delta[s]:
                            new.delta[s][c] = d1.delta[s][c]
                return new
        c = DFA()
        c.setSigma(d1.Sigma)
        l_states = []
        i = (d1.Initial, set([]))
        l_states.append(i)
        j = c.addState(i)
        c.setInitial(j)
        if d1.finalP(d1.Initial):
            i[1].add(d2.Initial)
            if d2.finalP(d2.Initial):
                c.addFinal(j)
        while True:
            stu = l_states[j]
            s = c.stateIndex(stu)
            for sym in d1.Sigma:
                stn = (d1.evalSymbol(stu[0], sym), d2.evalSymbolL(stu[1], sym))
                if d1.finalP(stn[0]):
                    stn[1].add(d2.Initial)
                if stn not in l_states:
                    l_states.append(stn)
                    new = c.addState(stn)
                    if d2.Final & stn[1] != set([]):
                        c.addFinal(new)
                else:
                    new = c.stateIndex(stn)
                c.addTransition(s, sym, new)
            if j == len(l_states) - 1:
                break
            else:
                j += 1
        return c

    def star(self, flag=False):
        """Star of a DFA. If the DFA is not complete, it is completed.

        ..versionchanged: 0.9.6

        :param bool flag: plus instead of star
        :returns: the result of the star
        :rtype: DFA"""
        j = None  # to keep the checker happy
        if len(self.States) == 1 and self.finalP(self.Initial):
            return self
        d = self.dup()
        d.complete()
        c = DFA()
        c.Sigma = d.Sigma
        if len(d.States) == 0 or len(d.Final) == 0:
            # Epsilon automaton
            s0, s1 = c.addState(0), c.addState(1)
            c.setInitial(s0)
            c.addFinal(s0)
            for sym in c.Sigma:
                c.addTransition(s0, sym, s1)
                c.addTransition(s1, sym, s1)
            return c
        f0 = d.Final - {d.Initial}
        if not flag:
            i = c.addState("initial")
            c.setInitial(i)
            c.addFinal(i)
            l_states = ["initial"]
            for sym in d.Sigma:
                stn = {d.evalSymbol(d.Initial, sym)}
                # correction
                if f0 & stn != set([]):
                    stn.add(d.Initial)
                if stn not in l_states:
                    l_states.append(stn)
                    new = c.addState(stn)
                    if d.Final & stn != set([]):
                        c.addFinal(new)
                else:
                    new = c.stateIndex(stn)
                c.addTransition(i, sym, new)
                j = 1
        else:
            i = c.addState({d.Initial})
            c.setInitial(i)
            if d.finalP(d.Initial):
                c.addFinal(i)
            l_states = [{d.Initial}]
            j = 0
        while True:
            stu = l_states[j]
            s = c.stateIndex(stu)
            for sym in d.Sigma:
                stn = d.evalSymbolL(stu, sym)
                if f0 & stn != set([]):
                    stn.add(d.Initial)
                if stn not in l_states:
                    # noinspection PyTypeChecker
                    l_states.append(stn)
                    new = c.addState(stn)
                    if d.Final & stn != set([]):
                        c.addFinal(new)
                else:
                    new = c.stateIndex(stn)
                c.addTransition(s, sym, new)
            if j == len(l_states) - 1:
                break
            else:
                j += 1
        return c

    def evalSymbolI(self, init, sym):
        """Returns the state reached from a given state.

        :arg init init: current state
        :arg str sym: symbol to be consumed
        :returns: reached state or -1
        :rtype: set of int

        :raise DFAsymbolUnknown: if symbol not in alphabet

        .. versionadded:: 0.9.5

        .. note:: this is to be used with non-complete DFAs"""
        if sym not in self.Sigma:
            raise DFAsymbolUnknown(sym)
        try:
            nexti = self.delta[init][sym]
        except KeyError:
            return -1
        except NameError:
            return -1
        return nexti

    def evalSymbolLI(self, ls, sym):
        """Returns the set of states reached from a given set of states through a given symbol

        :arg ls: set of current states
        :type ls: set of int
        :arg str sym: symbol to be consumed
        :returns: set of reached states
        :rtype: set of int


        .. versionadded:: 0.9.5

        .. note:: this is to be used with non-complete DFAs"""
        return set([self.evalSymbolI(s, sym) for s in ls if self.evalSymbolI(s, sym) != -1])

    def concatI(self, fa2, strict=False):
        """Concatenation of two DFAs.

        :param DFA fa2: the second DFA
        :arg bool strict: should alphabets be checked?
        :returns: the result of the concatenation
        :rtype: DFA

        :raises DFAdifferentSigma: if alphabet are not equal

        .. versionadded:: 0.9.5

        .. note:: this is to be used with non-complete DFAs"""
        if strict and self.Sigma != fa2.Sigma:
            raise DFAdifferentSigma
        n_sigma = self.Sigma.union(fa2.Sigma)
        d1, d2 = self.dup(), fa2.dup()
        d1.setSigma(n_sigma)
        d2.setSigma(n_sigma)
        if len(d1.States) == 0 or len(d1.Final) == 0:
            return d1
        if len(d2.States) <= 1:
            if not len(d2.Final):
                return d2
        c = DFA()
        c.setSigma(d1.Sigma)
        l_states = []
        i = (d1.Initial, set([]))
        l_states.append(i)
        j = c.addState(i)
        c.setInitial(j)
        if d1.finalP(d1.Initial):
            i[1].add(d2.Initial)
            if d2.finalP(d2.Initial):
                c.addFinal(j)
        while True:
            stu = l_states[j]
            s = c.stateIndex(stu)
            for sym in d1.Sigma:
                stn = (d1.evalSymbolI(stu[0], sym), d2.evalSymbolLI(stu[1], sym))
                if not ((stn[0] == -1) & (stn[1] == {-1})) | ((stn[0] == -1) & (stn[1] == set([]))):
                    if d1.finalP(stn[0]):
                        stn[1].add(d2.Initial)
                    if stn not in l_states:
                        l_states.append(stn)
                        new = c.addState(stn)
                        if d2.Final & stn[1] != set([]):
                            c.addFinal(new)
                    else:
                        new = c.stateIndex(stn)
                    c.addTransition(s, sym, new)
            if j == len(l_states) - 1:
                break
            else:
                j += 1
        return c

    def starI(self):
        """Star of an incomplete DFA.

        .. varsionadded::: 0.9.5

        :returns: the Kleene closure DFA
        :rtype: DFA"""
        if len(self.Final) == 1 and self.finalP(self.Initial):
            return self
        d = self.dup()
        c = DFA()
        c.Sigma = d.Sigma
        if len(d.States) == 0 or len(d.Final) == 0:
            # Epsilon automaton
            s0, s1 = c.addState(0), c.addState(1)
            c.setInitial(s0)
            c.addFinal(s0)
            for sym in c.Sigma:
                c.addTransition(s0, sym, s1)
                c.addTransition(s1, sym, s1)
            return c
        f0 = d.Final - {d.Initial}
        i = c.addState("initial")
        c.setInitial(i)
        c.addFinal(i)
        l_states = ["initial"]
        for sym in d.Sigma:
            stn = {d.evalSymbolI(d.Initial, sym)}
            if (stn != set([])) & (stn != {-1}):
                # correction
                if f0 & stn != set([]):
                    stn.add(d.Initial)
                if stn not in l_states:
                    l_states.append(stn)
                    new = c.addState(stn)
                    if d.Final & stn != set([]):
                        c.addFinal(new)
                else:
                    new = c.stateIndex(stn)
                c.addTransition(i, sym, new)
        j = 1
        while True:
            stu = l_states[j]
            s = c.stateIndex(stu)
            for sym in d.Sigma:
                stn = d.evalSymbolLI(stu, sym)
                if stn != set([]):
                    if f0 & stn != set([]):
                        stn.add(d.Initial)
                    if stn not in l_states:
                        l_states.append(stn)
                        new = c.addState(stn)
                        if d.Final & stn != set([]):
                            c.addFinal(new)
                    else:
                        new = c.stateIndex(stn)
                    c.addTransition(s, sym, new)
            if j == len(l_states) - 1:
                break
            else:
                j += 1
        return c

    def shuffle(self, other, strict=False):
        """CShuffle of two languages: L1 W L2

        :param DFA other: second automaton
        :param bool strict: should the alphabets be necessary equal?
        :rtype: DFA

        .. seealso::
           C. Câmpeanu, K. Salomaa and s. Yu, *Tight lower bound for the state complexity of CShuffle of regular
           languages.* J. Autom. Lang. Comb. 7 (2002) 303–310."""
        if strict and self.Sigma != other.Sigma:
            raise DFAdifferentSigma
        n_sigma = self.Sigma.union(other.Sigma)
        d1, d2 = self.dup(), other.dup()
        d1.setSigma(n_sigma)
        d2.setSigma(n_sigma)
        # d1.complete(); d2.complete()
        c = DFA()
        c.setSigma(d1.Sigma)
        j = c.addState({(d1.Initial, d2.Initial)})
        c.setInitial(j)
        if d1.finalP(d1.Initial) and d2.finalP(d2.Initial):
            c.addFinal(j)
        while True:
            s = c.States[j]
            sn = c.stateIndex(s)
            for sym in c.Sigma:
                stn = set()
                for st in s:
                    try:
                        stn.add((d1.evalSymbol(st[0], sym), st[1]))
                    except DFAstopped:
                        pass
                    try:
                        stn.add((st[0], d2.evalSymbol(st[1], sym)))
                    except DFAstopped:
                        pass
                if stn not in c.States:
                    new = c.addState(stn)
                    for sti in stn:
                        if d1.finalP(sti[0]) and d2.finalP(sti[1]):
                            c.addFinal(new)
                            break
                else:
                    new = c.stateIndex(stn)
                c.addTransition(sn, sym, new)
            if j == len(c.States) - 1:
                break
            else:
                j += 1
        return c

    def reorder(self, dicti):
        """Reorders states according to given dictionary. Given a dictionary (not necessarily complete)... reorders
        states accordingly.

        :param dict dicti: reorder dictionary"""
        if len(list(dicti.keys())) != len(self.States):
            for i in range(len(self.States)):
                if i not in dicti:
                    dicti[i] = i
        delta = {}
        for s in self.delta:
            delta[dicti[s]] = {}
            for c in self.delta[s]:
                delta[dicti[s]][c] = dicti[self.delta[s][c]]
        self.delta = delta
        self.Initial = dicti[self.Initial]
        final = set()
        for i in self.Final:
            final.add(dicti[i])
        self.Final = final
        states = list(range(len(self.States)))
        for i in range(len(self.States)):
            states[dicti[i]] = self.States[i]
        self.States = states

    def witnessDiff(self, other):
        """ Returns a witness for the difference of two DFAs and:

        +---+------------------------------------------------------+
        | 0 | if the witness belongs to the **other** language     |
        +---+------------------------------------------------------+
        | 1 | if the witness belongs to the **self** language      |
        +---+------------------------------------------------------+

        :param DFA other: the other DFA
        :returns: a witness word
        :rtype: list of symbols
        :raises DFAequivalent: if automata are equivalent"""
        x = ~self & other
        x = x.minimal()
        result = x.witness()
        v = 0
        if result is None:
            x = ~other & self
            x = x.minimal()
            result = x.witness()
            v = 1
            if result is None:
                raise DFAequivalent
        return result, v

    def universalP(self, minimal=False):
        """Checks if the automaton is universal through minimisation

        :arg bool minimal: is the automaton already minimal?
        :rtype: bool"""
        if minimal:
            foo = self
        else:
            foo = self.minimal()
        if len(foo) == 1 and len(foo.Final) == 1:
            return True
        else:
            return False

    def usefulStates(self, initial_states=None):
        """Set of states reacheable from the given initial state(s) that have a path to a final state.

        :param initial_states: starting states
        :type initial_states: iterable of int

        :returns: set of state indexes
        :rtype: set of int"""
        # ATTENTION CODER: This is mostly a copy&paste of
        # NFA.usefulStates(), except that the inner loop for adjacent
        # states is removed, and default initial_states is a list with
        # self.Initial and is considered useful
        if initial_states is None:
            initial_states = [self.Initial]
            # useful = set()
            useful = set(initial_states)
        else:
            useful = set([s for s in initial_states
                          if s in self.Final])
        stack = list(initial_states)
        preceding = {}
        for i in stack:
            preceding[i] = []
        while stack:
            state = stack.pop()
            if state not in self.delta:
                continue
            for symbol in self.delta[state]:
                adjacent = self.delta[state][symbol]
                is_useful = adjacent in useful
                if adjacent in self.Final or is_useful:
                    useful.add(state)
                    if not is_useful:
                        useful.add(adjacent)
                        preceding[adjacent] = []
                        stack.append(adjacent)
                    inpath_stack = [p for p in preceding[state] if p not in useful]
                    preceding[state] = []
                    while inpath_stack:
                        previous = inpath_stack.pop()
                        useful.add(previous)
                        inpath_stack += [p for p in preceding[previous] if p not in useful]
                        preceding[previous] = []
                    continue
                if adjacent not in preceding:
                    preceding[adjacent] = [state]
                    stack.append(adjacent)
                else:
                    preceding[adjacent].append(state)
        return useful

    def finalCompP(self, s):
        """ Verifies if there is a final state in  strongly connected component containing ``s``.

        :param int s: state
        :returns: 1 if yes, 0 if no"""
        if s in self.Final:
            return True
        lst = [s]
        i = 0
        while True:
            try:
                foo = list(self.delta[lst[i]].keys())
            except KeyError:
                foo = []
            for c in foo:
                s = self.delta[lst[i]][c]
                if s not in lst:
                    if s in self.Final:
                        return True
                    lst.append(s)
            i += 1
            if i >= len(lst):
                return False

    def unmark(self):
        """Unmarked NFA that corresponds to a marked DFA: in which each alfabetic symbol is a tuple (symbol, index)

        :returns: a NFA
        :rtype: NFA"""
        nfa = NFA()
        nfa.States = list(self.States)
        nfa.setInitial([self.Initial])
        nfa.setFinal(self.Final)
        for s in self.delta:
            for marked_symbol in self.delta[s]:
                sym, pos = marked_symbol
                nfa.addTransition(s, sym, self.delta[s][marked_symbol])
        return nfa

    def toNFA(self):
        """Migrates a DFA to a NFA as dup()

        :returns: DFA seen as new NFA
        :rtype: NFA"""
        new = NFA()
        new.setSigma(self.Sigma)
        new.States = self.States[:]
        new.addInitial(self.Initial)
        new.Final = self.Final.copy()
        for s in self.delta:
            new.delta[s] = {}
            for c in self.delta[s]:
                new.delta[s][c] = {self.delta[s][c]}
        return new

    def stateChildren(self, state, strict=False):
        """Set of children of a state

        :param bool strict: if not strict a state is never its own child even if a self loop is in place
        :param int state: state id queried
        :returns: map children -> multiplicity
        :rtype: dictionary"""
        l = {}
        if state not in self.delta:
            return l
        for c in self.Sigma:
            if c in self.delta[state]:
                dest = self.delta[state][c]
                l[dest] = l.get(dest, 0) + 1
        if not strict and state in l:
            del l[state]
        return l

    def _smAtomic(self, monoid):
        """Evaluation of the atomic transformations of a DFA

        :arg bool monoid: monoid
        :returns: list of transformations
        :rtype: set of list of int"""
        if not self.completeP():
            aut = self.dup()
            aut.complete()
        else:
            aut = self
        n = len(aut)
        mon = SSemiGroup()
        if monoid:
            a = tuple((x for x in range(n)))
            mon.elements.append(a)
            mon.words.append((None, None))
            mon.gen.append(0)
            mon.Monoid = True
        tmp = ([], [])
        for k in aut.Sigma:
            a = tuple((aut.delta[s][k] for s in range(n)))
            tmp = mon.add(a, None, k, tmp)
        if len(tmp[0]):
            mon.addGen(tmp)
        return mon

    def _ssg(self, monoid=False):
        """

        :param bool monoid:
        :return:"""
        sm = self._smAtomic(monoid)
        if not sm.gen[-1]:
            return sm
        if sm.Monoid:
            natomic = sm.gen[1]
            shift = 1
        else:
            natomic = sm.gen[0]
            shift = 0
        while True:
            ll = ([], [])
            if len(sm.gen) == 1:
                g0 = 0
            else:
                g0 = sm.gen[-2] + 1
            g1 = sm.gen[-1] + 1
            for (sym, t1) in enumerate(sm.elements[1:natomic + 1]):
                for (pr, t2) in enumerate(sm.elements[g0:g1]):
                    t12 = tuple((t2[t1[i]] for i in range(len(t1))))
                    ll = sm.add(t12, pr + g0, sm.words[sym + shift][1], ll)
            if len(ll[0]):
                sm.addGen(ll)
            else:
                break
        return sm

    def sMonoid(self):
        """Evaluation of the syntactic monoid of a DFA

        :returns: the semigroup
        :rtype: SSemiGroup"""
        return self._ssg(True)

    def sSemigroup(self):
        """Evaluation of the syntactic semigroup of a DFA

        :returns: the semigroup
        :rtype: SSemiGroup"""
        return self._ssg()

    def enumDFA(self, n=None):
        """
        returns the set of words of words of length up to n accepted by self
        :param int n: highest length or all words if finite

        :rtype: list of strings or None

        .. note: use with care because the number of words can be huge
        """
        if n is None:
            raise IndexError
        e = EnumDFA(self)
        words = []
        for i in range(n + 1):
            e.enumCrossSection(i)
            words += e.Words
        return words

    def completeP(self):
        """Checks if it is a complete FA (if delta is total)

        :return: bool"""
        if not self.Sigma:
            return True
        ss = len(self.Sigma)
        for s, _ in enumerate(self.States):
            if s not in self.delta:
                return False
            ni = set(self.delta[s])
            if len(ni) != ss:
                return False
        return True

    def complete(self, dead=DeadName):
        """Transforms the automata into a complete one. If sigma is empty nothing is done.

        :param str dead: dead state name
        :return: the complete FA
        :rtype: DFA

        .. note::
           Adds a dead state (if necessary) so that any word can be processed with the automata. The new state is
           named ``dead``, so this name should never be used for other purposes.

        .. attention::
           The object is modified in place.

        .. versionchanged:: 1.0"""
        if self.completeP():
            return self
        ss = len(self.Sigma)
        f = True
        trash = self.stateIndex(dead, True)
        for s, _ in enumerate(self.States):
            if s not in self.delta:
                self.delta[s] = {}
            ni = list(self.delta[s].keys())
            if len(ni) != ss:
                for c in self.Sigma:
                    if c not in ni:
                        self.addTransition(s, c, trash)
                        f = False
        if f:
            self.deleteState(trash)
        return self

    def transitions(self):
        """ Iterator over transitions
        :rtype: symbol, int"""
        for i in self.delta:
            for c in self.delta[i]:
                yield i, c, self.delta[i][c]

    def transitionsA(self):
        """ Iterator over transitions
        :rtype: symbol, int"""
        for i in self.delta:
            for c in self.delta[i]:
                yield i, c, [self.delta[i][c]]


class EnumL(object):
    """Class for enumerate FA languages
            See: Efficient enumeration of words in regular languages, M. Ackerman and J. Shallit,
            Theor. Comput. Sci. 410, 37, pp 3461-3470. 2009.
            http://dx.doi.org/10.1016/j.tcs.2009.03.018

        :ivar FA aut: Automaton of the language
        :ivar dict tmin: table for minimal words for each s in aut.States
        :ivar list Words: list of words (if stored)
        :ivar list sigma: alphabet
        :ivar deque stack:

        .. inheritance-diagram:: EnumL

        .. versionadded:: 0.9.8"""

    def __init__(self, aut, store=False):
        self.aut = aut
        self.tmin = {}
        self.stack = None
        self.Words = []
        self.Sigma = list(aut.Sigma)
        self.Sigma.sort(key=lambda x: x.__repr__())
        self.store = store
        self.initStack()

    @abstractmethod
    def initStack(self):
        """Abstract method"""
        pass

    @abstractmethod
    def minWordT(self, n):
        """Abstract method
        :param int n:
        :type n: int"""
        pass

    def minWord(self, m):
        """ Computes the minimal word of length m accepted by the automaton
        :param m:
        :type m: int"""
        if m == 0:
            return ""
        if len(self.tmin) == 0:
            self.minWordT(m)

        possiblew = [self.tmin[q][m] for q in self.stack[0] if q in self.tmin and m in self.tmin[q]]
        if not possiblew:
            return None
        return min(possiblew)

    def iCompleteP(self, i, q):
        """Tests if state q is i-complete

        :param int i: int
        :param int q: state index"""
        return i in self.tmin[q] or (i == 0 and self.aut.finalP(q))

    @abstractmethod
    def fillStack(self, w):
        """Abstract method
        :param str w:
        :type w: str"""
        pass

    @abstractmethod
    def nextWord(self, w):
        """Abstract method
        :param w:
        :type w: str"""
        pass

    def enumCrossSection(self, n):
        """ Enumerates the nth cross-section of L(A)

        :param int n: nonnegative integer"""
        self.Words = []
        if n == 0:
            if self.aut.evalWordP(""):
                self.Words.append("")
            return
        self.initStack()
        self.tmin = {}
        w = self.minWord(n)
        while w is not None:
            self.fillStack(w)
            self.Words.append(w)
            w = self.nextWord(w)
        self.tmin = {}
        self.initStack()

    def enum(self, m):
        """Enumerates the first m words of L(A) according to the lexicographic order if there are at least m words.
        Otherwise, enumerates all words accepted by A.

        :param int m: max number of words"""
        i = 0
        lim = 1
        num_cec = 0
        s = len(self.aut)
        if not (not (isinstance(self.aut, DFA) and self.aut.finalP(self.aut.Initial))
                and not (isinstance(self.aut, NFA) and not self.aut.Initial.isdisjoint(self.aut.Final))):
            self.Words = [""]
            i += 1
            num_cec += 1
        else:
            self.Words = []
        while i < m and num_cec < s:
            self.initStack()
            self.tmin = {}
            w = self.minWord(lim)
            if w is None:
                num_cec += 1
            else:
                num_cec = 0
                while w is not None and i < m:
                    i += 1
                    self.Words.append(w)
                    self.fillStack(w)
                    w = self.nextWord(w)
            lim += 1
        self.tmin = {}
        self.initStack()


class EnumDFA(EnumL):
    """Class for enumerating languages defined by DFAs

    .. inheritance-diagram:: EnumDFA"""

    def minWordT(self, n):
        """ Computes for each state the minimal word of length i<n
        accepted by the automaton. Stores the values in tmin

        :param int n: length of the word

        .. note:: Makinen algorithm for DFAs"""
        for i in range(len(self.aut)):
            if i not in self.tmin:
                self.tmin[i] = {}
            for sym in self.Sigma:
                if i in self.aut.delta and sym in self.aut.delta[i] and self.aut.finalP(self.aut.delta[i][sym]):
                    self.tmin.setdefault(i, {})[1] = sym
                    break
        for j in range(2, n + 1):
            for i in range(len(self.aut)):
                m = None
                if i in self.aut.delta:
                    for sym in self.Sigma:
                        if sym in self.aut.delta[i]:
                            q = self.aut.delta[i][sym]
                            if q in self.tmin and j - 1 in self.tmin[q]:
                                m = sym + self.tmin[q][j - 1]
                                break
                if m is not None:
                    self.tmin[i][j] = m

    def fillStack(self, w):
        """ Computes S_1,...,S_n-1 where S_i is the set of (n-i)-complete states reachable from S_i-1

        :param w: word"""
        n = len(w)
        self.initStack()
        for i in range(1, n):
            s = set({})
            for j in self.stack[0]:
                if j in self.aut.delta and w[i - 1] in self.aut.delta[j] and \
                        self.iCompleteP(n - i, self.aut.delta[j][w[i - 1]]):
                    s.add(self.aut.delta[j][w[i - 1]])
            self.stack.appendleft(s)

    def initStack(self):
        """Initializes the stack with initial states """
        self.stack = deque([{self.aut.Initial}])

    def nextWord(self, w):
        """Given an word, returns next word on the nth cross-section of L(aut)
        according to the radix order

        :param str w: word
        :rtype: str"""
        n = len(w)
        for i in range(n, 0, -1):
            s = self.stack[0]
            b = self.Sigma[-1]
            flag = 0
            for j in s:
                if j in self.aut.delta:
                    for sym in self.Sigma:
                        if sym in self.aut.delta[j] and self.iCompleteP(n - i, self.aut.delta[j][sym]):
                            if w[i - 1] < sym:
                                if sym < b:
                                    b = sym
                                flag = 1
            if flag == 0:
                self.stack.popleft()
            else:
                s1 = set([])
                for j in s:
                    if j in self.aut.delta:
                        if b in self.aut.delta[j] and self.iCompleteP(n - i, self.aut.delta[j][b]):
                            s1.add(self.aut.delta[j][b])
                if i != n:
                    self.stack.appendleft(s1)
                mw = self.minWord(n - i)
                if mw is not None:
                    return w[0:i - 1] + b + mw
        return None


class EnumNFA(EnumL):
    """Class for enumerating languages defined by NFAs

    .. inheritance-diagram:: EnumNFA"""

    def initStack(self):
        """Initializes the stack with initial states
        """
        self.stack = deque([self.aut.Initial])

    def minWordT(self, n):
        """ Computes for each state the minimal word of length i <= n
        accepted by the automaton. Stores the values in tmin.

        :param int n: length of the word

        .. note: Makinen algorithm for NFAs"""
        for i in range(len(self.aut)):
            self.tmin[i] = {}
            for sym in self.Sigma:
                if i in self.aut.delta and sym in self.aut.delta[i]:
                    if not self.aut.delta[i][sym].isdisjoint(self.aut.Final):
                        self.tmin.setdefault(i, {})[1] = sym
                        break
        for j in range(2, n + 1):
            for i in range(len(self.aut)):
                m = None
                if i in self.aut.delta:
                    for sym in self.Sigma:
                        if sym in self.aut.delta[i]:
                            for q in self.aut.delta[i][sym]:
                                if q in self.tmin and j - 1 in self.tmin[q]:
                                    if m is None or sym + self.tmin[q][j - 1] < m:
                                        m = sym + self.tmin[q][j - 1]
                if m is not None:
                    self.tmin.setdefault(i, {})[j] = m

    def fillStack(self, w):
        """ Computes S_1,...,S_n-1 where S_i is the set of (n-i)-complete states reachable from S_i-1

        :param w: word"""
        n = len(w)
        self.initStack()
        for i in range(1, n):
            s = set([])
            for j in self.stack[0]:
                if j in self.aut.delta and w[i - 1] in self.aut.delta[j]:
                    for q in self.aut.delta[j][w[i - 1]]:
                        if self.iCompleteP(n - i, q):
                            s.add(q)
            if len(s) != 0:
                self.stack.appendleft(s)

    def nextWord(self, w):
        """Given an word, returns next word in the the nth cross-section of L(aut)
        according to the radix order

        :param str w: word"""
        n = len(w)
        for i in range(n, 0, -1):
            if len(self.stack) == 0:
                return None
            s = self.stack[0]
            b = self.Sigma[-1]
            flag = 0
            for j in s:
                if j in self.aut.delta:
                    for sym in self.Sigma:
                        if sym in self.aut.delta[j]:
                            for q in self.aut.delta[j][sym]:
                                if self.iCompleteP(n - i, q):
                                    if w[i - 1] < sym:
                                        if sym < b:
                                            b = sym
                                        flag = 1
            if flag == 0:
                self.stack.popleft()
            else:
                s1 = set([])
                for j in s:
                    if j in self.aut.delta and b in self.aut.delta[j]:
                        for q in self.aut.delta[j][b]:
                            if self.iCompleteP(n - i, q):
                                s1.add(q)
                if i != n:
                    self.stack.appendleft(s1)
                mw = self.minWord(n - i)
                if mw is not None:
                    return w[0:i - 1] + b + mw
        return None


# class Word_Generator(DFA):
#     """DFA with anotations for word generation"""
#     def __init__(self, fa :DFA):
#         self.fa = fa
#         self.fa._compute_delta_inv()
#         self.min_sufix_sz = dict()
#         self.min_sufix = dict()
#         done = set()
#         todo = list(self.fa.Final)
#         for id in self.fa.Final:
#             self.min_sufix_sz[id] = 0
#             self.min_sufix[id] = [""]
#         while todo:
#             s = todo.pop(0)
#             done.add(s)
#             for c in self.fa.delta_inv[s]:
#                 for q in self.fa.delta_inv[s][c]:
#                     if q not in done:
#                         self.min_sufix_sz[q] = self.min_sufix_sz[s] + 1
#                         nl = [c + o for o in self.min_sufix[s]]
#                         self.min_sufix[q] = self.min_sufix.get(q,[]) + nl
#                         todo.append(q)
#
#     def generate_word(self, sz :int)->Word:
#         """Generate a word of size sz. Assumes that the automaton is trim
#
#         Args:
#             sz (int): size of the word
#         Returns:
#             Word:"""
#         if sz == 0:
#             return Word(Epsilon)
#         else:
#             w = Word()
#             cs = self.fa.Initial
#             while True:
#                 cn = len(self.fa.delta[cn])
#                 c = self.fa.Sigma[random.randint(0,cn-1)]
#                 w.append(self.Sigma[r])


def stringToDFA(s, f, n, k):
    """ Converts a string icdfa's representation to dfa.

    :param list s: canonical string representation
    :param list f: bit map of final states
    :param int n: number of states
    :param int k: number of symbols
    :returns: a complete dfa with sigma [``k``], States [``n``]
    :rtype: DFA

    .. versionchanged:: 0.9.8 symbols are converted to str"""
    fa = DFA()
    fa.setSigma([])
    fa.States = list(range(n))
    j = 0
    i = 0
    while i < len(f):
        if f[i]:
            fa.addFinal(j)
        j += 1
        i += 1
    fa.setInitial(0)
    for i in range(n * k):
        if s[i] != -1:
            fa.addTransition(i // k, str(i % k), s[i])
    return fa


def _cmpPair2(a, b):
    """Auxiliary comparision for sorting lists of pairs. Sorting on the second member of the pair."""
    (x, y), (z, w) = a, b
    if y < w:
        return -1
    elif y > w:
        return 1
    elif x < z:
        return -1
    elif x > z:
        return 1
    else:
        return 0


def _cmpPair2Key(a, b):
    return b, a


def _normalizePair(p, q):
    if p < q:
        pair = (p, q)
    else:
        pair = (q, p)
    return pair


def _sortWithNone(a, b):
    if a is None:
        return a, b
    elif b is None:
        return b, a
    elif a >= b:
        return a, b
    else:
        return b, a


def _deref(mp, val):
    if val in mp:
        return _deref(mp, mp[val])
    else:
        return val


def _dictGetKeyFromValue(elm, dic):
    try:
        key = [i for i, j in list(dic.items()) if elm in j][0]
    except IndexError:
        key = None
    return key


def statePP(state):
    """Pretty print state

    :param state:
    :return:"""

    def _spp(st):
        t = type(st)
        if t == str:
            return copy(st).replace(' ', '')
        elif t == int:
            return str(st)
        elif t == tuple:
            bar = "("
            for s in st:
                bar += _spp(s) + ","
            return bar[:-1] + ")"
        elif t == set:
            bar = "{"
            for s in st:
                bar += _spp(s) + ","
            return bar[:-1] + "}"
        else:
            return str(st)

    foo = _spp(state)
    if len(foo) > 1:
        return '"' + foo + '"'
    else:
        return foo


def saveToString(aut, sep="&"):
    """Finite automata definition as a string using the input format.

    .. versionadded:: 0.9.5
    .. versionchanged:: 0.9.6 Names are now used instead of indexes.
    .. versionchanged:: 0.9.7 New format with quotes and alphabet

    :param FA aut: the FA
    :arg str sep: separation between `lines`
    :returns: the representation
    :rtype: str """
    buff = ""
    if aut.Initial is None:
        return "Error: no initial state defined"
    if isinstance(aut, DFA):
        buff += "@DFA "
        nf_ap = False
    elif isinstance(aut, NFA):
        buff += "@NFA "
        nf_ap = True
    else:
        raise DFAerror()
    if not nf_ap and aut.Initial != 0:
        foo = {0: aut.Initial, aut.Initial: 0}
        aut.reorder(foo)
    for sf in aut.Final:
        buff += ("{0:>s} ".format(statePP(aut.States[sf])))
    if nf_ap:
        buff += " * "
        for sf in aut.Initial:
            buff += ("{0:>s} ".format(statePP(aut.States[sf])))
    buff += sep
    for s in range(len(aut.States)):
        if s in aut.delta:
            for a in aut.delta[s]:
                if isinstance(aut.delta[s][a], set):
                    for s1 in aut.delta[s][a]:
                        buff += ("{0:>s} {1:>s} {2:>s}{3:>s}".format(statePP(aut.States[s]), str(a),
                                                                     statePP(aut.States[s1]), sep))
                else:
                    buff += ("{0:>s} {1:>s} {2:>s}{3:>s}".format(statePP(aut.States[s]), str(a),
                                                                 statePP(aut.States[aut.delta[s][a]]), sep))
        else:
            buff += "{0:>s} {1:>s}".format(statePP(aut.States[s]), sep)
    return buff


def sigmaStarDFA(sigma=None):
    """Given a alphabet s returns the minimal DFA for s*

    :param set sigma: set of simbols
    :rtype: DFA

    .. versionadded:: 1.2"""
    if sigma is None:
        raise DFAerror
    d = DFA()
    d.setSigma(sigma)
    i = d.addState()
    d.setInitial(i)
    d.addFinal(i)
    for a in d.Sigma:
        d.addTransition(i, a, i)
    return d


def emptyDFA(sigma=None):
    """Given an alphabet returns the minimal DFA for the empty language

    :param set sigma: set of symbols
    :rtype: DFA

    .. versionadded:: 1.3.4.2"""
    if sigma is None:
        raise DFAerror
    d = DFA()
    d.setSigma(sigma)
    i = d.addState()
    d.setInitial(i)
    for a in d.Sigma:
        d.addTransition(i, a, i)
    return d

def symbolDFA(sym, sigma=None):
    """Given  symbol and an alphabet returns the minimal DFA that aceepts that symbol

        :param sym: symbol
        :param set sigma: set of symbols
        :rtype: DFA

        .. versionadded 2.1"""
    new = DFA()
    s0 = new.addState()
    s1 = new.addState()
    new.setInitial(s0)
    if sigma is None:
        new.setSigma(sym)
    else:
        new.setSigma(sigma)
    new.setFinal([s1])
    new.addTransition(s0, sym, s1)
    return new

def _addPool(pool, done, val):
    """ Adds to a pool with exception list

    :param set pool: pool to be added
    :param set done: exception list
    :param val: value"""
    if val in done:
        return
    else:
        pool.add(val)


def _initPool():
    """Initialize pool structure

    :return: pool and done objects
    :rtype: tuple"""
    return set(), set()
