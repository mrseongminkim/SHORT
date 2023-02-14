# coding=utf-8
"""**Syntactic SemiGroup.**

Deterministic and non-deterministic automata manipulation, conversion and evaluation.

.. *Authors:* Rogério Reis & Nelma Moreira

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

# from . common import *


# noinspection PyTypeChecker
class SSemiGroup(object):
    """Class support for the Syntactic SemiGroup.

    :var elements: list of tuples representing the transformations
    :var words: a list of pairs (index of the prefix transformation, index of the suffix char)
    :var gen: a list of the max index of each generation
    :var sigma: set of symbols
    """

    def __init__(self):
        """

        """
        self.elements = []
        self.words = []
        self.gen = []
        self.Monoid = False
        self.Sigma = {}

    def __len__(self):
        """Size of the semigroup

        :return: size of the semigroup
        :rtype: int """
        return len(self.elements)

    def __repr__(self):
        """SSemiGroup representation

        :rtype: str"""
        foo = "Semigroup:\n"
        for s in self.elements:
            foo += "%s \n" % str(s)
        return foo

    def WordI(self, i):
        """Representative of an element given as index

        :arg int i: index of the element
        :returns: the first word originating the element
        :rtype: str"""
        return self.WordPS(self.words[i][0], self.words[i][1])

    def WordPS(self, pref, sym):
        """Representative of an element given as prefix symb

        :arg int pref: prefix index
        :arg int sym: symbol index
        :returns: word
        :rtype: str"""
        if pref is None:
            if sym is None:
                return []
            else:
                return [sym]
        else:
            return self.WordPS(self.words[pref][0], self.words[pref][1]) + [sym]

    def add(self, tr, pref, sym, tmpLists):
        """Try to add a new transformation to the monoid

        :arg tr: transformation
        :type tr: tuple of int
        :arg pref: prefix of the generating word
        :type pref: int or None
        :arg int sym: suffix symbol
        :arg tmpLists: this generation lists
        :type tmpLists: pairs of lists as (elements,words)"""
        if tr not in self.elements and tr not in tmpLists[0]:
            tmpLists[0].append(tr)
            tmpLists[1].append((pref, sym))
        return tmpLists

    def addGen(self, tmpLists):
        """Add a new generation to the monoid

        :arg tmpLists: the new generation data
        :type tmpLists: pair of lists as (elements, words)"""
        gn = len(tmpLists[0])
        self.elements += tmpLists[0]
        self.words += tmpLists[1]
        if len(self.gen) > 1:
            self.gen.append(self.gen[-1] + gn)
        else:
            self.gen.append(gn)
