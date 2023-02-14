# -*- coding: utf-8 -*-
"""**Polynomial Random Approximation Algorithms**

.. *Authors:* Rogério Reis, Nelma Moreira & Stavros Konstantinidis

.. *This is part of FAdo project*   https://fado.dcc.fc.up.pt.

.. *Copyright:* 1999-2022 Rogério Reis & Nelma Moreira {rogerio.reis, nelma.moreira}@fc.up.pt

.. *Contributions by*
     - Mitja Mastnak

.. This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as published
   by the Free Software Foundation; either version 2 of the License, or
   (at your Option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   675 Mass Ave, Cambridge, MA 02139, USA."""

#  Copyright (c) 2023. Rogério Reis <rogerio.reis@fc.up.pt> and Nelma Moreira <nelma.moreira@fc.up.pt>.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

from . common import *
from .codes import *
from .fl import dfa_block
from random import random, randint
from math import ceil, pow
import multiprocessing as mp


def minI(a, t, u=None):
    """ An operator that returns a t-independent language containing L(a)

    Args:
        a (FA): the initial automaton
        t (Transducer): input-altering transducer
        u (FA | None): universe to consider
    Returns:
        NFA: """
    tinv = t.inverse()
    tt = t | tinv
    if u is None:
        b = ~(tt.runOnNFA(a).trim())
    else:
        b = (u.toNFA() & ~ (tt.runOnNFA(a).eliminateEpsilonTransitions()))
    return (b & ~(tinv.runOnNFA(b)).trim()).trim()


def unive_index(g, aut):
    """Universality index (approximate) of an automaton for a given distribution

    Args:
        g (GenWordDis): distribution
        aut (FA): automaton
    Returns:
        float:universality index"""
    n, m = g.prax_parameters()
    pos = 0.0
    for _i in range(n):
        w = next(g)
        if w is None or aut.evalWordP(w):
            pos += 1
    return pos/n


def maximality_index(g, aut, prop):
    """Maximality index (approximate) of a automaton for a given
        distribution and transducer property prop

    Args:
        g (GenWordDis): distribution
        aut (FA): automaton
        prop (IPTProp): transducer property
    Returns:
        float: universality index"""
    b = prop.Aut.runOnNFA(aut) | prop.Aut.inverse().runOnNFA(aut) | aut
    return unive_index(g, b)


class _Max_test_context(object):
    def __init__(self, g, b):
        self.g = g
        self.b = b

    def do_it(self, _x) -> bool:
        w = next(self.g)
        return self.b.evalWordP(w)


def _do_one_maximal_test(g, b):
    w = next(g)
    if b.evalWordP(w):
        return 1
    else:
        return 0


def maximal_index_p(g, aut, prop):
    """Maximality index of a automaton for a given distribution and code property (parallel version)

    Args:
        g (GenWordDis): distribution
        aut (FA): automaton
        prop (CodeProperty):
    Returns:
        float: maximality index"""
    n, m = g.prax_parameters()
    pool = mp.pool(mp.cpu_count())
    b = prop.Aut.runOnNFA(aut) | prop.Aut.inverse().runOnNFA(aut) | aut
    context = _Max_test_context(g, b)
    result = pool.map(context.do_it(), range(n))
    return float(sum(result))/n


def prax_univ_nfa(g, a, debug=False):
    """Polynomial Randomized Approximation (PRAX) for NFA universality

    Args:
        a (FA): the automaton being tested
        g (GenWordDis): word generator
        debug (bool):
    Returns:
        bool:

    .. seealso::
        S.Konstantinidis, M.Mastnak, N.Moreira, R.Reis. Approximate NFA Universality and Related Problems Motivated
        by Information Theory, arXiv, 2022.

    .. versionadded:: 2.0.4"""
    n, m = g.prax_parameters()
    for _i in range(n):
        w = next(g)
        if w is not None and not a.evalWordP(w):
            if debug:
                print("couterexample of size-> ", len(w), end=" ")
            return False
    return True


def prax_maximal_nfa(g, a, prop, debug=False):
    """Polynomial Randomized Approximation (PRAX) for NFA maximality wrt a code
        property

    Args:
        g (GenWordDis): distribution
        a (FA): automaton
        prop (IPTProp): transducer property
        debug (bool):
    Returns:
        bool:   """
    b = prop.Aut.runOnNFA(a) | prop.Aut.inverse().runOnNFA(a) | a
    return prax_univ_nfa(g, b.trim(), debug)


class PDistribution(object):
    """Probability Distribution"""
    def max_length(self, e):
        return None


class Dirichlet(PDistribution):
    """Dirichlet distribution function

    Args:
        d (int | float): displacement
        t (int | float):
    Returns:
        float:

    .. versionadded:: 2.0.4"""
    def __init__(self, t=2.000001, d=1):
        self.d = d
        self.t = t

    def f(self, n):
        if n >= self.d:
            return 1/zeta(self.t) * ((n + 1 - self.d) ** (-self.t))
        else:
            return 0

    def max_length(self, e):
        """Computes the maximal length that needs to be considered for a given error

        Args:
            e (float): error
        Returns:
            int:"""
        foo = pow(1 / (zeta(self.t) * e), (1 / (self.t - 1))) + self.d - 1
        # estimated correct upper bound
        # foo = pow(1 / e, (1 / (self.t - 1))) + self.d - 1
        return ceil(foo)

    def average(self):
        return (self.d - 1) + (zeta(self.t - 1)/zeta(self.t))

    def sum(self, n):
        return sum(1 / (i + 1 - self.d) ** self.t for i in range(self.d, n + 1)) / zeta(self.t)

    def sum_minus(self, l):
        return 1 - sum(1 / (i + 1 - self.d) ** self.t for i in l) / zeta(self.t)

    def sum_list2(self, L, F):
        """Returns the Dirichlet D_{t,d} probability of the set of words
           which is the union of (1/2**F[i]) of the words of length L[i],
           for i=0,...,len(L)-1
        
        Args:
            t (float): parameter in (1,+∞) of the Dirichlet distribution
            d (int):  minimum length of a word w for which D_{t,d}(w)>0
            L (list): list of word lengths (integers)
            F (list): list of lengths (integers)
        Returns:
            DFA:  """
        n = len(L)
        return sum(1 / 2 ** F[i] * 1 / (L[i] + 1 - self.d) ** self.t for i in range(n)) / zeta(self.t)

    def sum_minus2(self, L, F):
        """Returns the probability of the complement of the set referred
           to in the function  sum_dirichlet_list2(t,d,L,F) """
        return 1 - self.sum_list2(L, F)


class Lambert(PDistribution):
    """Laplace distribution function

    Args:
         d (int): displacement
         z (float): a number 9<z<1
    Returns:
         float:

    Raises:
        FAdoGeneralError: if z is null"""
    def __init__(self, d=1, z=0.9):
        self.d = d
        self.z = z

    def f(self, n):
        if self.z == 0.0:
            raise FAdoGeneralError("Value of z cannot be null")
        z = 1/self.z
        if n < self.d:
            return 0.0
        else:
            return (1 - z) * z ** (n - self.d)

    def average(self):
        z = 1 / self.z
        return self.d - (1 / (1 - z))


class GenWordDis(object):
    """Word generator according to a given distribution function (used for sizes), for prax test

    :ivar list sigma: alphabet
    :ivar PDistribution pd: distribution
    :ivar float e: acceptable error
    :ivar int n_tries: size of the sample
    :ivar int max_length: maximal size of the words sampled
    :ivar list dist: cumulative probability for each size considered (up to max_length)"""
    def __init__(self, f, alf, e, strict=False):
        self.sigma = list(alf)
        self.pd = f
        e1 = min(e, 1/6)
        self.e = e1
        self.n_tries = ceil(5 / (e1 - 5 * e1 ** 2) ** 2)
        foo = self.pd.max_length(e ** 2)
        # foo = None
        if foo is None:
            s, i = 0, 1
            while s + e1 * e1 < 1:
                s += self.pd.f(i)
                i += 1
            # was i-1
            self.max_length = i
        else:
            self.max_length = foo
        if strict:
            self.precompute()
        else:
            self.dist = None

    def precompute(self):
        foo = 0
        self.dist = []
        # was + 1
        for i in range(1, self.max_length):
            bar = self.pd.f(i)
            self.dist.append(foo + bar)
            foo += bar
        # new
        self.dist.append(1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.dist is None:
            self.precompute()
        r = random()
        sz = self._find(r, 0, self.max_length)
        k = len(self.sigma)
        # return Word([self.sigma[randint(0, k - 1)] for _ in range(sz)])
        if sz == self.max_length:
            return None
        else:
            return Word([self.sigma[randint(0, k - 1)] for _ in range(sz)])

    def _find(self, r, mi, ma):
        if mi == ma:
            return mi + 1
        elif ma - mi == 1:
            if r <= self.dist[mi]:
                return mi + 1
            else:
                return ma + 1
        else:
            i = (ma - mi) // 2
            if r <= self.dist[mi + i]:
                return self._find(r, mi, mi + i)
            else:
                return self._find(r, mi + i, ma)

    def prax_parameters(self):
        return self.n_tries, self.max_length



def block_unive_index(eps, a):
    """Universality index of a block automaton (accepting words of fixed length)
        Using approx tolerance eps

    Args:
        eps (float): approximation tolerance
        a (FA): block automaton (accepting words of fixed length)
    Returns:
        float: block universality index"""
    w = a.witness()
    if w is None:
        return 0
    m = len(w)
    n = ceil(1/eps**2)
    sigma = list(a.Sigma)
    k = len(sigma)
    pos = 0.0
    for _i in range(n):
        w = Word([sigma[randint(0, k - 1)] for _ in range(m)])
        if a.evalWordP(w):
            pos += 1
    return pos/n


def block_maximality_index(eps, a, prop):
    """Maximality index of a block automaton for a given transducer property prop
        Using approx tolerance eps

    Args:
        eps (float): approximation tolerance
        a (FA): block automaton (accepting words of fixed length)
        prop (IPTProp): transducer property
    Returns:
        float: block maximality index"""
    w = a.witness()
    if w is None:
        return 0
    m = len(w)
    b = prop.Aut.runOnNFA(a) | prop.Aut.inverse().runOnNFA(a) | a
    b = b.elimEpsilon().trim() & dfa_block(m, a.Sigma)
    return block_unive_index(eps, b)


def prax_block_univ_nfa(eps, a, debug=False):
    """Polynomial Randomized Approximation (PRAX) for block NFA universality

    Args:
        eps (float): approximation tolerance
        a (FA): block automaton (accepting words of fixed length)
        debug (bool):
    Returns:
        bool:

    .. seealso::
        S.Konstantinidis, M.Mastnak, N.Moreira, R.Reis. Approximate NFA Universality and Related Problems Motivated
        by Information Theory, arXiv, 2022."""

    w = a.witness()
    if w is None:
        return False
    m = len(w)
    n = ceil(1/eps**2)
    sigma = list(a.Sigma)
    k = len(sigma)
    for _i in range(n):
        w = Word([sigma[randint(0, k - 1)] for _ in range(m)])
        if not a.evalWordP(w):
            if debug:
                print("counterexample -> ", w, end=" ")
            return False
    return True


def prax_block_maximal_nfa(eps, a, prop, debug=False):
    """Polynomial Randomized Approximation (PRAX) for block NFA maximality
        wrt a code property

    Args:
        eps (float): approximation tolerance
        a (FA): block automaton (accepting words of fixed length)
        prop (IPTProp): transducer property
        debug (bool):
    Returns:
        bool:   """
    w = a.witness()
    if w is None:
        return False
    m = len(w)
    b = prop.Aut.runOnNFA(a) | prop.Aut.inverse().runOnNFA(a) | a
    b = b.elimEpsilon().trim() & dfa_block(m, a.Sigma)
    return prax_block_univ_nfa(eps, b, debug)

