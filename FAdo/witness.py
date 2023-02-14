# -*- coding: utf-8 -*-
""" Witness language families for operations on DFAs

@author: Rogério Reis & Nelma Moreira

This is part of U{FAdo project <https://fado.dcc.up.pt>}.

Deterministic and non-deterministic automata manipulation, conversion and
evaluation.

@copyright: 1999-2022 Rogério Reis & Nelma Moreira {rogerio.reis,nelma.moreira}@fc.up.pt

Contributions by
 - Marco Almeida
 - Hugo Gouveia 
 - Davide Nabais
 - Eva Maia
 - Stavros Konstantinidis

B{Naming convention:} methods suffixed by P have boolean return.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA."""

#  Copyright (c) 2023. Rogério Reis <rogerio.reis@fc.up.pt> and Nelma Moreira <nelma.moreira@fc.up.pt>.
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

from . comboperations import *
from . reex import *
from . fa import DFA
from . fl import *


# Useful automata
def emptyDFA(sigma=None):
    """
    Returns the minimal DFA for EmptySet (incomplete)

    :param sigma:
    :return:
    """
    d = DFA()
    if sigma is not None:
        d.setSigma(sigma)
    i = d.addState()
    d.setInitial(i)
    return d


def epsilonDFA(sigma=None):
    """
    Returns the minimal DFA for {CEpsilon} (incomplete)

    :param sigma:
    :return:
    """
    d = DFA()
    if sigma is not None:
        d.setSigma(sigma)
    i = d.addState()
    d.setInitial(i)
    d.addFinal(i)
    return d


# Worst case automata for each operation
witnessDFA = {"toDFA": [("toDFAWCMF", "int"),
                        ("toDFAWC2", "int"),
                        ("toDFAWC3", "int")],
              "reversal": [("reversalWC3M", "int"),
                           ("reversalMB", "int"),
                           ("reversalWC3L", "int"),
                           ("reversalternaryWC", "int"),
                           ("reversalbinaryWC", "int")],
              "star": [("starWC", "int"),
                       ("starWCM", "int")],
              "concat": [("concatWC", "int", "int"),
                         ("concatWCM", "int", "int")],
              "conjunction": [("interWC", "int", "int")],
              "__or__": [("disjWC", "int", "int")],
              "shuffle": [("shuffleWC", "int", "int")],
              "suff": [("suffWCe", "int"), ("suffWCd", "int")],
              "starDisj": [("starDisjWC", "int", "int")],
              "starInter": [("starInterBC", "int", "int")],
              "disjWStar": [("disjWStarWC", "int", "int")]}


def toDFAWC2MF(m=5, Sigma=["a", "b"]):
    """ Worst case automata for toDFA(NFA) with n > 2, k=2
    ..seealso::
    A. r. Meyer and M. J. Fischer. Economy of description by automata,
    grammars, and formal systems. Twelfth Annual Symposium on
    Switching and Automata Theory, 1971,  188–191. IEEE Society Press.

    :param m: number of states
    :type m: integer
    :param Sigma: alphabet
    :return: a dfa
    :rtype: DFA"""

    if m < 3:
        raise TstError("Number of states must be greater than 2")
    if len(Sigma) != 2:
        raise TstError("Alphabet must be binary")
    f = NFA()
    f.setSigma(Sigma)
    f.States = list(range(m))
    f.setInitial([0])
    f.addFinal(0)
    f.addTransition(0, Sigma[0], 1)
    for i in range(1, m):
        f.addTransition(i, Sigma[0], (i + 1) % m)
        f.addTransition(i, Sigma[1], i)
        f.addTransition(i, Sigma[1], 0)
    return f


def toDFAWCReMF(m=5, Sigma=["a", "b"]):
    """
    Same as toDFAWC2MF as regular expression
    :param m:
    :param Sigma: alphabet
    :return:
    """
    if len(Sigma) != 2:
        raise TstError("Alphabet must be binary")
    sig = [CAtom(i) for i in Sigma]
    r0 = CDisj(sig[1], CConcat(CConcat(sig[0], CStar(sig[1])), CDisj(sig[0], sig[1])))
    for j in range(1, m-1):
        r0 = CConcat(CConcat(sig[0], CStar(sig[1])), CDisj(r0, sig[1]))
    return CStar(r0)


def toDFAWC2(m=5):
    """ Worst case automata for toDFA(NFA) with n > 2, k=2
    ..seealso:: F.r. Moore. On the bounds for state-set size in the proofs
    of equivalence between deterministic, nondeterministic, and
    two-way finite automata. IEEE Transactions on computers, 2:1211–1214, 1971.

    :arg m: number of states
    :type m: integer
    :returns: a dfa
    :rtype: DFA"""

    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = NFA()
    f.setSigma(["a", "b"])
    f.States = list(range(m))
    f.setInitial([0])
    f.addFinal(m - 1)
    f.addTransition(0, "a", 1)
    f.addTransition(0, "b", 0)
    f.addTransition(m - 1, "a", 0)
    f.addTransition(m - 1, "a", 1)
    for i in range(1, m - 1):
        f.addTransition(i, "a", i + 1)
        f.addTransition(i, "b", i + 1)
    return f


def toDFAWC3(m=5):
    """ Worst case automata for toDFA(NFA) with n > 2, k=3.
    ..seealso:: O. B. Lupanov. A comparison of two types of finite sources.
    Problemy Kibernetiki, 9:321–326, 1963.
    
    :arg m: number of states
    :type m: integer
    :returns: a dfa
    :rtype: DFA"""

    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = NFA()
    f.setSigma(["a", "b", "c"])
    f.States = list(range(m))
    f.setInitial([0])
    f.addFinal(0)
    f.addTransition(0, "a", 1)
    f.addTransition(0, "b", 1)
    f.addTransition(1, "b", 0)
    f.addTransition(1, "c", 0)
    f.addTransition(1, "c", 1)
    f.addTransition(1, "a", 2)
    f.addTransition(m - 1, "a", 0)
    f.addTransition(m - 1, "b", m - 1)
    f.addTransition(m - 1, "c", m - 1)
    for i in range(2, m - 1):
        f.addTransition(i, "a", i + 1)
        f.addTransition(i, "b", i)
        f.addTransition(i, "c", i)
    return f


def reversalWC3M(m=5):
    """ Worst case automata for reversal(DFA) with m > 2, k=3.

    ..seealso:: Boris G. Mirkin. On dual automata. Kibernetika, 2:7–10, 1966.
   
    :arg m: number of states
    :type m: integer
    :returns: a dfa
    :rtype: DFA"""

    if m < 3:
        raise TstError("number of states  must be greater than 2")
    return toDFAWC3(m).reversal()


def starSC(m=5):
    """ Worst case state complexity for star
    :arg m: number of states
    :type m: integer
    :returns: state complexity
    :rtype: integer"""

    if m > 1:
        return 3 * 2 ** (m - 2)
    return 1


def starWC(m=5):
    """ Worst case automata for star(DFA) with m > 2, k=2
    ..seealso:: s. Yu, Q. Zhuang, and K. Salomaa. The state complexities
    of some basic operations on regular languages.
    Theor. Comput. Sci., 125(2):315–328, 1994.

    :arg m: number of states
    :type m: integer
    :returns: a dfa
    :rtype: DFA"""

    if m < 3:
        raise TstError("number of states must be greater than 2")
        # for m=2, L=\{w\in\{a,b\}*| |w|a odd \}
    f = DFA()
    f.setSigma(["a", "b"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(m - 1)
    f.addTransition(0, "a", 1)
    f.addTransition(0, "b", 0)
    for i in range(1, m):
        f.addTransition(i, "a", (i + 1) % m)
        f.addTransition(i, "b", (i + 1) % m)
    return f


def starWCM(m=5):
    """ Worst case automata for star(DFA) with m > 2, k=2

    ..seealso:: A. N. Maslov. Estimates of the number of states of
    finite automata. Dokllady Akademii Nauk SSSR, 194:1266–1268, 1970. 
    
    :arg m: number of states
    :type m: integer
    :returns: a dfa
    :rtype: DFA"""

    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(m - 1)
    f.addTransition(m - 1, "a", 0)
    f.addTransition(m - 1, "b", m - 2)
    f.addTransition(0, "b", 0)
    f.addTransition(0, "a", 1)
    for i in range(1, m - 1):
        f.addTransition(i, "a", (i + 1))
        f.addTransition(i, "b", (i - 1))
    return f


def concatSC(m, n, k=1):
    """Worst case state complecity for concatenation
    :arg m: number of states
    :arg n: number of states
    :arg k: number of letters
    :type m: integer
    :type n: integer
    :type k: integer
    :returns: state compelxity
    :rtype: integer"""

    return m * 2 ** n - k * 2 ** (n - 1)


def concatWCM(m=4, n=4):
    """ Worst case automata for catenation(DFA,DFA) with m,n > 1, k=2,

    ..seealso:: A. N. Maslov. Estimates of the number of states of
    finite automata. Dokllady Akademii Nauk SSSR, 194:1266–1268, 1970. 
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA, DFA)"""

    if n < 2 or m < 2:
        raise TstError("number of states must be both greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(m - 1)
    d1.addTransition(m - 1, "b", 0)
    d1.addTransition(m - 1, "a", m - 1)
    for i in range(m - 1):
        d1.addTransition(i, "a", i)
        d1.addTransition(i, "b", i + 1)
    d2.setSigma(["a", "b"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    d2.addTransition(n - 1, "a", n - 1)
    d2.addTransition(n - 1, "b", n - 2)
    d2.addTransition(n - 2, "b", n - 1)
    d2.addTransition(n - 2, "a", n - 1)
    for i in range(n - 2):
        d2.addTransition(i, "a", i + 1)
        d2.addTransition(i, "b", i)

    return d1, d2


def concatWC(m=6, n=6):
    """ Worst case automata for catenation(DFA,DFA) with m,n > 1
    ..seealso:: s. Yu, Q. Zhuang, and K. Salomaa. The state complexities
    of some basic operations on regular languages.
    Theor. Comput. Sci., 125(2):315–328, 1994.
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA, DFA)"""
    if n < 2 or m < 2:
        raise TstError("number of states must both  greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b", "c"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(m - 1)
    for i in range(m):
        d1.addTransition(i, "a", (i + 1) % m)
        d1.addTransition(i, "b", 0)
        d1.addTransition(i, "c", i)
    d2.setSigma(["a", "b", "c"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    for i in range(n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
        d2.addTransition(i, "c", 1)
    return d1, d2


def concatWCB(m=4, n=4):
    """ Worst case automata for catenation(DFA,DFA) with m,n > 1, k=2,

    ..seealso::Jirásek, J., Jiráaskováa, G., Szabari, A., 2005.
     State complexity of concatenation and complementation of regular
     languages. Int. J. Found. Comput. Sci. 16 (3), 511–529.
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas
    :rtype: (DFA, DFA)"""

    if n < 2 or m < 2:
        raise TstError("number of states must be both greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(m - 1)
    d1.addTransition(m - 1, "b", 0)
    d1.addTransition(m - 1, "a", m - 1)
    for i in range(m - 1):
        d1.addTransition(i, "a", i)
        d1.addTransition(i, "b", i + 1)
    d2.setSigma(["a", "b"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    d2.addTransition(n - 1, "a", 0)
    d2.addTransition(n - 1, "b", 0)
    d2.addTransition(0, "a", 0)
    d2.addTransition(0, "b", 1)
    d2.addTransition(n - 2, "a", n - 1)
    for i in range(1, n - 1):
        d2.addTransition(i, "a", i + 1)
        d2.addTransition(i, "b", i + 1)

    return d1, d2


def interWC(m=6, n=5):
    """ Worst case automata for intersection(DFA,DFA) with m,n >1

    ..seealso:: s. Yu, Q. Zhuang, and K. Salomaa. The state complexities
    of some basic operations on regular languages.
    Theor. Comput. Sci., 125(2):315–328, 1994.
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA, DFA)"""
    if n < 2 or m < 2:
        raise TstError("number of states must be both greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(0)
    for i in range(m):
        d1.addTransition(i, "a", (i + 1) % m)
        d1.addTransition(i, "b", i)
    d2.setSigma(["a", "b"])
    d2.States = list(range(m))
    d2.setInitial(0)
    d2.addFinal(0)
    for i in range(n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
    return d1, d2


def disjWC(m=6, n=5):
    """ Worst case automata for disjunction(DFA,DFA) with m,n >1
    ..seealso:: s. Yu, Q. Zhuang, and K. Salomaa. The state complexities
    of some basic operations on regular languages.
    Theor. Comput. Sci., 125(2):315–328, 1994.
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA, DFA)"""

    if n < 2 or m < 2:
        raise TstError("number of states must be both greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addTransition(0, "a", 1)
    d1.addTransition(0, "b", 0)
    for i in range(1, m):
        d1.addTransition(i, "a", (i + 1) % m)
        d1.addTransition(i, "b", i)
        d1.addFinal(i)
    d2.setSigma(["a", "b"])
    d2.States = list(range(m))
    d2.setInitial(0)
    d2.addTransition(0, "b", 1)
    d2.addTransition(0, "a", 0)
    for i in range(n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
    d2.addFinal(0)
    return d1, d2


def reversalMB(m=8):
    """Worst case automata for reversal(DFA)

    ..seealso:: s. Yu, Q. Zhuang, and K. Salomaa. The state complexities
    of some basic operations on regular languages.
    Theor. Comput. Sci., 125(2):315–328, 1994.
    :arg m: number of states
    :type m: integer
    :returns: a dfa
    :rtype: DFA"""
    if m < 3:
        raise TstError("number of states must be greater than 2")
    d = DFA()
    d.setSigma(["a", "b"])
    d.States = list(range(m))
    d.setInitial(0)
    for i in range(m):
        if i == m - 1:
            d.addTransition(m - 1, "a", 0)
        else:
            d.addTransition(i, "a", i + 1)
        if i == 2:
            d.addTransition(2, "b", 0)
        elif i == 3:
            d.addTransition(3, "b", 2)
        else:
            d.addTransition(i, "b", i)
    return d


def reversalWC3L(m=5):
    """ Worst case automata for reversal(DFA) with m > 2, k=3

    ..seealso:: E. L. Leiss. Succinct representation of regular languages
        by boolean automata ii. Theor. Comput. Sci., 38:133–136, 1985.
    :arg m: number of states
    :type m: integer
    :returns: a dfa
    :rtype: DFA"""

    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b", "c"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(0)
    f.addTransition(0, "b", 1)
    f.addTransition(1, "b", 0)
    f.addTransition(0, "a", 1)
    f.addTransition(1, "a", 2)
    f.addTransition(0, "c", m - 1)
    f.addTransition(1, "c", 1)
    for i in range(2, m):
        f.addTransition(i, "a", (i + 1) % m)
        f.addTransition(i, "b", i)
        f.addTransition(i, "c", i)
    return f


def reversalternaryWC(m=5):
    """Worst case automata for reversal(DFA) ternary alphabet
       
        :arg m: number of states
        :type m: integer
        :returns: a dfa
        :rtype: DFA"""
    if m < 3:
        raise TstError("number of states must be greater than 2")
    d = DFA()
    d.setSigma(["a", "b", "c"])
    d.setInitial(0)
    d.addFinal(0)
    d.States = list(range(m))
    d.addTransition(0, "a", m - 1)
    d.addTransition(0, "c", 0)
    d.addTransition(0, "b", 0)
    d.addTransition(1, "c", m - 1)
    d.addTransition(1, "b", 0)
    d.addTransition(1, "a", 0)
    for i in range(2, m):
        d.addTransition(i, "a", i - 1)
        d.addTransition(i, "c", i - 1)
        d.addTransition(i, "b", i)
    return d


def reversalbinaryWC(m=5):
    """Worst case automata for reversal(DFA) binary
    ..seealso:: G. Jir{\'a}skov{\'a} and J. s\v ebej. Note on Reversal of binary regular languages. Proc. DCFS 2011,
    LNCS 6808, Springer, pp 212-221.
    @arg m: number of states
    @type m: integer
    @returns: a dfa
    @rtype: DFA"""

    if m < 2:
        raise TstError("number of states must be greater than 1")
    d = DFA()
    d.setSigma(["a", "b"])
    d.States = list(range(m))
    d.setInitial(0)
    d.addFinal(m - 1)
    d.addTransition(0, "a", 1)
    d.addTransition(0, "b", 0)
    d.addTransition(1, "b", 0)
    if m == 2:
        d.addTransition(1, "a", 0)
    else:
        d.addTransition(1, "a", 2)
        d.addTransition(2, "a", 0)
        if m == 3:
            d.addTransition(2, "b", 2)
        else:
            d.addTransition(2, "b", 3)
            d.addTransition(3, "b", 2)
            d.addTransition(3, "a", 4)
            d.addTransition(m - 1, "a", 3)
            d.addTransition(m - 1, "b", m - 1)
            for i in range(4, m - 1):
                d.addTransition(i, "a", i + 1)
                d.addTransition(i, "b", i)
    return d


def shuffleWC(m=3, n=3):
    """Worst case automata for CShuffle(DFA,DFA) with m.n>1

    ..seealso::
     C. Campeanu, K. Salomaa, and s. Yu. Tight lower bound for
    the state complexity of CShuffle of regular languages.
    Journal of Automata, Languages and Combinatorics, 7(3):303–310, 2002.

    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA, DFA)"""
    if n < 2 or m < 2:
        raise TstError("number of states must be both greater than 1")
    d1, d2 = DFA(), DFA()
    d1.States = list(range(m))
    d1.setSigma(["a", "b", "c", "d", "f"])
    d1.setInitial(0)
    d1.addFinal(0)
    for i in range(m):
        d1.addTransition(i, "a", (i + 1) % m)
        if i != m - 1:
            d1.addTransition(i, "c", i + 1)
        d1.addTransition(i, "d", i)
        if i != 0:
            d1.addTransition(i, "f", i)
    d2.States = list(range(n))
    d2.setSigma(["a", "b", "c", "d", "f"])
    d2.setInitial(0)
    d2.addFinal(0)
    for i in range(n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "c", i)
        if i != n - 1:
            d2.addTransition(i, "d", i + 1)
        if i != 0:
            d2.addTransition(i, "f", i)
    return d1, d2


def starDisjWC(m=6, n=5):
    """Worst case automata for starDisj(DFA,DFA) with m.n>1

     ..seealso: Arto Salomaa, Kai Salomaa, and Sheng Yu. 'State complexity of
    combined operations'. Theor. Comput. Sci., 383(2-3):140–152, 2007.
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA,DFA)"""

    if n < 2 or m < 2:
        raise TstError("number of states must be both greater than 1")
    d1, d2 = DFA(), DFA()
    d1.States = list(range(m))
    d1.setSigma(["a", "b", "c"])
    d1.setInitial(0)
    d1.addFinal(0)
    for i in range(m):
        d1.addTransition(i, "a", (i + 1) % m)
        d1.addTransition(i, "b", i)
        if i != 0:
            d1.addTransition(i, "c", i)
    d1.addTransition(0, "c", 1)
    d2.States = list(range(n))
    d2.setSigma(["a", "b", "c"])
    d2.setInitial(0)
    d2.addFinal(0)
    for i in range(n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
        if i != 0:
            d2.addTransition(i, "c", i)
    d2.addTransition(0, "c", 1)
    return d1, d2


def starInterBC(m=3, n=3):
    """Bad case automata for starInter(DFA,DFA) with m,n>1
    ..seealso:: Arto Salomaa, Kai Salomaa, and Sheng Yu. 'State complexity of
    combined operations'. Theor. Comput. Sci., 383(2-3):140–152, 2007.
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA,DFA)"""

    if n < 2 or m < 2:
        raise TstError("number of states must be both greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b", "c", "d", "e"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(m - 1)
    for i in range(m):
        d1.addTransition(i, "a", (i + 1) % m)
        d1.addTransition(i, "b", i)
        d1.addTransition(i, "c", i)
        d1.addTransition(i, "d", i)
        d1.addTransition(i, "e", i)
    d2.setSigma(["a", "b", "c", "d", "e"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    for i in range(n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
        d2.addTransition(i, "c", n - 2)
        if i == n - 2:
            d2.addTransition(i, "d", n - 1)
        elif i == n - 1:
            d2.addTransition(i, "d", n - 2)
        else:
            d2.addTransition(i, "d", i)
        if i > n - 4:
            d2.addTransition(i, "e", i)
        else:
            d2.addTransition(i, "e", i + 1)
    return d1, d2


def disjWStarWC(m=6, n=5):
    """
     ..seealso:: Yuan Gao and Sheng Yu. 'State complexity of union and intersection
  combined with star and reversal'. CoRR, abs/1006.3755, 2010.
  :arg m: number of states
  :arg n: number of states
  :type m: integer
  :type n: integer
  :returns: two dfas 
  :rtype: (DFA,DFA)"""

    if n < 3 or m < 3:
        raise TstError("number of states must be greater than 2")
    f1 = DFA()
    f1.setSigma(["a", "b", "c"])
    f1.States = list(range(m))
    f1.setInitial(0)
    f1.addFinal(m - 1)
    f1.addTransition(0, "a", 1)
    f1.addTransition(0, "b", 0)
    f1.addTransition(0, "c", 0)
    for i in range(1, m):
        f1.addTransition(i, "a", (i + 1) % m)
        f1.addTransition(i, "b", (i + 1) % m)
        f1.addTransition(i, "c", i)
    f2 = DFA()
    f2.setSigma(["a", "b", "c"])
    f2.States = list(range(n))
    f2.setInitial(0)
    f2.addFinal(n - 1)
    for i in range(n):
        f2.addTransition(i, "a", i)
        f2.addTransition(i, "b", i)
        f2.addTransition(i, "c", (i + 1) % n)
    return f1, f2


# Worst cases for transition complexity
#     UNION   #

def unionWCTk2(m=6, n=6):
    """ @ worst-case family union where
    @m>=2 and n>=2 and k=2
    ..seealso:: Gao, Y., Salomaa, K., Yu, s.: Transition complexity of
    incomplete dfas. Fundam. Inform.  110(1-4), 143–158 (2011)
    @ the conjecture in this article fails for this family
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA,DFA)"""

    if n < 2 or m < 2:
        raise TstError("number of states must both  greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(0)
    d1.addTransition(m - 1, "a", 0)
    for i in range(0, m - 1):
        d1.addTransition(i, "b", i + 1)
    d2.setSigma(["a", "b"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    d2.addTransition(n - 1, "b", n - 1)
    for i in range(0, n - 1):
        d2.addTransition(i, "a", i + 1)
        d2.addTransition(i, "b", i)
    return d1, d2


def unionWCT2(n=6):
    """ @ worst-case family union where
    @m=1 and n>=2 and k=3
    @ Note that the same happens to m>=2 and n=1
    :arg n: number of states
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA,DFA)"""
    m = 1
    if n < 2:
        raise TstError("number of states must both  greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b", "c"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(0)
    d1.addTransition(0, "b", 0)
    d1.addTransition(0, "c", 0)

    d2.setSigma(["a", "b", "c"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    d2.addTransition(0, "a", 0)
    d2.addTransition(0, "b", 1)
    for i in range(1, n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
        d2.addTransition(i, "c", 1)
    return d1, d2


def unionWCT(m=6, n=6):
    """ @ worst-case family union where
    @m>=2 and n>=2 and k=3
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA,DFA)"""

    if n < 2 or m < 2:
        raise TstError("number of states must both  greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b", "c"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(m - 1)
    d1.addTransition(0, "a", 1)
    d1.addTransition(0, "c", 0)
    for i in range(1, m):
        d1.addTransition(i, "a", (i + 1) % m)
        d1.addTransition(i, "b", 0)
        d1.addTransition(i, "c", i)
    d2.setSigma(["a", "b", "c"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    d2.addTransition(0, "a", 0)
    d2.addTransition(0, "b", 1)
    for i in range(1, n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
        d2.addTransition(i, "c", 1)
    return d1, d2


# CONCAT
def concatWCT2(n=6):
    """ Worst-case family concatenation where m=1 and n>=2 and k=3
    :arg n: number of states
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA,DFA)"""
    m = 1
    if n < 2:
        raise TstError("number of states must both  greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b", "c"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(0)
    d1.addTransition(0, "b", 0)
    d1.addTransition(0, "c", 0)

    d2.setSigma(["a", "b", "c"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    d2.addTransition(0, "a", 0)
    d2.addTransition(0, "b", 1)
    for i in range(1, n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
        d2.addTransition(i, "c", (i + 1) % n)
    return d1, d2


def concatWCT3(m=6):
    """ @ worst-case family concatenation where
    @m>=2 and n=1 and k=3
    :arg m: number of states
    :type m: integer
    :returns: two dfas 
    :rtype: (DFA,DFA)"""
    n = 1
    if m < 2:
        raise TstError("number of states must both  greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b", "c"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(m - 1)
    d1.addTransition(0, "a", 0)
    d1.addTransition(0, "b", 1)
    d1.addTransition(0, "c", 1)
    d1.addTransition(1, "a", 1)
    d1.addTransition(1, "b", 2)
    for i in range(2, m):
        d1.addTransition(i, "b", (i + 1) % m)
        d1.addTransition(i, "c", (i + 1) % m)
        d1.addTransition(i, "a", i)
    d2.setSigma(["a", "b", "c"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(0)
    d2.addTransition(0, "c", 0)
    d2.addTransition(0, "b", 0)

    return d1, d2


def concatWCT(m=6, n=6):
    """ @ worst-case family concatenation where
    @m>=2 and n>=2 and k=3
    :arg m: number of states
    :arg n: number of states
    :type m: integer
    :type n: integer
    :returns: two dfas 
    :rtype: (DFA,DFA)"""

    if n < 2 or m < 2:
        raise TstError("number of states must both  greater than 1")
    d1, d2 = DFA(), DFA()
    d1.setSigma(["a", "b", "c"])
    d1.States = list(range(m))
    d1.setInitial(0)
    d1.addFinal(m - 1)
    d1.addTransition(0, "a", 1)
    d1.addTransition(0, "c", 0)
    for i in range(1, m):
        d1.addTransition(i, "a", (i + 1) % m)
        d1.addTransition(i, "b", 0)
        d1.addTransition(i, "c", i)
    d2.setSigma(["a", "b", "c"])
    d2.States = list(range(n))
    d2.setInitial(0)
    d2.addFinal(n - 1)
    d2.addTransition(0, "a", 0)
    d2.addTransition(0, "b", 1)
    for i in range(1, n):
        d2.addTransition(i, "b", (i + 1) % n)
        d2.addTransition(i, "a", i)
        d2.addTransition(i, "c", 1)
    return d1, d2


# Star
def starWCT(m=5):
    """ Worst-case family star where m>=2 and k=2
    :arg m: number of states
    :type m: integer
    :returns: dfa 
    :rtype: DFA"""
    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(m - 1)
    f.addTransition(0, "a", 1)
    for i in range(1, m):
        f.addTransition(i, "a", (i + 1) % m)
        f.addTransition(i, "b", (i + 1) % m)
    return f


def starWCT1(m=5):
    """ @ worst-case family star where
    @m>=2 and k=2
    :arg m: number of states
    :type m: integer
    :returns: dfa 
    :rtype: DFA"""
    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(m - 1)
    f.addTransition(0, "b", 0)
    f.addTransition(0, "a", 1)
    f.addTransition(m - 2, "a", m - 1)
    f.addTransition(m - 1, "a", 0)
    for i in range(1, m - 2):
        f.addTransition(i, "a", (i + 1) % m)
        f.addTransition(i, "b", (i + 1) % m)
    return f


def universal(n, l=None, Finals=None, dialect=False, d=None):
    """ Universal witness for state compelxity
    :arg int n: number of states
    :arg [str] l: alphabet
    :arg [int] Finals: list of final states
    :arg bool dialect: is it a dialect
    :returns: dfa 
    :rtype: DFA
    """
    if n < 3:
        raise TstError("number of states must be greater than 2")
    u = DFA()
    u.States = list(range(n))
    if l is None:
        l = ("a", "b", "c")
    u.setSigma(list(l))
    u.setInitial(0)
    u.addFinal(n - 1)
    u.addTransition(0, "b", 1)
    u.addTransition(1, "b", 0)
    if "c" in l:
        u.addTransition(n - 1, "c", 0)
    for i in range(n):
        u.addTransition(i, "a", (i + 1) % n)
        if i >= 2:
            u.addTransition(i, "b", i)
        if i != n - 1 and "c" in l:
            u.addTransition(i, "c", i)
    return u


def nCr(n, r):
    import math
    if r > n:
        return 0
    else:
        f = math.factorial
        return f(n) / (f(r) * f(n - r))


def boundarySC(n, k):
    return 4 ** (n - 1) - nCr(n - 1, k - 1) + 2 ** (n - k) * 2 ** (n - 1) - 3 ** (n - k) * 2 ** (k - 1) + 2 ** (
        k - 1) * 2 ** (n - 1) - 3 ** (k - 1) * 2 ** (n - k) + 1


# Don't care automata
def dcMilano1(n):
    """Return the special dcNFA to prove the titness of proposed bound

    .. versionadded:: 0.9.8

    :param n: number of 'columns'
    :type n: int
    :rtype: NFA"""
    new = fa.NFA()
    st = []
    for _ in range(3 * n):
        s = new.addState()
        st.append(s)
    new.setInitial([st[n - 1]])
    for s in range(n):
        new.addTransition(st[n - 1], 'c', st[s])
    for s in range(3):
        for r in range(n - 1):
            new.addTransition(st[(n * s) + r], 'a', st[(n * s) + r + 1])
        new.addTransition(st[(n * s) + n - 1], 'a', st[0])
    for s in range(n):
        new.addTransition(st[s], 'b', st[s + n])
        new.addTransition(st[s + n], 'b', st[s + 2 * n])
        new.addTransition(st[s + 2 * n], 'b', st[s])
    new.addFinal(st[n - 1])
    return new


def dcMilano2(n):
    """Return the special dcNFA to prove the titness of proposed bound

    .. versionadded:: 0.9.8

    :param n: number of 'columns'
    :type n: int
    :rtype: NFA"""
    new = fa.NFA()
    st = []
    for _ in range(3 * n):
        s = new.addState()
        st.append(s)
    for s in range(n):
        new.addInitial(st[s])
    for s in range(3):
        for r in range(n - 1):
            new.addTransition(st[(n * s) + r], 'a', st[(n * s) + r + 1])
        new.addTransition(st[(n * s) + n - 1], 'a', st[0])
    for s in range(n):
        new.addTransition(st[s], 'b', st[s + n])
        new.addTransition(st[s + n], 'b', st[s + 2 * n])
        new.addTransition(st[s + 2 * n], 'b', st[s])
    new.addFinal(st[n - 1])
    return new


# Closure operations
def suffWCe(m=3):
    """Witness for suff(L) when L does not have empty as a quotient

     :rtype: DFA

     ..seealso:
          Janusz A. Brzozowski, Galina Jirásková, Chenglong Zou,
          Quotient Complexity of Closed Languages.
          Theory Comput. Syst. 54(2): 277-292 (2014)

     """
    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(0)
    f.addTransition(0, "a", 1)
    f.addTransition(1, "a", 2)
    f.addTransition(0, "b", 0)
    f.addTransition(1, "b", 0)
    for i in range(2, m):
        f.addTransition(i, "a", (i + 1) % m)
        f.addTransition(i, "b", i)
    return f


def suffWCd(m=3):

    """Witness for suff(L) when L has  empty as a quotient

    :rtype: DFA

    ..seealso: as above
    """
    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(0)
    f.addTransition(0, "a", 1)
    f.addTransition(1, "a", 2)
    f.addTransition(0, "b", m-1)
    f.addTransition(1, "b", 0)
    f.addTransition(m-1, "b", m-1)
    f.addTransition(m-1, "a", m-1)
    for i in range(2, m-1):
        f.addTransition(i, "a", (i + 1) % (m-1))
        f.addTransition(i, "b", i)
    return f


def suffWCsynt(m=3):
    """ Worst case witness for synt of suff(L)

    """
    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b", "c", "d", "e"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(m-1)
    f.addTransition(0, "a", 0)
    f.addTransition(0, "b", 0)
    f.addTransition(0, "c", 0)
    f.addTransition(0, "d", 0)
    f.addTransition(0, "e", 1)
    f.addTransition(1, "a", 2)
    f.addTransition(1, "b", 2)
    f.addTransition(1, "c", 1)
    f.addTransition(1, "d", 1)
    f.addTransition(1, "e", 1)
    f.addTransition(2, "b", 1)
    f.addTransition(2, "e", 1)
    f.addTransition(2, "c", 2)
    f.addTransition(2, "d", 2)
    f.addTransition(2, "a", 3)
    for i in range(3, m-1):
        f.addTransition(i, "a", (i+1) % m)
        f.addTransition(i, "b", i)
        f.addTransition(i, "c", i)
        f.addTransition(i, "d", i)
        f.addTransition(i, "e", 1)
    f.addTransition(m-1, "a", 1)
    f.addTransition(m-1, "c", 1)
    f.addTransition(m-1, "e", 1)
    f.addTransition(m-1, "d", 0)
    return f


def booleanWCSymGrp(m=3):
    """Witness for symmetric group

   :rtype: DFA
   ..seealso: Jason Bell, Janusz A. Brzozowski, Nelma Moreira, Rogério Reis.
    Symmetric Groups and Quotient Complexity of Boolean Operations.
    ICALP (2) 2014: 1-12
    """
    if m < 3:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(0)
    f.addFinal(1)
    f.addTransition(0, "a", 1)
    f.addTransition(1, "a", 0)
    f.addTransition(0, "b", 1)
    f.addTransition(1, "b", 2)
    for i in range(2, m):
        f.addTransition(i, "b", (i + 1) % m)
        f.addTransition(i, "a", i)
    return f


def suffFreeSyntWC(m=5):
    """

    """
    if m < 5:
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a", "b", "c", "d", "e"])
    f.States = list(range(m))
    f.setInitial(0)
    f.addFinal(1)
    f.addTransition(0, "a", m-1)
    f.addTransition(0, 'b', m-1)
    f.addTransition(0, 'c', m-1)
    f.addTransition(0, 'd', m-1)
    f.addTransition(0, 'e', 1)
    f.addTransition(1, "a", 2)
    f.addTransition(1, 'c', 1)
    f.addTransition(1, 'e', m-1)
    f.addTransition(1, 'd', m-1)
    f.addTransition(1, 'b', 2)
    f.addTransition(2, 'b', 1)
    f.addTransition(2, "a", 3)
    f.addTransition(2, "c", 2)
    f.addTransition(2, "e", m-1)
    f.addTransition(2, 'd', 2)
    f.addTransition(1, 'd', m-1)
    f.addTransition(m-2, 'c', 1)
    f.addTransition(m-2, 'a', 1)
    for sym in f.Sigma:
        f.addTransition(m - 1, sym, m-1)
    for i in range(3,  m-1):
        f.addTransition(i, "b", i)
        if i != m-2:
            f.addTransition(i, "c", i)
            f.addTransition(i, "a", (i+1))
        f.addTransition(i, "d", i)
        f.addTransition(i, "e", m-1)
    return f


# Shuffle
def shuffle_ReBC(n=3):
    """ Witness for Shuffle toDFA
    """
    if n < 2:
        raise TstError("at least n=2")
    rs = CStar(CConcat(CAtom('a0'), CAtom('b0')))
    for i in range(1, n):
        rs = Shuffle(rs, CStar(CConcat(CAtom('a' + str(i)), CAtom('b' + str(i)))))
    return rs


def shuffle_ReBCF(n=3):
    """
    :param n:
    :return:RegExp
    """
    if n < 2:
        raise TstError("at least n=2")
    rs = CConcat(CAtom('a0'), CAtom('b0'))
    for i in range(1, n):
        rs = Shuffle(rs, CConcat(CAtom('a' + str(i)), CAtom('b' + str(i))))
    return rs


def shuffle_NFABC(n=3):
    """

    :param  int n:
    :rtype: CAtom
    """
    if n < 2:
        raise TstError("at least n=2")
    ns = CStar(CConcat(CAtom('a0'), CAtom('b0'))).toNFA()
    for i in range(1, n):
        ni = CStar(CConcat(CAtom('a' + str(i)), CAtom('b' + str(i)))).toNFA()
        ns = ns.CShuffle(ni)
    return ns


def shuffle_ReWC(n=4):
    """
    Regular expressions for worst case partial derivative automaton from CShuffle operator
    :param int n:
    :rtype: CAtom
    """
    if n < 2:
        raise TstError("at least n=2")
    rs = CAtom('a0')
    for i in range(1, n):
        rs = Shuffle(rs, CAtom('a' + str(i)))
    return rs


def intersectionReWC(n, sym='a'):
    """
    Regular expressions for "bad" case partial derivative automaton from intersection  operator
    :param n: number of letters
    :param sym: alphabetic symbol
    :return: RegExp
    """
    if n == 0:
        return CEpsilon()
    if n == 1:
        return CStar(CAtom(sym))
    else:
        s1 = set()
        s2 = set()
        for i in range(1, n+1):
            ri = str2sre(sym * i)
            s1.add(CStar(ri))
            s2.add(ri)
        r1 = SDisj(frozenset(s1))
        r2 = SStar(SDisj(frozenset(s2)))
        return SConj(frozenset([r1, r2]))


def intersectionRetoReLB(m=3, n=3, Sigma={"a", "b"}):
    """
    Regular expressions for a lower bound for intersection.
    ...seealso:
        H. Gruber and M. Holzer, Tight Bounds on the Descriptional Complexity of Regular Expressions, DLT 2009
    :param m: alphabetic size
    :param n: alphabetic size
    :param Sigma: alphabet
    :return: regular expression for the intersection
    """
    a = CAtom(list(Sigma)[0])
    b = CAtom(list(Sigma)[1])
    bs = CStar(b)
    r1 = a
    for i in range(m):
        r1 = CConcat(r1, CConcat(bs, a))
    return CStar(CDisj(b, r1))


# (b + ((((a b*) a) b*) a))* & (a + ((((b a*) b) a*) b))*
# reveversible DFA
def revFibonnacci(n):
    """
    Worst-case for reversible DFA
    :param n: number of states
    :return: revdfa
    """
    a = DFA()
    a.setSigma(["a", "b"]),
    for i in range(n):
        a.addState(i)
    a.setInitial(0)
    for i in range(n):
        if i % 2 != 0:
            if i < n - 1:
                a.addTransition(i, 'b', i + 1)
            if i < n - 2:
                a.addTransition(i, 'a', i + 2)
        else:
            if i < n - 2:
                a.addTransition(i, 'b', i + 2)
            if i < n - 1:
                a.addTransition(i, 'a', i + 1)
    a.addFinal(n - 1)
    return a


# Block Languages
def dfa_minus(L):
    """Returns DFA over {a,b} accepting all words except those
        whose length is in the list L
    Args:
        L (list): list of word lengths (integers) each > 0
    Returns:
        DFA:
    ..Note: also allows 0 in L    """
    if len(L) == 0:
        return sigmaStarDFA({"a", "b"})
    m = max(L)
    a = DFA()
    a.setSigma(["a", "b"])
    for i in range(m+2):
        a.addState(i)
    a.setInitial(0)
    for i in range(m+1):
        a.addTransition(i, 'a', i + 1)
        a.addTransition(i, 'b', i + 1)
    a.addTransition(m + 1, 'a', m + 1)
    a.addTransition(m + 1, 'b', m + 1)
    L1 = set(range(m+2))-set(L)
    a.setFinal(L1)
    return a


def dfa_minus_re(L):
    """Returns DFA over {a,b} accepting all words except those
        whose length is in the list L
    Args:
        L (list): list of word lengths (integers) each > 0
    Returns:
        DFA:  """
    if len(L) == 0:
        return str2regexp('(a+b)*').toDFA()
    rall = ''
    for l in L:
        r = ''
        for i in range(l):
            r = r + '(a+b)'
        if rall == '':
            rall = r
        else:
            rall = rall + '+' + r
    d = str2regexp(rall).toDFA()
    return ~d


def dfa_minus2(L, F) -> DFA:
    """Returns DFA over {a,b} accepting all words except those whose
        length is one of the L[i]'s and their first F[i] letters are 'a'
    Args:
        L (list): list of word lengths (integers) each > 0
        F (list): list of lengths (integers)
    Returns:
        DFA:  """
    n = len(L)
    if n == 0:
        return sigmaStarDFA({'a', 'b'})
    if n != len(F):
        print('dfa_minus(L,F) error: L,F have unequal lengths')
        return None
    rall = CEmptySet()
    a1 = CAtom('a')
    b1 = CAtom('b')
    d1 = CDisj(a1, b1)
    for i in range(n):
        r = CEpsilon()
        for l in range(L[i]):
            if l < F[i]:
                c =  a1
            else:
                c =  d1
            if r == CEpsilon():
                r = c
            else:
                r = CConcat(r, c)
        if rall == CEmptySet():
            rall = r
        else:
            rall = CDisj(rall, r)
    # print(rall)
    d = rall.toDFA()
    d.Sigma = {'a', 'b'}
    return ~d


def dfa_minus2_re(L, F):
    """Returns DFA over {a,b} accepting all words except those whose
        length is one of the L[i]'s and their first F[i] letters are 'a'
    Args:
        L (list): list of word lengths (integers) each > 0
        F (list): list of lengths (integers)
    Returns:
        DFA:  """
    n = len(L)
    if n == 0:
        return str2regexp('(a+b)*').toDFA()
    if n != len(F):
        print('dfa_minus(L,F) error: L,F have unequal lengths')
        return None
    rall = ''
    for i in range(n):
        r = ''
        for l in range(L[i]):
            if l < F[i]:
                r = r + 'a'
            else:
                r = r + '(a+b)'
        if rall == '':
            rall = r
        else:
            rall = rall + '+' + r
    # print(rall)
    d = str2regexp(rall).toDFA()
    d.Sigma = {'a', 'b'}
    return ~d


def dfa_maxlen_fixed_end(m, k) -> DFA:
    """Returns DFA over {a,b} accepting all words of length <= m such
        that, for the lengths m-k+1,..,m-k+k, the last letter is 'b'
    Args:
        m (int): max length of the accepted words
        k (int):
    Returns:
        DFA:  """
    if m == 0:
        return epsilonDFA({'a', 'b'})
    if k > m:
        k = m
    new = DFA()
    new.setSigma(['a', 'b'])
    s = new.addState()
    new.setInitial(s)
    new.addFinal(s)
    for i in range(1, m - k + 1):
        s1 = new.addState()
        new.addFinal(s1)
        new.addTransition(s, 'a', s1)
        new.addTransition(s, 'b', s1)
        s = s1
    for i in range(m - k + 1, m + 1):
        s1 = new.addState()
        new.addFinal(s1)
        new.addTransition(s, 'b', s1)
        s = s1
    return new


def dfa_maxlen_fixed_end_re(m, k):
    """Returns DFA over {a,b} accepting all words of length <= m such
        that, for the lengths m-k+1,..,m-k+k, the last letter is 'b'
    Args:
        m (int): max length of the accepted words
        k (int):
    Returns:
        DFA:  """
    if m == 0:
        return str2regexp('@epsilon', sigma=['a', 'b']).toDFA()
    if k > m:
        k = m
    rall = ''
    for i in range(m):
        if i < m-k:
            rall = rall + '(@epsilon+a+b)'
        else:
            rall = rall + '(@epsilon+b)'
    d = str2regexp(rall).toDFA()
    d.setSigma(['a', 'b'])
    return d


def dfa_minlen_fixed_prefix(m, k) -> DFA:
    """Returns DFA over {a,b} accepting all words of length >= m
    whose first k letters are equal to 'b'
    Args:
        m (int): min length of the words of the DFA to be constructed
        k (int): the number of b's at the beginning of the accepted words
    Returns:
        DFA:  """
    sigma = {'a', 'b'}
    if m == 0:
        return epsilonDFA(sigma)
    if k > m:
        d1 = sigmaInitialSegment({'b'},m,True)
        d1.setSigma(sigma)
    else:
        d0 = sigmaInitialSegment({'b'}, k, True)
        d0.setSigma(sigma)
        d1 = d0.concat(sigmaInitialSegment(sigma, m - k, True))
    return d1.concat(sigmaStarDFA(sigma))


def dfa_minlen_fixed_prefix_re(m, k):
    """Returns DFA over {a,b} accepting all words of length >= m
    whose first k letters are equal to 'b'
    Args:
        m (int): min length of the words of the DFA to be constructed
        k (int): the number of b's at the beginning of the accepted words
    Returns:
        DFA:  """
    sigma = {'a', 'b'}
    if m == 0:
        return epsilonDFA(sigma)
    if k > m:
        k = m
    rall = ''
    for i in range(m):
        if i < k:
            rall = rall + 'b'
        else:
            rall = rall + '(a+b)'
    rall = rall + '(a+b)*'
    d = str2regexp(rall, sigma=sigma).toDFA()
    return d


def dfa_block_fixed_end(m, k) -> DFA:
    """Returns block DFA over {a,b} of word length m accepting all
    words of length = m that end with k  b's .
    Args:
        m (int): block length of the DFA to be constructed
        k (int): the number of b's at the end'
    Returns:
        DFA:  """
    if m == 0:
        return epsilonDFA(["a", "b"])
    if k > m:
        k = m
    a = DFA()
    a.setSigma(["a", "b"])
    for i in range(m + 1):
        a.addState(i)
    a.setInitial(0)
    for i in range(m - k):
        a.addTransition(i, 'a', i + 1)
        a.addTransition(i, 'b', i + 1)
    for i in range(m - k, m):
        a.addTransition(i, 'b', i + 1)
    a.addFinal(m)
    return a


def dfa_block_fixed_end_re(m, k):
    """Returns block DFA over {a,b} of word length m accepting all
    words of length = m that end with k  b's .
    Args:
        m (int): block length of the DFA to be constructed
        k (int): the number of b's at the end'
    Returns:
        DFA:  """
    if m == 0:
        return str2regexp('@epsilon', sigma=["a", "b"]).toDFA()
    if k > m:
        k = m
    rall = ''
    for i in range(m):
        if i < m-k:
            rall = rall + '(a+b)'
        else:
            rall = rall + 'b'
    d = str2regexp(rall).toDFA()
    d.setSigma(["a", "b"])
    return d


def nfa_with_exponential_dfa(m) -> NFA:
    """Returns NFA accepting all words in (a+b)*a(a+b)^m
    Args:
        m (int): the parameter >= 0
    Returns:
        NFA:  """
    a1 = CAtom('a')
    b1 = CAtom('b')
    d1 = CDisj(a1, b1)
    s1 = CStar(d1)
    r1 = CConcat(s1, a1)
    for i in range(m):
        r1 = CConcat(r1, d1)
    return r1.toNFA()


def nfa_with_exponential_dfa_re(m) -> NFA:
    """Returns NFA accepting all words in (a+b)*a(a+b)^m
    Args:
        m (int): the parameter >= 0
    Returns:
        NFA:  """
    rall = '(a+b)*a'
    for i in range(m):
        rall = rall + '(a+b)'
    a = str2regexp(rall).toNFA()
    return a


def nfa_with_exponential_dfa2(m) -> NFA:
    """Returns NFA accepting all words in (a+b)*a(a+b)^m + (a+b)^{<=m}
        Note that if D is a word distribution then the universality index
        of the NFA is  0.5+0.5*D(Len<=m)
    Args:
        m (int): the parameter >= 0
    Returns:
        DFA:  """
    a1 = CAtom('a')
    b1 = CAtom('b')
    d1 = CDisj(a1, b1)
    s1 = CStar(d1)
    r1 = CConcat(s1, a1)
    d2 = CDisj(CEpsilon(), d1)
    r2 = CEpsilon()
    for i in range(m):
        r1 = CConcat(r1, d1)
        r2 = CConcat(r2, d2)
    a = CDisj(r1, r2)
    return a.toNFA()


def nfa_with_exponential_dfa2_re(m):
    """Returns NFA accepting all words in (a+b)*a(a+b)^m + (a+b)^{<=m}
        Note that if D is a word distribution then the universality index
        of the NFA is  0.5+0.5*D(Len<=m)
    Args:
        m (int): the parameter >= 0
    Returns:
        DFA:  """
    r1 = '(a+b)*a'
    if m > 0:
        r2 = '+'
    else:
        r2 = '+@epsilon'
    for i in range(m):
        r1 += '(a+b)'
        r2 += '(@epsilon+a+b)'
    a = str2regexp(r1 + r2).toNFA()
    return a

def dfa_with_exponential_rev(m) -> DFA:
    """Returns DFA accepting all words in (a+b)^{m-1} a (@epsilon+a+b)^m
    Args:
        m (int): should be > 0
    Returns:
        DFA:  """
    if m < 1:
        return None
    sigma = {'a', 'b'}
    da  = symbolDFA('a', sigma)
    db = symbolDFA('b', sigma)
    if m == 1:
        return da.concat(da.disj(db.disj(epsilonDFA(sigma))))
    d1 = sigmaInitialSegment(sigma, m - 1, True)
    d2 = d1.concat(da)
    d3 = d2.concat(sigmaInitialSegment(sigma, m))
    return d3

def dfa_with_exponential_rev_re(m):
    """Returns DFA accepting all words in (a+b)^{m-1} a (@epsilon+a+b)^m
    Args:
        m (int): should be > 0
    Returns:
        DFA:  """
    r1 = ''
    r2 = 'a'
    for i in range(m-1):
        r1 += '(a+b)'
    for i in range(m):
        r2 += '(@epsilon+a+b)'
    a = str2regexp(r1+r2).toNFA()
    return a.toDFA().trim()

def dfa_for_rev(m) -> DFA:
    """Returns DFA accepting all words in (a+b)^{m-1} a (@epsilon+a+b)^m
    Args:
        m (int): should be > 0
    Returns:
        DFA:  """
    if m < 1:
        return None
    sigma = {'a', 'b'}
    if m  == 1:
        d2 = symbolDFA('a', sigma)
        d3 = d2.concat(d2.disj(symbolDFA('b', sigma)))
    else:
        d1 = sigmaInitialSegment(sigma, m - 1, True)
        d2 = d1.concat(symbolDFA('a', sigma))
        d3 = d2.concat(sigmaInitialSegment(sigma, m, True))
    return d3


def dfa_for_rev_re(m):
    """Returns NFA accepting all words in (a+b)^{m-1} a (@epsilon+a+b)^m
    Args:
        m (int): should be > 0
    Returns:
        DFA:  """
    if m < 1:
        return None
    sigma = {'a', 'b'}
    r1 = ''
    r2 = 'a('
    for i in range(m-1):
        r1 += '(a+b)'
    for i in range(m+1):
        r3 = ''
        r4 = ''
        for j in range(m-i):
            r3 += 'a'
        for j in range(i):
            r4 += '(a+b)'
        if i < m:
            r2 += r3+r4+'+'
        else:
            r2 += r3+r4
    a = str2regexp(r1+r2+')', sigma=sigma ).toNFA()
    return a.toDFA().trim()


def dfa_exponential(m) -> DFA:
    """Returns DFA accepting all words in (a+b)*a(a+b)^m
    Args:
        m (int): the parameter
    Returns:
        DFA:  """
    sigma = {'a', 'b'}
    d1 = sigmaStarDFA(sigma)
    d2 = d1.concat(symbolDFA('a', sigma))
    d3 = d2.concat(sigmaInitialSegment(sigma, m, True))
    return d3


def dfa_exponential_re(m):
    """Returns DFA accepting all words in (a+b)*a(a+b)^m
    Args:
        m (int): the parameter
    Returns:
        DFA:  """
    sigma = {'a', 'b'}
    rall = '(a+b)*a'
    for i in range(m):
        rall = rall + '(a+b)'
    d = str2regexp(rall, sigma =sigma).toDFA()
    return d


def dfa_block_with_unive_index(m, eps) -> DFA:
    """Returns block DFA over {a,b} of word length m whose block
        universality index is less than, but very close to, 1-eps
        (in fact the difference (1-eps)-(block univ index) <= 1/2**m)
        The main idea is to construct a block DFA d accepting x
        words, where x = int(1+floor(2**m*eps)).
        Then the required DFA is one that accepts all words of length m other
        than the x ones.
        The block universality index of the returned DFA  is  1- x/2**m
        The block DFA d corresponds to the reg. expr. rall defined as follows:
        Suppose  x = 2**(yt)+...+2**(y0). Then, rall is the sum, for i=t,..,0,
        of the reg. expr. r accepting the words ua^ib^(m-yi-i), where |u|=yi
    Args:
        m (int): block length of the DFA to be constructed
        eps (float): approximation tolerance
    Returns:
        DFA: """
    from math import floor, log
    sigma = {'a', 'b'}
    if m == 0:
        return emptyDFA(sigma)
    x = int(1+floor(2**m*eps))
    L = []
    y = int(floor(log(x, 2)))
    while True:
        L += [y]
        next_x = x-2**y
        if next_x == 0:
            break
        x = next_x
        y = int(floor(log(x, 2)))
    lenL = len(L)
    a1 = CAtom('a')
    b1 = CAtom('b')
    d1 = CDisj(a1, b1)
    rall = CEpsilon()

    for j in range(1, lenL + 1):
        r = CEpsilon()
        k = m - L[lenL - j]
        for i in range(m):
            if i < m - k:
                c = d1
            elif i == m-k:
                c = a1
            else:
                c = b1
            if r == CEpsilon():
                r = c
            else:
                r = CConcat(r,c)
        if j == 1:
            rall = r
        else:
            rall = CDisj(rall, r, sigma=sigma)
    d = rall.toDFA()
    return coBlockDFA(d, m)


def dfa_block_with_unive_index_re(m, eps):
    """Returns block DFA over {a,b} of word length m whose block
        universality index is less than, but very close to, 1-eps
        (in fact the difference (1-eps)-(block univ index) <= 1/2**m)
        The main idea is to construct a block DFA d accepting x
        words, where x = int(1+floor(2**m*eps)).
        Then the required DFA is one that accepts all words of length m other
        than the x ones.
        The block universality index of the returned DFA  is  1- x/2**m
        The block DFA d corresponds to the reg. expr. rall defined as follows:
        Suppose  x = 2**(yt)+...+2**(y0). Then, rall is the sum, for i=t,..,0,
        of the reg. expr. r accepting the words ua^ib^(m-yi-i), where |u|=yi
    Args:
        m (int): block length of the DFA to be constructed
        eps (float): approximation tolerance
    Returns:
        DFA: """
    from math import floor, log
    sigma = {'a', 'b'}
    if m == 0:
        return emptyDFA(sigma)
    x = int(1+floor(2**m*eps))
    L = []
    y = int(floor(log(x, 2)))
    while True:
        L += [y]
        next_x = x-2**y
        if next_x == 0:
            break
        x = next_x
        y = int(floor(log(x, 2)))
    lenL = len(L)
    rall = ''
    maxk = m-L[lenL-1]
    for j in range(1, lenL + 1):
        r = ''
        k = m - L[lenL - j]
        for i in range(m):
            if i < m - k:
                r = r + '(a+b)'
            elif i == m-k:
                r = r + 'a'
            else:
                r = r + 'b'
        if j == 1:
            rall = r
        else:
            rall = rall + '+' + r
    d = str2regexp(rall, sigma=sigma).toDFA()
    return coBlockDFA(d, m)


def approx_unive_index(m, eps):
    """This should be understood in connection with the above function.
        It returns a universality index close to, but strictly less than, 1-eps
    Args:
        m (int): block length of the DFA to be constructed
        eps (float): approximation tolerance
    Returns:
        float:  """
    from math import floor
    x = int(1+floor(2**m*eps))
    return 1.0-x/2**m


def dfa_block_fixed_end3(m, k, c):
    """Returns block DFA over {a,b} accepting all
    words of length = m that end with k  c's .
    Args:
        m (int): block length of the DFA to be constructed
        k (int): the number of b's at the end'
        c (str): character
    Returns:
        DFA:  """
    sigma = {'a', 'b'}
    if m == 0:
        return epsilonDFA(sigma)
    d0 = symbolDFA(c, sigma)
    if k > m:
        d = d0.star()
        d.setSigma(sigma)
        return d
    if k == 0:
        return sigmaInitialSegment(sigma, m , True)
    d1 = sigmaInitialSegment(sigma, m - k , True)
    d2 = sigmaInitialSegment({c}, k , True)
    d2.setSigma(sigma)
    d = d1.concat(d2)
    return d


def dfa_block_fixed_end3_re(m, k, c):
    """Returns block DFA over {a,b} accepting all
    words of length = m that end with k  c's .
    Args:
        m (int): block length of the DFA to be constructed
        k (int): the number of b's at the end'
        c (str): character
    Returns:
        DFA:  """
    if m == 0:
        return str2regexp('@epsilon').toDFA()
    if k > m:
        k = m
    rall = ''
    for i in range(m):
        if i < m-k:
            rall = rall + '(a+b)'
        else:
            rall = rall + c
    d = str2regexp(rall).toDFA()
    return d


def nfa_KH(sigma, d, x) -> NFA:
    """Returns an NFA corresponding to L_n,a as
    introduced in Karhumaki & Okhotin paper
    Args:
        sigma (set): alphabet of the language
        d (int): half the size of words
        x (str): forbidden letter
    Returns:
        NFA:

    ..seealso:
          Juhani Karhumaki and Alexander Okhotin,
          On the Determinization Blowup for Finite
          Automata Recognizing Equal-Length Languages.
          LNCS 8808, pp. 71–82, 2014
    """
    fa: NFA = NFA()
    fa.setSigma(sigma)
    sigma0: set = fa.Sigma - {x}
    sti1 = fa.stateIndex((0, None, None, False), True)
    fa.addInitial(sti1)
    for n in range(d):
        if n < d - 1:
            sti1 = fa.stateIndex((n, None, None, False), True)
            sti2 = fa.stateIndex((n + 1, None, None, False), True)
            fa.addTransitionStar(sti1, sti2)
        for c in sigma0:
            sti1 = fa.stateIndex((n, None, None, False), True)
            sti2 = fa.stateIndex((n + 1, n, c, False), True)
            fa.addTransition(sti1, c, sti2)
    for m in range(d):
        for n in range(m+1, m+d):
            for c in sigma0:
                sti1 = fa.stateIndex((n, m, c, False), True)
                sti2 = fa.stateIndex((n + 1, m, c, False), True)
                fa.addTransitionStar(sti1, sti2)
    for c in sigma0:
        for m in range(d):
            sti1 = fa.stateIndex((m + d, m, c, False), True)
            sti2 = fa.stateIndex((m + d + 1, None, None, True), True)
            fa.addTransition(sti1, c, sti2)
    for n in range(d + 1, 2 * d):
        sti1 = fa.stateIndex((n, None, None, True), True)
        sti2 = fa.stateIndex((n + 1, None, None, True), True)
        fa.addTransitionStar(sti1, sti2)
    fa.addFinal(fa.stateIndex((2*d, None, None, True)))
    return fa
