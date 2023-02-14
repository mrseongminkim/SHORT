# -*- coding: utf-8 -*-
"""**In/Out.**

FAdo I/O methods. The parsing grammars for most of the objects reside here.

.. *Authors:* Rogério Reis & Nelma Moreira

.. *This is part of FAdo project*   https://fado.dcc.fc.up.pt.

.. *Copyright:* 2014-2022 Rogério Reis & Nelma Moreira {rvr,nam}@dcc.fc.up.pt

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

import io
from . import common
from . fa import DFA, NFA, statePP
from . transducers import SFT, GFT, Transducer
from . sst import PSPVanila, PSPEqual, PSPDiff, SSOneOf, SSAnyOf, SSNoneOf, SSEpsilon, SST, SSFA
from . fl import ADFA
import lark


def readOneFromFile(fileName):
    """ Read the first of the FAdo objects from File

    :param fileName: name of the file
    :type fileName: str
    :rtype: DFA|FA|STF|SST"""
    o = readFromFile(fileName)
    if type(o) == list:
        return o[0]
    else:
        return o


def readFromFile(FileName):
    """Reads list of finite automata definition from a file.

    :param str FileName: file name
    :rtype: list

    The format of these files must be the as simple as possible:

    .. hlist::
       :columns: 1

       * ``#`` begins a comment
       * ``@DFA`` or ``@NFA`` begin a new automata (and determines its type) and must be followed by the list of the
         final states separated by blanks
       * fields are separated by a blank and transitions by a CR: ``state`` ``symbol`` ``new state``
       * in case of a NFA declaration, the "symbol" @epsilon is interpreted as an epsilon-transition
       * the source state of the first transition is the initial state
       * in the case of a NFA, its declaration ``@NFA``  can, after the declaration of the final states,
         have a ``*`` followed by the list of initial states
       * both, NFA and DFA, may have a declaration of alphabet starting with a ``$`` followed by the symbols of the
         alphabet
       * a line with a sigle name, decrares a state

    .. productionlist:: Fado Format
       FAdo: FA | FA CR FAdo
       FA: DFA | NFA | Transducer
       DFA: "@DFA" LsStates Alphabet CR dTrans
       NFA: "@NFA" LsStates Initials Alphabet CR nTrans
       Transducer: "@Transducer" LsStates Initials Alphabet Output CR tTrans
       Initials: "*" LsStates | /Epsilon
       Alphabet: "$" LsSymbols | /Epsilon
       Output: "$" LsSymbols | /Epsilon
       nSymbol: symbol | "@epsilon"
       LsStates: stateid | stateid , LsStates
       LsSymbols: symbol | symbol , LsSymbols
       dTrans: stateid symbol stateid |
        :| stateid symbol stateid CR dTrans
       nTrans: stateid nSymbol stateid |
        :| stateid nSymbol stateid CR nTrans
       tTrans: stateid nSymbol nSymbol stateid |
        :| stateid nSymbol nSymbol stateid CR nTrans
    .. note::
       If an error occur, either syntactic or because of a violation of the declared automata type,
       an exception is raised

    .. versionchanged:: 0.9.6
    .. versionchanged:: 1.0"""

    with open(FileName, "r") as file:
        s =file.read()
    # f.close()
    tree = FAdoGrammar.parse(s)
    return BuildFadoObject().transform(tree)


def readOneFromString(s):
    """Reads one finite automata definition from a file.

    .. seealso::
        readFromFile for description of format

    :param str s: the string
    :rtype: DFA|NFA|SFT"""
    tree = FAdoGrammar.parse(s)
    return BuildFadoObject().transform(tree)


def alphabetPP(sigma):
    ssig = "[ \"{0:>s}\"".format(str(sigma.pop()))
    for sym in sigma:
        ssig += ", \"{0:>s}\"".format(str(sym))
    ssig += " ]"
    return ssig


def toJson(aut):
    """ Json for a FA

    :param FA aut: the automaton
    :rtype: str
    """
    ioc = io.StringIO()
    if isinstance(aut, DFA):
        jtype = "DFA"
    elif isinstance(aut, NFA):
        jtype = "NFA"
    elif isinstance(aut, Transducer):
        jtype = "Transducer"
    elif isinstance(aut, ADFA):
        jtype = "ADFA"
    ioc.write("{ \"automaton\": {\n\t\"title\": \"\", \n\t\"version\": \"\",\n")
    # noinspection PyUnboundLocalVariable
    ioc.write("\t\"type\": \"{0:>s}\",\n".format(jtype))
    ioc.write("\t\"states\": [\n")
    sn = 0
    for s in range(len(aut.States)):
        if sn == 0:
            ioc.write("{ \n")
        else:
            ioc.write(",\n{ \n")
        ioc.write("\t\t\"name\": \"{0:>s}\",\n".format(str(s)))
        ioc.write("\t\t\"label\": \"{0:>s}\",\n".format(statePP(aut.States[s])))
        ioc.write("\t\t\"output\": \"\",\n")
        if aut.initialP(s):
            ioc.write("\t\t\"initial\": true,\n")
        else:
            ioc.write("\t\t\"initial\":false,\n")
        if aut.finalP(s):
            ioc.write("\t\t\"final\": true\n")
        else:
            ioc.write("\t\t\"final\": false\n")
        ioc.write("}")
        sn += 1
    ioc.write("], \n")
    ioc.write("\t\"trans\": [\n")
    trn = 0
    for s in range(len(aut.States)):
        if s in aut.delta:
            for a in list(aut.delta[s].keys()):
                if isinstance(aut.delta[s][a], set):
                    for s1 in aut.delta[s][a]:
                        if trn == 0:
                            ioc.write("{ \n")
                        else:
                            ioc.write(",\n{ \n")
                        ioc.write("\t\t\"name\": \"{0:>s}\\,\n".format(str(trn)))
                        ioc.write("\t\t\"orig_name\": \"{0:>s}\\,\n".format(statePP(aut.States[s])))
                        ioc.write("\t\t\"dest_name\": \"{0:>s}\",\n".format(statePP(aut.States[s1])))
                        ioc.write("\t\t\"label\": \"{0:>s}\",\n".format(str(a))),
                        ioc.write("\t\t\"weight\": \"\"\n")
                        ioc.write("}")
                        trn += 1
                else:
                    if trn == 0:
                        ioc.write("{ \n")
                    else:
                        ioc.write(",\n{ \n")
                    # io.write(", \n{ \n")
                    ioc.write("\t\t\"name\": \"{0:>s}\",\n".format(str(trn)))
                    ioc.write("\t\t\"orig_name\": \"{0:>s}\",\n".format(statePP(aut.States[s])))
                    ioc.write("\t\t\"dest_name\": \"{0:>s}\",\n".format(statePP(aut.States[aut.delta[s][a]])))
                    ioc.write("\t\t\"label\": \"{0:>s}\",\n".format(str(a))),
                    ioc.write("\t\t\"weight\": \"\"\n")
                    ioc.write("}")
                    trn += 1
    ioc.write("],\n")
    ioc.write("\t\"alphabet\": {0:>s} \n".format(alphabetPP(aut.Sigma)))
    ioc.write(" } ")
    return ioc.getvalue()


def saveToJson(FileName, aut, mode="w"):
    """ Saves a finite automata definition to a file using the JSON format
    """
    try:
        f = open(FileName, mode)
    except IOError:
        raise common.DFAerror()
    f.write(toJson(aut))
    f.close()


def saveToString(fa):
    """ Saves a finite automaton definition to a string
    :param fa: automaton
    :return: the string containing the automaton definition
    :rtype: str

    ..versionadded:: 1.2.1"""

    def _save_SFTransducer(tr, ioc):
        ioc.write("@Transducer ")
        for s in tr.Final:
            ioc.write("{0:>s} ".format(statePP(tr.States[s])))
        ioc.write("* ")
        for s in tr.Initial:
            ioc.write("{0:>s} ".format(statePP(fa.States[s])))
        ioc.write("\n")
        for sin in tr.delta:
            for syin in tr.delta[sin]:
                for (syout, sout) in tr.delta[sin][syin]:
                    ioc.write("{0:>s} {1:>s} {2:>s} {3:>s}\n".format(statePP(tr.States[sin]), str(syin), str(syout),
                                                                     statePP(tr.States[sout])))
        ioc.write("\n")

    def _saveFA(aut, ioc):
        if isinstance(aut, DFA):
            ioc.write("@DFA ")
            NFAp = False
        elif isinstance(aut, NFA):
            ioc.write("@NFA ")
            NFAp = True
        else:
            raise common.DFAerror()
        if not NFAp and aut.Initial != 0:
            foo = {0: aut.Initial, aut.Initial: 0}
            aut.reorder(foo)
        for sf in aut.Final:
            ioc.write("{0:>s} ".format(statePP(aut.States[sf])))
        if NFAp:
            ioc.write(" * ")
            for sf in aut.Initial:
                ioc.write("{0:>s} ".format(statePP(aut.States[sf])))
        ioc.write("\n")
        for s in range(len(aut.States)):
            if s in aut.delta:
                for a in list(aut.delta[s].keys()):
                    if isinstance(aut.delta[s][a], set):
                        for s1 in aut.delta[s][a]:
                            ioc.write("{0:>s} {1:>s} {2:>}\n".format(statePP(aut.States[s]),
                                                                     str(a), statePP(aut.States[s1])))
                    else:
                        ioc.write("{0:>s} {1:>s} {2:>s}\n".format(statePP(aut.States[s]), str(a),
                                                                  statePP(aut.States[aut.delta[s][a]])))
            else:
                ioc.write("{0:>s} \n".format(statePP(aut.States[s])))

    out = io.StringIO()
    if isinstance(fa, Transducer):
        _save_SFTransducer(fa, out)
        return out.getvalue()
    else:
        _saveFA(fa, out)
        return out.getvalue()


def saveToFile(FileName, fa, mode="a"):
    """ Saves a list finite automata definition to a file using the input format

    .. versionchanged:: 0.9.5
    .. versionchanged:: 0.9.6
    .. versionchanged:: 0.9.7 New format with quotes and alphabet

    :param str FileName: file name
    :param fa: the FA
    :type fa: list of FA
    :param str mode: writing mode"""

    # TODO: write the complete information into file according with the new format
    try:
        f = open(FileName, mode)
    except IOError:
        raise common.DFAerror()
    if type(fa) == list:
        for d in fa:
            f.write(saveToString(d))
    else:
        f.write(saveToString(fa))
    f.close()


def _exportToTeX(FileName, fa):
    """ Saves a finite automatom definition to a latex tabular. Saves a finite automata definition to a file using
    the input format

    .. versionchanged:: 0.9.4

    :param str FileName: file name
    :param FA fa: the FA
    :raises DFAerror: if a file error occurs"""
    try:
        f = open(FileName, "w")
    except IOError:
        raise common.DFAerror()
        # initial is the first one
    if fa.Initial:
        foo = {0: fa.Initial, fa.Initial: 0}
        fa.reorder(foo)
    f.write("$$\\begin{array}{r|")
    for i in range(len(fa.Sigma)):
        f.write("|c")
    f.write("}\n")
    for c in fa.Sigma:
        f.write("&{0:>s}".format(str(c)))
    f.write(" \\\\\\hline\n")
    for s in range(len(fa.States)):
        if s in fa.delta:
            if fa.Initial == s:
                f.write("\\rightarrow")
            if s in fa.Final:
                f.write("\\star")
            f.write("{0:>s}".format(str(s)))
            for a in list(fa.delta[s].keys()):
                if isinstance(fa.delta[s][a], set):
                    f.write("&\\{")
                    for s1 in fa.delta[s][a]:
                        f.write("{0:>s} ".format(str(s1)))
                    f.write("\\}")
                else:
                    s1 = fa.delta[s][a]
                    f.write("&{0:>s}".format(str(s1)))
            f.write("\\\\\n")
    f.write("\\end{array}$$")
    f.close()


def show(obj):
    """ General, context sensitive, display method
    :param obj: the object to show

    .. versionadded:: 1.2.1 """
    pass


class BuildFadoObject(lark.Transformer):
    """ Semantics of the FAdo grammars' objects """

    @staticmethod
    def object(s):
        return s

    def dfa(self, s):
        f = DFA()
        if "AlphabetI" in s:
            for x in self.AlphabetI:
                f.addSigma(x)
        initial = True
        for t in self.Transitions:
            if t is None:
                continue
            if len(t) == 3:
                i0 = f.stateIndex(t[0], True)
                i1 = f.stateIndex(t[2], True)
                if t[1] not in f.Sigma:
                    f.addSigma(t[1])
                f.addTransition(i0, t[1], i1)
                if initial:
                    f.setInitial(i0)
                    initial = False
        if "Finals" in s:
            for x in self.Finals:
                f.addFinal(f.stateIndex(x, True))
        if "declaredStates" in s:
            for x in self.DFADStates:
                f.addSigma(x)
        return f

    def nfa(self, s):
        f = NFA()
        if "AlphabetI" in s:
            for x in self.AlphabetI:
                f.addSigma(x)
        if "Initials" in s:
            initial = False
            for x in self.Initials:
                i = f.stateIndex(x, True)
                f.addInitial(i)
        else:
            initial = True
        for t in self.Transitions:
            if t is None:
                continue
            if len(t) == 3:
                i0 = f.stateIndex(t[0], True)
                i1 = f.stateIndex(t[2], True)
                if t[1] not in f.Sigma and t[1] != common.Epsilon:
                    f.addSigma(t[1])
                f.addTransition(i0, t[1], i1)
                if initial:
                    f.addInitial(i0)
                    initial = False
        if "Finals" in s:
            for x in self.Finals:
                f.addFinal(f.stateIndex(x, True))
        return f

    def transducer(self, s):
        tt = []
        gft = False
        for t in self.Transitions:
            if t is None:
                continue
            tt.append(t)
            if len(t[1]) > 1 and t[1] != common.Epsilon and not gft:
                gft = True
            if len(t[2]) > 1 and t[2] != common.Epsilon and not gft:
                gft = True
        if gft is True:
            f = GFT()
        else:
            f = SFT()
        if "AlphabetI" in s:
            if self.AlphabetI is not None:
                for x in self.AlphabetI:
                    f.addSigma(x)
        if "AlphabetO" in s:
            for x in self.AlphabetO:
                f.addOutput(x)
        if "Initials" in s:
            initial = False
            for x in self.Initials:
                i = f.stateIndex(x, True)
                f.addInitial(i)
        else:
            initial = True
        for t in tt:
            i0 = f.stateIndex(t[0], True)
            i1 = f.stateIndex(t[3], True)
            f.addTransition(i0, t[1], t[2], i1)
            if initial:
                f.addInitial(i0)
                initial = False
        if "Finals" in s:
            for x in self.Finals:
                try:
                    i = f.stateIndex(x)
                except common.DFAstateUnknown:
                    i = f.addState(x)
                f.addFinal(i)
        return f

    def sstransducer(self, s):
        if "AlphabetI" in s:
            a = {x for x in self.AlphabetI}
        else:
            a = set()
        f = SST(a)
        if "Initials" in s:
            initial = False
            for x in self.Initials:
                i = f.stateIndex(x, True)
                f.addInitial(i)
        initial = True
        for t in self.Transitions:
            if t is None:
                continue
            if len(t) == 3:
                try:
                    i0 = f.stateIndex(t[0])
                except common.DFAstateUnknown:
                    i0 = f.addState(t[0])
                try:
                    i1 = f.stateIndex(t[2])
                except common.DFAstateUnknown:
                    i1 = f.addState(t[2])
                if not t[1].isAInvariant() and "AlphabetI" not in s:
                    raise common.SSMissAlphabet()
                f.addTransition(i0, t[1], i1)
                if initial:
                    f.addInitial(i0)
                    initial = False
        if "Finals" in s:
            for x in self.Finals:
                try:
                    i = f.stateIndex(x)
                except common.DFAstateUnknown:
                    i = f.addState(x)
                f.addFinal(i)
        return f

    def ssfa(self, s):
        if "AlphabetI" in s:
            a = {x for x in self.AlphabetI}
        else:
            a = set()
        f = SSFA(a)
        if "Initials" in s:
            initial = False
            for x in self.Initials:
                i = f.stateIndex(x, True)
                f.addInitial(i)
        initial = True
        for t in self.Transitions:
            if t is None:
                continue
            if len(t) == 3:
                try:
                    i0 = f.stateIndex(t[0])
                except common.DFAstateUnknown:
                    i0 = f.addState(t[0])
                try:
                    i1 = f.stateIndex(t[2])
                except common.DFAstateUnknown:
                    i1 = f.addState(t[2])
                if not t[1].isAInvariant() and "AlphabetI" not in s:
                    raise common.SSMissAlphabet()
                f.addTransition(i0, t[1], i1)
                if initial:
                    f.addInitial(i0)
                    initial = False
        if "Finals" in s:
            for x in self.Finals:
                try:
                    i = f.stateIndex(x)
                except common.DFAstateUnknown:
                    i = f.addState(x)
                f.addFinal(i)
        return f

    def finals(self, s):
        self.Finals = list(s)
        return "Finals"

    def initials(self, s):
        self.Initials = list(s)
        return "Initials"

    def alphabeti(self, s):
        self.AlphabetI = s[0]
        return "AlphabetI"

    def alphabeto(self, s):
        self.AlphabetO = s[0]
        return "AlphabetO"

    def alphabet(self, s):
        return s

    def name(self, xxx_todo_changeme):
        (s,) = xxx_todo_changeme
        return s[:]

    def transitions(self, s):
        self.Transitions = [x for x in s if x is not None]
        return "Transitions"

    def ttransitions(self, s):
        self.Transitions = [x for x in s if x is not None]
        return "Transitions"

    def ssttransitions(self, s):
        self.Transitions = [x for x in s if x is not None]
        return "Transitions"

    def ssatransitions(self, s):
        self.Transitions = [x for x in s if x is not None]
        return "Transitions"

    def transition(self, xxx_todo_changeme1):
        (s1, s2, s3) = xxx_todo_changeme1
        return (s1, s2, s3)

    def ttransition(self, xxx_todo_changeme2):
        (s1, s2, s3, s4) = xxx_todo_changeme2
        return (s1, s2, s3, s4)

    def ssttransition(self, xxx_todo_changeme3):
        (s1, s2, s3) = xxx_todo_changeme3
        return (s1, s2, s3)

    def ssatransition(self, xxx_todo_changeme4):
        (s1, s2, s3) = xxx_todo_changeme4
        return (s1, s2, s3)

    def symbol(self, s):
        return s[0]

    def number(self, s):
        return int(s[0])

    def statedecl(self, s):
        if not hasattr(self, 'DFADStates'):
            self.DFADStates = []
        self.DFADStates.append(s)
        return "DeclaredStates"

    def sallspec(self, s):
        return SSAnyOf()

    def sonespec(self, s):
        return SSOneOf(sorted(list({x for x in s[0]})))

    def snotspec(self, s):
        return SSNoneOf(sorted(list({x for x in s[0]})))

    def vpspec(self, xxx_todo_changeme5):
        (s1, s2) = xxx_todo_changeme5
        return PSPVanila(s1, s2)

    def eqpspec(self, s1):
        return PSPEqual(s1[0])

    def neqpspec(self, xxx_todo_changeme6):
        (s1, s2) = xxx_todo_changeme6
        return PSPDiff(s1, s2)

    def sepsilon(self, s):
        return SSEpsilon()

    def names(self, s):
        return s

    def epsilon(self, s):
        return common.Epsilon

    def spec(self, s):
        return s[0]

    eol = lambda self, _: None
    dollar = lambda self, _: None


FAdoGrammar = lark.Lark.open("automata_grammar.lark", start="object", rel_to=__file__)
