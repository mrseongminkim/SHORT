from .fa import *
# from sm import *

def RIWC(n=5):
    """ Lower bound n^(n-1) for right ideals syntactic complexity 
        J. Brzozowski and Y. Ye "Syntactic Complexityt of Ideal and Closed Languages" DLT 2011
        """
    if n < 4: 
        raise TstError("number of states must be greater than 3")
    f = DFA()
    f.setSigma(["a","b","c","d"])
    f.States = list(range(n))
    f.setInitial(0)
    f.addFinal(n-1)
    f.addTransition(n-2,"a",0)
    f.addTransition(n-2,"c",0)
    f.addTransition(n-2,"d",n-1)
    f.addTransition(n-1,"a",n-1)
    f.addTransition(n-1,"b",n-1)
    f.addTransition(n-1,"c",n-1)
    f.addTransition(n-1,"d",n-1)
    f.addTransition(0,"b",1)
    f.addTransition(1,"b",0)
    for i in range(n-2):
        f.addTransition(i,"a",i+1)
        f.addTransition(i,"c",i)
        f.addTransition(i,"d",i)
    for i in range(2,n):
        f.addTransition(i,"b",i)
    return f    

def LIWC(n=3,Finals=None):
    """ Lower bound n^(n-1) for left ideals syntactic complexity
    J. Brzozowski and Y. Ye "Syntactic Complexityt of Ideal and Closed Languages" DLT 2011
    """
    if n < 3: 
        raise TstError("number of states must be greater than 2")
    f = DFA()
    f.setSigma(["a","b","c","d","e"])
    f.States = list(range(n))
    f.setInitial(0)
    if Finals is None :
        Finals = list(range(1,n))
    f.setFinal(Finals)
    f.addTransition(0,"a",0)
    f.addTransition(0,"b",0)
    f.addTransition(0,"c",0)
    f.addTransition(0,"d",0)
    f.addTransition(0,"e",1)
    f.addTransition(n-1,"a",1)
    f.addTransition(n-1,"c",1)
    f.addTransition(n-1,"d",0)
    f.addTransition(n-1,"e",1)
    f.addTransition(1,"b",2)
    f.addTransition(2,"b",1)
    for i in range(1,n-1):
        f.addTransition(i,"a",i+1)
        f.addTransition(i,"c",i)
        f.addTransition(i,"d",i)
        f.addTransition(i,"e",1)
    for i in range(3,n):
        f.addTransition(i,"b",i)
    return f    

