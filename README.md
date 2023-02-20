# SHORT: Simplifying regular expression with Heuristics Optimization and Reinforcement learning Techniques

## Requirement
1. Python 3.10.10
2. pip intstall -r requirements.txt

## Overview
1. random_nfa_generator  
    make 100 NFAs per each parameters combinations  
    the number of NFAs reduced to 100 from 20,000 since RAM capacity was limited.  
2. data  
    raw: result of random_nfa_generator.py  
    pkl: pkl converted version of raw datas, much faster, always prefer this one over raw one.  
3. utils  
    data_loader: read txt/pkl and save it to list  
    fadomata: modified FAdo functions/mehtods  
    heuristics: bunch of functions related to problems  
4. result  
    experimental_result.pkl holds all the accumulated datas.  
    C_[1 - 6].csv holds average time and size  