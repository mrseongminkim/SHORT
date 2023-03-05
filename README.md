# SHORT: Simplifying regular expression with Heuristics Optimization and Reinforcement learning Techniques

## Requirement
1. Python 3.10.10
2. pip intstall -r requirements.txt

## Directories and its Descriptions
* alpha_zero: alpha zero playable state elimination game.
* data: randomly generated FAdo recognizable NFAs
* result: test results for heuristics methods and alpha zero method
* utils: miscellaneous modules

## Overview
* SHORT/alpha_zero/models  
    pretrained neural network parameters  
* SHORT/alpha_zero/state_elimination

* SHORT/data  
    raw: result of random_nfa_generator.py  
    pkl: pkl converted version of raw datas  

* SHORT/result  
   alpha_zero_experiment_result.pkl: test result from alpha zero method  
   heuristics_experiment_result.pkl: test results from combinations of heuristics  
   C1, C2, C3, C4, C4, C5, C6, C7 denotes random elimination, decomposition with random elimination, state weight, decomposition with state weight, repeated state weight and decomposition with repeated state weight respectively  

* SHORT/utils/data_loader  
    read txt/pkl and save it to list  
* SHORT/utils/fadomata  
    modified FAdo functions/methods  
* SHORT/utils/graph_maker  
    make graph from the test result  
* SHORT/utils/heuristics  
    implementation of random, decomposition and state weight heuristic and their combinations  
* SHORT/utils/random_nfa_generator  
    make 100 NFAs per each parameters combinations  