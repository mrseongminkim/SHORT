# SHORT: Simplifying regular expression with Heuristics Optimization and Reinforcement learning Techniques

This repo provides the source code & data of our paper: Obtaining Smaller Regular Expressions from Finite Automata (CIAA 2023 submitted).

## Dependencies
* Python 3.10.10
* Python libraries
    ```
    pip install -r requirements.txt
    ```

## Data
All the preprocessed data can be found in 
```
./data
```

The repo structure looks like the following:
```plain
.
├─alpha_zero
│  ├─models (pre-trained model)
│  └─state_elimination (set of functions for our model)
│      └─pytorch (neural network for our model)
│
├─data
│  ├─random_dfa (preprocessed random DFAs)
│  ├─random_nfa (preprocessed random NFAs)
│  └─random_nfa_sparse (preprocessed random sparse DFAs)
│
├─result
│  ├─heuristics_dfa_false (heuristics experiment results for random DFAs without simplification)
│  ├─heuristics_dfa_true (heuristics experiment results for random DFAs with simplification)
│  ├─heuristics_nfa_false (heuristics experiment results for random NFAs without simplification)
│  ├─heuristics_nfa_true (heuristics experiment results for random DFAs without simplification)
│  ├─optimal (shortest length obtained from random DFAs/NFAs by enumerating all the permutations)
│  ├─rl_dfa_false (ours experiment results for random DFAs without simplification)
│  ├─rl_dfa_true (ours experiment results for random DFAs with simplification)
│  ├─rl_nfa_false (ours experiment results for random NFAs without simplification)
│  └─rl_nfa_true (ours experiment results for random NFAs with simplification)
│
└─utils (codes for heuristics, FAdo modification, FA generator, etc.)
```

## Train models
In order to train models
```
python main.py train
```
You can modify the hyperparameters in main.py.

## Evaluate models
We need to set target FAs and decide enable or disable on-the-fly simplification.

### Heuristics
To test heuristics, you can use following commands.
```
python main.py heuristics [DFA | NFA] [true | false]
```

### Reinforcement Learning
To test RL approach, you can use following commands.
```
python main.py rl [DFA | NFA] [true | false]
```

## Acknowledgment
The AlphaZero framework used in this work come from the following repository:
```
[https://github.com/suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
```
