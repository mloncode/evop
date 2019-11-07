# Optimization using Evolutionary Operators
Deterministic algorithms like Knuth’s dynamic programming algorithm is capable of constructing the statically optimal tree in O(n^2) time, so it becomes impractical when the number of elements in the tree is very large.
For the last couple of years new ideas related to genetics and evolution theory have gained in popularity.
Genetic algorithms (GA) are based on natural selection mechanisms together with genetic operators make searching for results close to optimal.
Our work is kind of an experiment, checking for what cases optimization binary trees by genetic and evolutionary operators makes sense.

### Binary tree optimization using evolutionary operators
Not always super optimal solutions are needed. More often we need fast, quasi-optimal solutions. In other words, the tree where the _search cost_ is minimal. In this case, the search cost refers to the number of needed comparisons, to find a certain element.

In this experiment we tried to encode binary trees into genomes and apply evolutionary operators to find a quasi-optimal binary tree much faster than deterministic algorithms.

### Monte Carlo Tree Search
We also experimented with Monte Carlo tree search (MCTS).
MCTS is a heuristic search algorithm for some kinds of decision processes, most notably those employed in game play.
We tried to apply _mcts (Monte Carlo Tree Search)_ algorithms as an alternative selection algorithm.

