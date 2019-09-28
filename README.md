# Optimization using Evolutionary Operators

- bintree (Binary tree optimization using evolutionary operators)
> The research on binary trees and searching information to try answer the question on what is the best binary search tree. In other words, the tree where the _search cost_ is minimal.

In this case, the _search cost_ refers to the number of needed comparisons, to find a certain element.
Deterministic algorithms like Knuth's dynamic programming algorithm is capable of constructing the statically optimal tree in `O(n^2)` time, so it becomes impractical when the number of elements in the tree is very large. For the last couple of years new ideas related to genetics and evolution theory have gained in popularity. Genetic algorithms (GA) are based on natural selection mechanisms together with genetic operators make searching for results close to optimal.
Our work is kind of an experiment, checking for what cases optimization binary trees by genetic and evolutionary operators makes sense.

- mcts (Monte Carlo Tree Search)
This is a continuation of _bintree (Binary tree optimization using evolutionary operators)_ research,
but instead of (1+1) selection algorithm, we tried to apply _mcts (Monte Carlo Tree Search)_ algorithms.

> Monte Carlo tree search (mcts) is a heuristic search algorithm for some kinds of decision processes, most notably those employed in game play.
