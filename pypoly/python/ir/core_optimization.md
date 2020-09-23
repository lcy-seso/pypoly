# Parallelism

The structure of nested *for* loops is used as a boundary for two optimization strategies
- computation graph to optimize a block of statements
- mathemetic model (Polyhedral) to analysis iteration domain and access domain

## Detect

- a statement is seen as a node reading in and computing output in a computation graph (since currently no control construct is introduced in this level, the graph is directed and acyclic). Existed algorithms can be applied to DAGs to detect parallelizable statements (gate computations in LSTM cell).
- raw computation order can be changed to expose more parallelism while obeying dependence relations

## Exploit

two exploitation approaches are available
- distribute then sync
- gather data, feed to functions, scatter the output tensor

we are focusing on the second one, because
1. more practical and efficient, according to existed experiment results and current architecture & programming model of accelerators (GPU)
2. help us to understand and design the IR because it requires more complex functionalities
    - statement level fusion (recursive fusion of subtrees)
    - batching dimension inference (root cause: parallelism nature of certain *for* loops, propagation recursively helps to analyze a high level function)
    - automatically insert data movement commands (gather and scatter) to ensure transformed code is correct

# Scan

Scan is a widely studied computation pattern. There are several algorithms to optimize this region of code. *Blelloch Scan* reduces the time complexity from *O(n)* to *O(logn)* by consuming more memory.

# Reduce

Like scan, reduce is an important pattern too. Different granularities of reuduced items need different algorithms / optimizations.

# Fusion related to Map
In the context of deep learning, map corresponds to element wise operators, which can be merged into predecessor or successor under certain rules.

# Gather and Scatter

Data movements (host to device, device to host, device to device) and kernel computations are main contributors of the final execution time. Ideally, we hope time of data movements can be *minimized* or *gathered* to reduce the cost of scheduling (frequently switch between differnt streams).
If the dependence flow can be represented by affine expressions and affine scheduling is used to detect and exploit the parallelism, the memory layout plan can be arranged by a plan consistent with the affine scheduling of computations.
In the end, there are few data movements during the main computation. A transpose operation is needed to recover to the initial memory plan. 