class Tensor:
    pass

class TensorArray:
    pass

class Node:
    pass

class Edge:
    pass

class Graph:
    pass

# meta info is stored in this class: meta operators register their properties
# example: 2D matmul
# A = matmul(B, C)
# signature: (Tensor(float, (N, K)), Tensor(float, (K, M))) -> (Tensor(float, (N, M)))
# parallelism meta info: loop(N) and loop(M) is parallelizable, loop(K) is conditionally parallelizable, since it is a reduction
# informal definition for meta info of parallelism:
# for a module of computation: a function call, a block made up of statements, a nested for loop
# there must(?) be loops to iterate over certain dimensions of tensors or tensor arrays (granuality?)
# a record of in parallelism meta info is a label of these loops to tell whether a loop is parallelizable
class FunctionSignature:
    pass

class Expr:
    pass

class Tree:
    def __init__(self, type, children):
        self.type_ = type
        self.annotation_ = "none"
        self.children_ = children
    
    # enum of types
    # kBlock, kSequence, kSet, kStatement, kCall, kLoop, kIf
    def type(self):
        return self.type_
    
    # enum of annotations
    # kBoudary, kParallelizable
    def annotation(self):
        return self.annotation_

    def children(self):
        return self.children_
    
    def set_child(self, i, child):
        self.children_[i] = child
    
    def generate_dataflow_graph(self):
        return Graph()
    
    # a strong assumption of the input code's pattern:
    #  L_0(i_0)
    #    L_1(i_1)
    #      ...
    #        L_n(i_n)
    #          [S_0, S_1, ..., S_m]
    # constraints:
    #  1. each iterator is an induced variable, initial value is 0, increment is 1, upper bound is a symbolic constant or a certain bound of a Tensor Array
    #  2. all statements are in the deepest loop and follow the SSA requirements: left hand side (write in) is a accessed tensor in array, right hand side is an element in tensor array or context (scope)
    #  3. a statement S_i can be if-else parsed from filter (further specifications are needed)
    def down_to_deepest_loop(self):
        node = self
        while node.children_[0].type() == "kLoop":
            node = node.children_[0]
        return node
    
    def foreach_node(self, func):
        if self.annotation() == "kBoundary":
            return self
        self = func(self)
        if self.type() == "kLoop":
            node = self.down_to_deepest_loop()
        else:
            node = self
        # kBlock, kSequence, kSet, kStatement, kCall
        for i in range(len(node.children_)):
            node.children_[i] = node.children_[i].foreach_node(func)
        return self

    def generate_function_signature(self):
        # placeholder for Propagation related interfaces
        return FunctionSignature()

# reference to data structures in PET
class PolyIR:
    def generate_ast(self):
        return Tree("None", [])