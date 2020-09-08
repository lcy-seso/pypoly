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
    
    def type(self):
        return self.type_
    
    def annotation(self):
        return self.annotation_

    def children(self):
        return self.children_
    
    def generate_dataflow_graph(self):
        return Graph()
    
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

class PolyIR:
    def generate_ast(self):
        return Tree("None", [])