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

# meta info is stored in this class, registered by some meta operators or through inference (propagation)
# need to constrain the pattern of if-else(filter), affine?

# rule: any variable appears at the left hand side of an assignment consumes memory space (tmp and output)

# example of the signature
# matrix multiply:
# input:
#  p (float, (a1, a2))
#  q (float, (a2, a3))
# output:
#  r (float, (a1, a3))
# LSTM cell: 
# input:
#   x (float, (a1, a2))
#   h (float, (a1, a3))
#   c (float, (a1, a3))
#   w_ii, w_if, w_ig, w_io (float, (a2, a3))
#   w_hi, w_hf, w_hg, w_ho (float, (a3, a3))
#   b_ii, b_hi, b_if, b_hf, b_ig, b_hg, b_io, b_ho (float, (a3))
# tmp:
#   i (float, (a1, a3))
#   f (float, (a1, a3))
#   g (float, (a1, a3))
#   o (float, (a1, a3))
# output:
#   h_out (float, (a1, a3))
#   c_out (float, (a1, a3))
class Signature:
    def __init__(self):
        self.input_ = []
        self.tmp_ = []
        self.output_ = []

        # parallelism meta info: parallelizable loop -> dims of tensors / tensor arrays
        self.parallel_info_ = {}

    @staticmethod
    def generate_signature(tree):
        # gather and scatter added by framework will be labeled as "kBoundary"
        if tree.annotation() == "kBoundary":
            return Signature.get_default_signature(tree)
        if tree.type() == "kBlock" or tree.type() == "kSequence" or tree.type() == "kSet":
            return Signature.merge_signatures([Signature.generate_signature(sub_tree) for sub_tree in tree.children()])
        elif tree.type() == "kLoop":
            # 1. register parallelizable loop to parallel_info_
            # 2. properly handle tensor and tensor array (expanded matrix multiply && stacked LSTM)
            # TODO
            return Signature()
        elif tree.type() == "kStatement":
            return Signature.generate_signature4statement(tree)
        else:
            assert tree.type() == "kIf"
            # assume each kIf has a if-body and a else-body
            return Signature.merge_signatures([Signature.generate_signature(tree.children()[0]), Signature.generate_signature(tree.children()[1])])

    # take the gate computation statement in LSTM cell as an example
    @staticmethod
    def generate_signature4expr(expr):
        # TODO
        return Signature()

    @staticmethod
    def generate_signature4statement(tree):
        # assume shape inference is launched before the generation of signature
        # two cases
        # a. normal tensors
        # b. tensor slices or tensors indexed by iteration variables
        if tree.is_call():
            # name rebinding
            return Signature.generate_signature(tree.children()[0])
        # actually, this a sub case of function call
        elif tree.is_expr():
            return Signature.generate_signature4expr(tree.expr())
        else:
            # return statement
            assert tree.is_return()
            # TODO: label data to outputs
            return Signature()

    @staticmethod
    def get_default_signature(tree):
        # TODO: built in meta computation operators
        pass

    @staticmethod
    def merge_signatures(items):
        # two parts
        # 1. aggregate input, tmp and output info
        # 2. merge parallel info following certain rules
        # one possible case, tensor A is used as input in two statements, in the first statement one of A's dimension is parallelizable while in the second statement none of A's dimension is parallelizable
        pass

# statement is represented by Expr, like i = sigma(x*w_ii+b_ii+h*w_hi+b_hi)
# or we can argue that this statement can be expressed in a functional style by replacing ambiguous arithmatic operators with explicit function calls
class Expr:
    pass

class Tree:
    def __init__(self, type, children):
        self.type_ = type
        self.annotation_ = "none"
        self.children_ = children
    
    # enum of types
    # kBlock, kSequence, kSet, kStatement, kLoop, kIf
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
        return Signature()

# refer to data structures in PET
class PolyIR:
    def generate_ast(self):
        return Tree("None", [])