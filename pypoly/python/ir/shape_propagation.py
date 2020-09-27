def handler_scan(block, scop):
    return scop

def handler_map(block, scop):
    return scop

def handler_fold(block, scop):
    return scop

# normal functions and lambda functions
def handler_normal(block, scop):
    return scop

# tree is a computation node which accepts several args with annotated type and produces one output
# this function is responsible to infer the type info of the output tensor / tensor array
# scope maintain the context by storing a dictionary mapping from variable name to type

# assume the return statement only contains one variable instead of an expression (need a canonization pass)

# explicit type at compiler time
# Tensor
#   dims -> List[int], data_type -> enum
# TensorArrayDesc
#   input: depth -> int, tensor_info -> TensorDesc
#   output: depth_arr -> List[tuple<string, int>], tensor_info -> TensorDesc
def shape_propagation(tree, scop):
    assert tree.type() == "kComputation"
    cur_scop = MergeScop(scop, tree.args())

    if tree.func_type() == "kScan":
        cur_scop = handler_scan(tree.get_block(), cur_scop)
    elif tree.func_type() == "kMap":
        cur_scop = handler_map(tree.get_block(), cur_scop)
    elif tree.func_type() == "kFold":
        cur_scop = handler_map(tree.get_block(), cur_scop)
    else:
        cur_scop = handler_normal(tree.get_block(), cur_scop)
    
    return cur_scop.find(tree.get_return_var())