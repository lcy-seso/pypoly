class Tree:
    pass

class Tensor:
    pass

class TensorArray:
    pass

class Context:
    def __init__(self):
        self.func_name2signature_ = {}
        self.func_name2computation_ = {}

    def find_func_signature(self, func_name):
        if func_name in self.func_name2signature_:
            return self.func_name2signature_[func_name]
        else:
            computation_tree = self.func_name2computation_[func_name]
            # TODO: this context should be the top level
            func_signature = shape_propagation(computation_tree, None)
            self.func_name2signature_[func_name] = func_signature
            return func_signature

def computation_tree4lambda_expr(expr):
    return Tree()

# 'args_signatures' here is used to generate annotations on TensorArray
def infer_desc(func_signature, args_signatures):
    out_type = func_signature.get_out_type()
    if isinstance(out_type, Tensor):
        return out_type
    
    assert isinstance(out_type, TensorArray)

    for i in range(out_type.depth()):
        src_tensor_array_name, src_tensor_array_level = out_type.get_depth(i)
        src_tensor_array_signature = args_signatures[src_tensor_array_name]
        out_type.set_depth(i, src_tensor_array_signature.get_depth(src_tensor_array_level))
    
    return out_type

def build_args_signature(types):
    return []

def extract_applied_func_out_desc(expr, context):
    applied_func_expr = expr.args[1]
    if applied_func_expr.is_lambda_func_expr():
        computation_tree = computation_tree4lambda_expr(applied_func_expr)
        applied_func_signature = shape_propagation(computation_tree, context)
    else:
        applied_func_signature = context.find_func_signature(applied_func_expr.to_str())
    desc = infer_desc(applied_func_signature, build_args_signature(expr.args[2:]))
    return desc

def scan_handler(expr, context):
    # expr.args[1] -> func to be applied 
    # expr.args[2] -> init
    # expr.args[3] -> iterable list
    # expr.args[4:] -> remaining arguments of the applied func

    desc = extract_applied_func_out_desc(expr, context) 
    iterable_list_type = context.get_type(expr.args[3].to_str())
    if isinstance(desc, Tensor):
        return TensorArray(desc, [iterable_list_type.get_depth(0)])
    else:
        return TensorArray(desc.tensor_info(), [iterable_list_type.get_depth(0)] + desc.depth())

def map_handler(expr, context):
    # expr.args[1] -> func to be applied
    # expr.args[2] -> iterable list
    # expr.args[3:] -> remaining arguments of the applied func

    desc = extract_applied_func_out_desc(expr, context)
    iterable_list_type = context.get_type(expr.args[3].to_str())
    if isinstance(desc, Tensor):
        return TensorArray(desc, [iterable_list_type.get_depth(0)])
    else:
        return TensorArray(desc.tensor_info(), [iterable_list_type.get_depth(0)] + desc.depth())

def fold_handler(expr, context):
    # expr.args[1] -> func to be applied
    # expr.args[2] -> init
    # expr.args[3] -> iterable list
    # expr.args[3:] -> remaining arguments of the applied func
    return extract_applied_func_out_desc(expr, context)

def update_context(context, args):
    return context

def is_system_func(expr):
    if expr.args[0].to_str() in ["scan", "map", "fold"]:
        return True
    else:
        return False

def system_func_handler(func_expr, context):
    func_name = func_expr.get_arg(0).to_str()
    # here func can also be meta operations, like conv, matmul, etc
    if func_name == "scan":
        return scan_handler(func_expr, context)
    elif func_name == "map":
        return scan_handler(func_expr, context)
    elif func_name == "fold":
        return fold_handler(func_expr, context)
    else:
        assert False

def user_func_handler(expr, context):
    func_name = expr.to_str()
    func_signature = context.find_func_signature(func_name)
    return infer_desc(func_signature, build_args_signature([])) 

# tree is a computation node which accepts several args with annotated type and produces one output
# this function is responsible to infer the type info of the output tensor / tensor array
# context
# - a map from variable name to type
# - signatures of named functions
# - a map from function name to the internal representation

# assume the return statement only contains one variable instead of an expression (need a canonization pass)

# explicit type at compiler time
# TensorDesc
#   dims -> List[int], data_type -> enum
# TensorArrayDesc
#   input: depth -> int, tensor_info -> TensorDesc
#   output: depth_arr -> List[tuple<string, int>], tensor_info -> TensorDesc
def shape_propagation(tree, context):
    assert tree.type() == "kComputation"

    cur_context = update_context(context, tree.args())

    for node in tree.get_block().children():
        # each node is a statement to do function evaluation
        assert node.type() == "kStatement"
        expr = node.get_expr()
        assert expr.type() == "op"
        assert expr.op_type() == "eval"

        var_expr = expr.get_arg(0)
        var_name_str = var_expr.to_str()

        func_expr = expr.get_arg(1)
        assert func_expr.type() == "apply"

        if is_system_func(func_expr):
            desc = system_func_handler(func_expr, context)
        else:
            desc = user_func_handler(func_expr, context)
        cur_context = update_context(context, {var_name_str:desc})

    # TODO: resolve the type of Tuple
    return cur_context.find_var_type(tree.get_return_var().to_str())