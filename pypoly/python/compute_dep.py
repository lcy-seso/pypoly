class Info:
    def __init__(self, src_var_name, depth=0):
        self.src_var_name_ = src_var_name
        self.depth_ = depth
        self.dep_id = -1

class Context:
    def __init_(self):
        self.var2info_ = {}
        self.ret_info_ = None
        # space_[i] = 0 -> ci >= 0
        # space_[i] = 1 -> ci = 0
        # space_[i] = 2 -> ci > 0
        self.space_ = []

def is_tensor_operation(func_info):
    return False

def compute_dep_vecs(in_info, out_info):
    pass

# input: type info of known variables
# output: dependence vectors of the called function
def scan_handler(tree, ctx):
    pass

def map_handler(tree, ctx):
    func_info = tree.get_lambda_func_info()
    if len(ctx.space_) == 0:
        # top level
        if is_tensor_operation(func_info):
            # do nothing
            return None
        ctx.space_.append(0)
        ctx.var2info_[func_info.get_param_name()] = Info(tree.get_operand_name())
        ctx.ret_info_ = Info(tree.get_return_name())
        return domain(tree.get_lambda_func(), ctx)
    else:
        # inner level
        if is_tensor_operation(func_info):
            # TODO: compute and return the dependence vector
            pass
        ctx.space_.append(0)

def reduce_handler(tree, ctx):
    pass

def domain(tree, ctx):
    pass