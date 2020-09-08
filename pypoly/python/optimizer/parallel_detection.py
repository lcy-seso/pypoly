from ir import PolyIR

# implicit computation order is scheduled by clear semantic syntax tree constructs: sequence, set(execute in parallel).
class BlockScheduler:
    # the return element is a syntax tree
    @staticmethod
    def run(tree):
        # a statement (operation) as a node
        # dependence relation / data (tensor / tensor array) as an edge
        # each node has a integer to indicate the logical execution time, by default the number is the topological order
        dataflow_graph = tree.generate_dataflow_graph()
        StatementScheduler.greedy_scheduler(dataflow_graph)
        return tree.generate_ast(dataflow_graph)
    
    @staticmethod
    def greedy_scheduler(dataflow_graph):
        pass

class NestedLoopScheduler:
    @staticmethod
    def feautrier(ir):
        # 1. dependence ananlysis
        # 2. build the optimization objective through input constraints
        # 3. launch feautrier scheduler
        return ir

    # frontend interfaces support source to source transformation (how to express the loop after affine scheduling)
    @staticmethod
    def run(tree):
        poly_ir = PolyIR(tree)
        transformed_poly_ir = NestedLoopScheduler.feautrier(poly_ir)
        return transformed_poly_ir.generate_ast()

# a takeaway: the recursive expansion to handle function call may be annotated by user
class ParallelDetection:
    @staticmethod
    def run(tree):
        def node_handler(tree):
            if tree.type() == "kBlock":
                tree = BlockScheduler.run(tree)
            elif tree.type() == "kLoop":
                tree = NestedLoopScheduler.run(tree)
            return tree
        tree.foreach_node(node_handler)