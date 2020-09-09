from ir import PolyIR

# implicit computation order is scheduled by clear semantic syntax tree constructs: sequence, set(execute in parallel), e.g. an explicit order
# TODO: check the implementation of instruction_scheduler in XLA
class BlockScheduler:
    # the return element is a syntax tree
    @staticmethod
    def run(tree):
        # a statement (operation) as a node
        # dependence relation / data (tensor / tensor array) as an edge
        # each node has a integer to indicate the logical execution time, by default the number is the topological order
        # nodes with the same number can be executed in parallel
        dataflow_graph = tree.generate_dataflow_graph()
        BlockScheduler.greedy_scheduler(dataflow_graph)
        return tree.generate_ast(dataflow_graph)
    
    @staticmethod
    def greedy_scheduler(dataflow_graph):
        pass

class NestedLoopScheduler:
    @staticmethod
    def feautrier(ir):
        # 1. dependence ananlysis
        # 2. build the optimization objective through constraints (validity, proximity, coincidence, linear independence of previous rows)
        # 3. launch feautrier scheduler
        return ir

    # transformed model is expressed in IR, which can be translated to target output code(like python or c++)
    @staticmethod
    def run(tree):
        poly_ir = PolyIR(tree)
        transformed_poly_ir = NestedLoopScheduler.feautrier(poly_ir)
        return transformed_poly_ir.generate_ast()

class ParallelDetection:
    @staticmethod
    def run(tree):
        def node_handler(tree):
            if tree.annotation() == "kBoundary":
                return tree
            if tree.type() == "kBlock":
                tree = BlockScheduler.run(tree)
            elif tree.type() == "kLoop":
                tree = NestedLoopScheduler.run(tree)
            return tree
        tree.foreach_node(node_handler)