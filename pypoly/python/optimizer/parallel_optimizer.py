from parallel_detection import ParallelDetection
from parallel_exploitation import BatchTensor

class ParallelOptimizer:
    @staticmethod
    def run(tree):
        # pass object reference by value
        ParallelDetection.run(tree)
        BatchTensor.run(tree)