# Translate from frontend API to IR

scan to for

# SSA (canonization)

Transform the frontend user code into SSA format by variable renaming.

No *phi node* in IR. Control constructs are limited to *if-else* and *for* under some constraints.

# Shape info propagation

This process is launched after *SSA*. Before propagating shape info, we may assume that most of the initialization statements of *Tensor* or *TensorArray* are placeholders(essential fields are missing).

Propagation is executed recursively. From the top level, the computation structure is a directed acyclic graph split into multiple blocks(regions). Each block is seen as a function, which accepts some inputs whose shape info is known, and generates some outputs whose info needs to be inferred.

According to variable types listed in another document, different algorithms will be applied
- **Tensor**: similar to existed framework, inference is build on properties of meta operators (the root is *loop* program representation, e.g., how to infer the shape of output of matrix mutiply, refer to dimension inference in Tensor Comprehension)
- **Tensor Array**: apart from inside tensor info, we need data fields to describe the hierarchical dimension structure. Since we constrain the interfaces when building the Tensor Array, annotations can be added at each level
  - some dimensions may be static -> use *for* with a constant or symbolic loop bound
  - some dynamic dimensions across different Tensor Arrays may share a same value -> iterate over another Tensor Array at a certain level
- **Tuple**: a combination of references to existed tensors or tensor arrays

More specifically, following protobufs are used to demonstrate the idea
```protobuf
message Tensor {
  required Shape shape = 1;
  enum DataType {
    FLOAT32 = 0;
    FLOAT64 = 1;
    INT32 = 2;
    INT64 = 3;
  }
}

message TensorArray {
  required int32 id = 1;
  required int32 length = 2;
  oneof item {
    Tensor tensor = 1;
    TensorArray tensor_array = 2;
  }
}
```
However, we cannot assume that *length* in TensorArray is known at compiler time. On the other hand, to apply some specified optimizations, we need more information of TensorArray.
```protobuf
message DimInfo {
  oneof info {
    int32 value = 1;
    int32 ref_tensor_array_id = 2;
  }
}

message TensorArrayV2 {
  repeated DimInfo dims = 1;
  required Tensor tensor_info = 2;
}
```
Ideally, at runtime, we need a tree like stucture to store the full dimension information. Since we constrain operators used currently, we can assume that remaining tensor arrays' dimension info is transformed from the source tensor array.
# Replacement / Inline of variables

*Is this pass included in the SSA*?
- if a variable is assigned only once, occurrences of this variable in the following code can be replaced with right hand side expression
- generally, if SSA is satisfied, this kind of inlining always can achieved?