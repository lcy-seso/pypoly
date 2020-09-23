# SSA (canonization)

Transform the frontend user code into SSA format by variable renaming (*phi node*?)

# Shape info propagation

Shape information is useful at runtime when allocating memories.
- Shape of tensor is easily inferred similar to the procedure of existed deep learning (compiler) frameworks
- although the exact shape info of tensor array is evaluated only runtime
  - some dimensions may be static
  - some dynamic dimensions across different Tensor Arrays may share a same value. This information is useful at runtime.

# Replacement / Inline of variables

*Is this pass included in the SSA*?
- if a variable is assigned only once, occurrences of this variable in the following code can be replaced with right hand side expression
- generally, if SSA is satisfied, this kind of inlining always can achieved?