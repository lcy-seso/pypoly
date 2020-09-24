# Translate from frontend API to IR

scan to for

# SSA (canonization)

Transform the frontend user code into SSA format by variable renaming.

No *phi node* in IR. Control constructs are limited to *if-else* and *for* under some constraints.

# Shape info propagation

Shape information is useful at runtime when allocating memories.
- Shape of tensor is easily inferred similar to the procedure of existed deep learning (compiler) frameworks. This process is recursive and built on meta operations.
- although the exact shape info of tensor array is evaluated only at runtime
  - some dimensions may be static
  - some dynamic dimensions across different Tensor Arrays may share a same value. This information is useful when allocating memory.

# Replacement / Inline of variables

*Is this pass included in the SSA*?
- if a variable is assigned only once, occurrences of this variable in the following code can be replaced with right hand side expression
- generally, if SSA is satisfied, this kind of inlining always can achieved?