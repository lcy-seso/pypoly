# pypoly

Parse polyhedral representation from PyTorch JIT AST, perform dependence analysis and code transformations to generate efficient device codes, making more innovative Deep Learning models able to be (semi-)automatically optimized.

## How to compile

```bash
mkdir build
cd build

# set your torchlib path, if not set, torchlib will be downloaded from the Internet.
TORCHLIB_PATH="path of the torch library"
cmake -DPYTHON_EXECUTABLE:FILEPATH=`which python3` \
    -DTORCHLIB_PREFIX_DIR=$TORCHLIB_PATH \
    ..

make
```
