# MTNess
### Machine Translation in C++ with the PyTorch C++ front-end

Only BiDeep RNN training has been implemented so far.

## Installation
Requirements:
* cmake - `apt install cmake`
* PyTorch C++ front-end (libtorch) (https://pytorch.org/get-started/locally/)

Build:
```bash
git clone https://github.com/Proyag/MTNess
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
make -j
```
