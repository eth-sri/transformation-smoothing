# Tools for soundly Analyzing Image Transformations #

This is a set of tools to analyze image transformations, mostly
computing their inverse for bilinear interpolation, via sound
overapproximations. It is designed be use used a C++ library and a
Python library.

## Requirements

- GNU C++ compiler (tested with 7.5.0)
- `python` command that points to python3.6 or higher with numpy installed
- modern CUDA developer toolkit (tested with 10.1, 10.2)

## Installation

### C++

Include `libsmoothing.h` and compile with the required flags.  See
`test.cpp` and the target `test` in `Makefile` for examples.

### Python

To build and install the python library enter the venv/conda env with
the correct prerequisites and run:

``` shell
make python
```

### Running Tests

``` shell
make test
```

## Known Issues
If the tests pass, but you are triggering assertions, bus errors and
segfaults in python this likely means that the version of your CUDA
driver and your cudatoolkit in the conda/torch installation does not
match.  Make sure they do.
