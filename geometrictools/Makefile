# CPP files
TRANSFORMS = $(wildcard transforms/*.cpp)
UTILS = $(wildcard utils/*.cpp)
DOMAINS = $(wildcard domains/*.cpp)
CPP := $(TRANSFORMS) $(UTILS) $(DOMAINS)

# Object files
OBJECTS = $(addprefix lib/, $(patsubst %.cpp, %.o, $(CPP)))
GPU_OBJECTS = build/gpu.o build/image_gpu.o

# libraries
PYTHONI := $(shell python3-config --includes)
PYTHONLD := $(shell python3-config --ldflags)
NUMPYI := $(shell python -c "import numpy as np; print('-I'+np.get_include())")
PYTHON := $(PYTHONI) $(PYTHONLD) $(NUMPYI)

# flags
OLVL = -O0 #fast
DEBUG_CXX =
DEBUG_NVCC =
FLAGS = -march=native -fopenmp -fno-lto -Wall
FASTFLAGS = -mavx -mavx2 -mfma
INCL = -I ${CURDIR} -I CLI11/include/ -std=c++11 -lpthread $(PYTHON) -I indicators/include
O_NVCC =  -Xptxas -O3 -Xcompiler ${OLVL}

# compiler
CXX = g++
NVCC = nvcc

clean:
	python setup.py clean
	rm -rf build lib

build/gpu.o: gpu/gpu.cu
	mkdir -p build
	${NVCC} -v --default-stream per-thread ${DEBUG_NVCC} ${O_NVCC} -c -o build/gpu.o gpu/gpu.cu ${INCL} -ccbin ${CXX} -Xcompiler -fPIC

build/image_gpu.o: gpu/image_gpu.cpp
	mkdir -p build
	${NVCC} -v --default-stream per-thread ${DEBUG_NVCC} ${O_NVCC} -c -o build/image_gpu.o gpu/image_gpu.cpp ${INCL} -ccbin ${CXX} -Xcompiler -fPIC

%.o:
	mkdir -p $(dir $@)
	${CXX} ${DEBUG_CXX} ${OLVL} ${FLAGS} ${FASTFLAGS} -fPIC -c $(patsubst lib/%.o, %.cpp, $@) -o $@ ${INCL} -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart


lib/libpygeometrictools.o: libgeometrictools.cpp
	mkdir -p lib
	${CXX} ${DEBUG_CXX} ${OLVL} ${FLAGS} ${FASTFLAGS} -fPIC -c libgeometrictools.cpp -o lib/libpygeometrictools.o ${INCL} -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

build/test: $(CPP) $(GPU_OBJECTS) test.cpp lib/libpygeometrictools.o
	mkdir -p build
	${CXX} ${DEBUG_CXX} ${OLVL} ${FLAGS} ${FASTFLAGS} test.cpp $(CPP) $(GPU_OBJECTS) lib/libpygeometrictools.o -o build/test ${INCL} -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -I Catch2/single_include/catch2

python: $(OBJECTS) $(GPU_OBJECTS)	lib/libpygeometrictools.o
	ar rcs lib/libpygeometrictools.a lib/libpygeometrictools.o $(GPU_OBJECTS) $(OBJECTS)
	python setup.py install
	echo $(OBJECTS)

test: build/test
	./build/test
