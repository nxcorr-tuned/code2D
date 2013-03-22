CXX = g++
CC  = gcc
LD  = g++
F90  = ifort

.SUFFIXES: .o .cpp .ptx .cu

#CUDA_TK  = /usr/local/cuda
#CUDASDK  = $(HOME)/NVIDIA_GPU_Computing_SDK

OFLAGS = -O3 -g -Wall  -msse4 -fopenmp
CFLAGS =  $(OFLAGS) -Wstrict-aliasing=2
CXXFLAGS =  $(OFLAGS) -Wstrict-aliasing=2

NVCC      = nvcc  
NVCCFLAGS = -arch sm_12
#NVCCFLAGS = -arch sm_20   
#NVCCFLAGS = -arch sm_30 
#NVCCFLAGS = -arch sm_30   -Xptxas -v,-abi=no 


LDFLAGS = -lcudart -lOpenCL   -fopenmp


CUDAKERNELSPATH = ./
CUDAKERNELS = cunxcor.cu_o

OBJ = main.o nxcor.o clocks.o nxcor_avx.o nxcor_sse.o

PROG = main

all:	  $(OBJ)  $(PROG)


$(PROG): $(OBJ) $(CUDAKERNELS)
	$(LD)  $^ -o $@  $(LDFLAGS)

%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.cu_o:  $(CUDAKERNELSPATH)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


clean:
	/bin/rm -rf *.o main  *.cu_o

nxcor_sse.o: sse.h
nxcor_avx.o: avx.h


