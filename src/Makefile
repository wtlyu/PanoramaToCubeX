LIBOPENCV = $(shell pkg-config --cflags --libs opencv)
convert_cuda: cuda.cu
	nvcc cuda.cu -o ../bin/convert_cpu $(LIBOPENCV)

convert_cpu: cpu.cpp
	g++ cpu.cpp -o ../bin/convert_cpu $(LIBOPENCV)

all: convert_cpu convert_cuda
