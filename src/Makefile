LIBOPENCV = $(shell pkg-config --cflags --libs opencv)
convert_cuda: ../bin/convert_cuda
	nvcc cuda.cu -o ../bin/convert_cpu $(LIBOPENCV)

convert_cpu: ../bin/convert_cpu
	g++ cpu.cpp -o ../bin/convert_cpu $(LIBOPENCV)

all: convert_cpu convert_cuda
