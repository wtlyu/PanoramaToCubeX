# PanoramaToCubeX
Spherical to cubic panorama converter using CUDA

# Build

Before you compile the code, please make sure that opencv is already installed and configured correctly. If you want to compile the CUDA optimized version, CUDA SDK (CUDA Toolkit) must also be installed.

We provide two kinds of code, for CPU only and for CUDA optimized. In directory src, run

```
cd src
make convert_cuda
```

will make CUDA optimized version, and run

```
make convert_cpu
```

will compile for CPU only. You can also use

```
make all
```

to compile for all version.

All binary programs are saved in directory bin.

# Usage

useage: convert <inimagefile> <outimagefile>

Both convert_cuda and convert_cpu have the same usage. You can try the given panorama picture under test directory:

```
bin/convert_cuda test/input.jpg test/output_test.jpg
```

The correct output is test/output.jpg. You can compare with your output_test.jpg to ensure that the program runs correctly.

# Performance Test

We provide two performance test under directory script. Run like this:

```
script/test0.sh
```

Here is the result on my computer:

```
======== test for 2k (2048 × 1024) picture ========
cpu time 0.311445s
cuda time 0.310888s
======== test for 4k (4096 × 2048) picture ========
cpu time 1.263972s
cuda time 0.457165s
======== test for 8k (8192 × 4096) picture ========
cpu time 5.094330s
cuda time 0.646269s
======== test for 16k (16384 × 8192) picture ========
cpu time 20.477008s
cuda time 1.667253s
```

```
======== test for thread_pre_block = 32 ========
cuda time 1.581090s
======== test for thread_pre_block = 64 ========
cuda time 1.595847s
======== test for thread_pre_block = 128 ========
cuda time 1.575221s
======== test for thread_pre_block = 256 ========
cuda time 1.622817s
======== test for thread_pre_block = 512 ========
cuda time 1.694060s
```
