echo "======== test for thread_pre_block = 32 ========"
./bin/convert_cuda_32 ./test/pic16k.jpg ./test/output.jpg

echo "======== test for thread_pre_block = 64 ========"
./bin/convert_cuda_64 ./test/pic16k.jpg ./test/output.jpg

echo "======== test for thread_pre_block = 128 ========"
./bin/convert_cuda_128 ./test/pic16k.jpg ./test/output.jpg

echo "======== test for thread_pre_block = 256 ========"
./bin/convert_cuda_256 ./test/pic16k.jpg ./test/output.jpg

echo "======== test for thread_pre_block = 512 ========"
./bin/convert_cuda_512 ./test/pic16k.jpg ./test/output.jpg
