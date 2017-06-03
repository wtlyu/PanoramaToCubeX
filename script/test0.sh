echo "======== test for 2k (2048 × 1024) picture ========"
./bin/convert_cpu ./test/pic2k.jpg ./test/output.jpg
./bin/convert_cuda ./test/pic2k.jpg ./test/output.jpg

echo "======== test for 4k (4096 × 2048) picture ========"
./bin/convert_cpu ./test/pic4k.jpg ./test/output.jpg
./bin/convert_cuda ./test/pic4k.jpg ./test/output.jpg

echo "======== test for 8k (8192 × 4096) picture ========"
./bin/convert_cpu ./test/pic8k.jpg ./test/output.jpg
./bin/convert_cuda ./test/pic8k.jpg ./test/output.jpg

echo "======== test for 16k (16384 × 8192) picture ========"
./bin/convert_cpu ./test/pic16k.jpg ./test/output.jpg
./bin/convert_cuda ./test/pic16k.jpg ./test/output.jpg
