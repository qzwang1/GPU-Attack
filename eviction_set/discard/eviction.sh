nvcc -arch=sm_80 -O0 -Xptxas -O0 eviction.cu -o test
./test