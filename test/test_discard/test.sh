nvcc -arch=sm_80 -O0 -Xptxas -O0 test.cu -o test
rm test.as
cuobjdump -sass test > test.as
./test