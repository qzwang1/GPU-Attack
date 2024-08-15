nvcc -arch=sm_80 -O0 -Xptxas -O0 simple.cu -o simple
rm simple.as
cuobjdump -sass simple > simple.as
./simple