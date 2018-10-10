/* -*- mode: c++ -*- */

#include <stdio.h>
#include <cuda.h>
#include "amcommon.h"

__global__ void test (int *x) {
  x[0] = x[0] + 1;
}

int main(void) {
  int *y;
  cudaDeviceProp cu;
  cudaError_t err;
  int version;

  // use CUDA_VISIBLE_DEVICES to control the actual device
  // in a system with multiple GPUs

  if((err = cudaGetDeviceProperties(&cu, 0)) == cudaSuccess) {
    printf("GPU: %s (%d.%d)\n", cu.name, cu.major, cu.minor);
    printf("Managed memory supported: %d\n", cu.managedMemory);

    if(cudaDriverGetVersion(&version) == cudaSuccess) 
      printf("Driver: %d\n", version);    
    else
      fprintf(stderr, "Unable to get driver version, error: %d\n", cudaGetLastError());

    if(cudaRuntimeGetVersion(&version) == cudaSuccess) 
      printf("Runtime: %d\n", version);    
    else
      fprintf(stderr, "Unable to get runtime version, error: %d\n", cudaGetLastError());
  } else {
    fprintf(stderr, "Unable to get CUDA device properties, error: %d (%s)\n", err, cudaGetErrorString(err));
    exit(1);
  }

  // ALLOC y
  allocate_managed((void **) &y, 1, sizeof(int));

  // RDNTRF
  // WRITE CPU y
  y[0] = 0;

  // TRF
  // RW GPU y
  test<<<1,1>>>(y);

  // WAIT
  cudaDeviceSynchronize();
  
  // TRF
  // READ CPU y
  printf("y[0] = %d\n", y[0]);
  if(y[0] == 1) {
    fprintf(stderr, "Unified Memory is working fine.\n");
  }

  // to flush profiler log
  cudaDeviceSynchronize();
}
