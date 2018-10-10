/* -*- mode: c++ -*- */

#include <stdio.h>
#include <cuda.h>
#include "amcommon.h"

// slightly more complicated version of gpu_ro.cu to defeat static
// analysis and to detect dynamic read/write detection capabilities

__global__ void test (int *x, int a) {
  if(a > 10) 
    x[0] = 1;
}

int main(int argc, char *argv[]) {
  int *y;
  int x = 1;

  // optionally take a command line argument for value of 'a'
  if(argc == 2) {
    x = atoi(argv[1]);
  } 

  // ALLOC y
  allocate_managed((void **) &y, 1024, sizeof(int));
  
  // RDNTRF
  // WRITE CPU y
  y[0] = 3;

  // TRF
  // WRITE GPU y (conditional)
  // should transfer y
  test<<<1,1>>>(y, x);
  // WAIT
  cudaDeviceSynchronize();

  /* in the standard run this is RDNTRF, though with values of a > 10, this is TRF */
  // RDNTRF
  // READ CPU y
  // only read y, should transfer only if GPU changed y
  printf("y[0] = %d\n", y[0]);

  // to flush profiler log
  cudaDeviceSynchronize();

  // expected: 0 H2D, 0 D2H (when a <= 10)
}
