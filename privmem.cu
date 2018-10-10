/* -*- mode: c++ -*- */

#include <stdio.h>
#include <cuda.h>
#include "amcommon.h"

__global__ void test1 (int *x) {
  x[0] = 1;
}

__global__ void test2 (int *x) {
  x[0] *= 2;	   
}

int main(void) {
  int *y;

  // ALLOC y
  allocate_managed((void **) &y, 1, sizeof(int));

  // WRITE GPU y
  test1<<<1,1>>>(y);
  // WAIT
  cudaDeviceSynchronize(); // technically not needed, and possibly triggers some CPU-side work
  // WRITE GPU y
  test2<<<1,1>>>(y);
  // WAIT
  cudaDeviceSynchronize();

  // to flush profiler log
  cudaDeviceSynchronize();

  // expected: 0 H2D, 0 D2H
}
