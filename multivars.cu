
/* -*- mode: c++ -*- */

#include <stdio.h>
#include <cuda.h>
#include "amcommon.h"

__global__ void test (int *x, int a) {
  x[0] = a;
}

int main(void) {
  int *y, *z;

  // ALLOC y
  allocate_managed((void **) &y, 1024, sizeof(int));
  // ALLOC z
  allocate_managed((void **) &z, 160*1024, sizeof(int));
  
  // RDNTRF
  // WRITE CPU z
  for(int i = 0; i < 160*1024; i += INTS_PER_PAGE)
    z[i] = i;
  
  // RDNTRF
  // WRITE GPU y
  // should not transfer y (uninitialized) or z (not used)
  test<<<1,1>>>(y, 1);
  // WAIT
  cudaDeviceSynchronize();

  // TRF
  // READ CPU y
  // only read y, should transfer only y
  printf("y[0] = %d\n", y[0]);

  // RDNTRF
  // READ CPU z
  // should not transfer z (unchanged)
  for(int i = 0; i < 160*1024; i += INTS_PER_PAGE)
    printf("%d %d\n", i, z[i]);

  // to flush profiler log
  cudaDeviceSynchronize();

  // expected: 0 H2D, 4096 D2H
}
