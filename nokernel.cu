/* -*- mode: c++ -*- */

#include <stdio.h>
#include <cuda.h>
#include "amcommon.h"

int main(int argc, char *argv[]) {
  int *y;
  int read = 0;

  if(argc == 2)
    read = atoi(argv[1]);

  // ALLOC y
  allocate_managed((void **) &y, 1, sizeof(int));

  // RDNTRF
  // WRITE CPU y
  y[0] = 0;

  // WAIT
  //cudaDeviceSynchronize();

  if(read) {
    // the actual test should not execute this

    printf("y[0] = %d\n", y[0]);
  } else {
    printf("y was not read\n");
  }
  
  // to flush profiler log
  cudaDeviceSynchronize();

  // expected: 0 H2D, 0 D2H
  
}
