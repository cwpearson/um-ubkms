#pragma once

#include <cuda.h>
#include <stdlib.h>

#define PAGE_SIZE 4096  // cpu
#define INTS_PER_PAGE (PAGE_SIZE/sizeof(int))

static void allocate_managed(void **p, size_t nelem, size_t sz_elem) {
  if(cudaMallocManaged(p, nelem * sz_elem) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate managed memory.\n");
    exit(1);
  }
}
