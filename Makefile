ALL=automem privmem multivars readonly nokernel tarball

all: $(ALL)

%: %.cu
	nvcc -cudart shared -arch sm_30 $< -o $@

tarball: um-ubmks.tar.bz2

um-ubmks.tar.bz2: automem.cu privmem.cu multivars.cu readonly.cu nokernel.cu runall.sh run.sh Makefile
	hg archive um-ubmks.tar.bz2
