#!/bin/bash

if [ $# -lt 1 ]; then
	echo USage: $0 program_name [opts...]
	exit 1;
fi;

P=$1
shift;


nvprof --csv --log-file $P.log --unified-memory-profiling per-process-device --print-gpu-trace -- ./$P "$@" > $P.output.txt
./summ.py $P.log

