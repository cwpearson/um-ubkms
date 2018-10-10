#!/bin/bash

ALL="automem privmem multivars readonly nokernel"

function r {
for a in $ALL; do
    ./run.sh $a
done;

O1=output-logs-`date +%Y-%m-%d-%H-%M-%S`
tar cvf $O1.tar --remove-files *.output.txt *.log
}

r

