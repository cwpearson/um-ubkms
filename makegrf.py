#!/usr/bin/env python

import sys
import re

def read_src(f):
    out = []
    for l in open(f, "r"):
        x = rw_re.search(l)
        if not x: 
            x = wait_re.search(l)
            if not x:
                x = alloc_re.search(l)
                if not x:
                    x = trf_re.search(l)

        if x:
            out.append((l, x))

    return out
    
def gen_grf(nodes, o):
    def process(last, n, es, is_cpu = True):

        if es and is_cpu:
            elbl = "DtoH"
        else:
            elbl = "HtoD"

        if es == "TRF":
            ess = ' [label="%s",style=bold,color=green];' % (elbl,)
        elif es == "RDNTRF":
            ess = ' [label="%s",style=dashed,color=red];' % (elbl,)
        else:
            ess = ""

        if last != -1:
            print >>of, "%d -> %d%s;" % (last, n, ess)

        return n, ""

    last_gpu = -1
    last_cpu = -1
    prev_node_was = ""
    edge_style = ""

    of = open(o, "w")

    print >>of, """digraph g{
    rankdir="LR";
    colorscheme=set312;
"""

    for n, (l, x) in enumerate(nodes):
        cmd = x.group(1)
        if cmd == "ALLOC":
            print >>of, n, '[label="ALLOC %s",group=cpu,shape=box,style=filled,fillcolor="/set312/9"];' % (x.group(2))
            last_cpu, edge_style = process(max(last_cpu, last_gpu), n, edge_style)

        elif cmd == "WAIT":
            print >>of, n, '[label="SYNC",group=sync,shape=circle,style=filled,fillcolor="/set312/2"];'
            process(min(last_cpu, last_gpu), n, edge_style)
            last_cpu, edge_style = process(max(last_cpu, last_gpu), n, edge_style)

        elif cmd == "READ" or cmd == "WRITE" or cmd == "RW":
            grp = x.group(2).lower()
            cmd = x.group(1)

            glbl = "" if grp == "cpu" else "GPU: "
            shp = "box" if grp == "cpu" else "ellipse"
            #fillcolor = "darkolivegreen3" if cmd == "WRITE" else "lightskyblue"
            fillcolor = "4" if cmd == "WRITE" else "7"

            if cmd == "RW": 
                cmd = "READ/WRITE"
                fillcolor = "6"

            print >>of, n, '[label="%s%s %s",group=%s,shape=%s,style=filled,fillcolor="/set312/%s"];' % (glbl, cmd, x.group(3), grp, shp, fillcolor)
            
            if grp == "cpu": 
                last_cpu, edge_style = process(max(last_cpu, last_gpu), n, edge_style)
            else:
                last_gpu, edge_style = process(max(last_cpu, last_gpu), n, edge_style, False)
        elif cmd == "TRF" or cmd == "RDNTRF":
            edge_style = cmd

    print >>of, "}"

rw_re = re.compile("(READ|WRITE|RW) (CPU|GPU) (.+)")
wait_re = re.compile("(WAIT)")
alloc_re = re.compile("(ALLOC) (.+)")
trf_re = re.compile("(TRF|RDNTRF)")

f = sys.argv[1]
o = sys.argv[2]

n = read_src(f)
gen_grf(n, o)

