#!/usr/bin/env python

import sys
import csv
import StringIO

unit_conv = {"B" : 1}
units = set(unit_conv.keys())

a = sys.argv[1]
HTML=True
if len(sys.argv) == 3:
    if sys.argv[2] == "-p":
        HTML=False

x = open(a, "r").readlines()

out = ""
for i, l in enumerate(x):
    if "Profiling result:" in l:
        out = "\n".join(x[i+1:])
        break

indata = StringIO.StringIO(out)
properties = {}

for r in csv.DictReader(indata):
    name = r['Name']
    if "Unified Memory" in name:
        if "page faults" in name:
            properties[name] = int(r['Unified Memory']) # Unified memory page faults is already a counter
        elif 'Memcpy' in r['Name']:
            value, unit = r['Unified Memory'].split(' ', 1)
            
            assert unit in units, "Unsupported unit '%s'" % (unit,)

            properties[name] = int(value) * unit_conv[unit] # Unified memory Memcpy HtoD/DtoH are already counters

props = properties.keys()
props.sort()

if HTML: print "<table>"
for p in props:    
    if HTML: 
        print "<tr> <th align=left>%s</th> <td align=right>%s</td> </tr>" % (p.replace("[", "").replace("]", ""), properties[p])
    else:
        print "%s|%s" % (p.replace("[", "").replace("]", ""), properties[p])

if HTML: print "</table>"
