#!/bin/sh

for i in 61 60 36
do
    python dd.py -k $i 4000 0.15
    echo "finished width$i"
done