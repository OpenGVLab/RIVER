#!/bin/bash
date


for memsize in 16;do
for duration in "imme" "second" "short" "middle" "long"; do
    for idx in $(seq 0 7); do  
        srun -p videop1 -n1 -N1 --gres=gpu:1 --quotatype auto --async -J "iv-$idx-$duration" -o ~/outs-iv25/M$memsize-W16-$idx-$duration.log\
            python longshort.py "$duration" "$idx" 8 $memsize
        sleep 5
    done
done
done
