#!/bin/bash
date
for duration in "imme" "second" "short" "middle" "long"; do
    for idx in $(seq 0 7); do  
        srun -p videop1 -n1 -N1 --gres=gpu:1 --quotatype auto --async -J "vl-$idx-$duration" python longshort-off.py "$duration" "$idx" 8
        sleep 3
    done
done

