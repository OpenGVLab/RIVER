#!/bin/bash
date
for duration in "imme" "second" "short" "middle" "long"; do
    for idx in $(seq 0 7); do
        echo $idx-$duration
        srun -p videop1 -n1 -N1 --gres=gpu:1 --quotatype spot --async -J "fo-$idx-$duration" python longshort-offline.py "$duration" "$idx" 8
        sleep 2
    done
done
