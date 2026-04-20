#!/bin/bash
date

for winsize in 64;do
for duration in "second" "short" "middle" "long"; do
    for idx in $(seq 0 7); do
        echo M$memsize-W16-$idx-$duration
        srun -p video5 -n1 -N1 --gres=gpu:1 --quotatype spot --async -J "lav-$idx-$duration" python longshort.py "$duration" "$idx" 8
            python longshort.py "$duration" "$idx" 8 $winsize 16
        sleep 5
    done
done
done


