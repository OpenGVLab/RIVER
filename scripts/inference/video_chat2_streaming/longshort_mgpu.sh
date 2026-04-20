#!/bin/bash
date

for winsize in 16;do
for duration in "imme" "second" "short" "middle" "long"; do
    for idx in $(seq 0 7); do   
    # for idx in $(seq 0 0); do   
        echo vc-$idx-$duration-W$winsize-$idx-$duration
        # srun -p videop1 -n1 -N1 --gres=gpu:1 --quotatype spot --async -J "$duration-$idx" python longshort_mgpu_openend.py "$duration" "$idx" 8
        srun -p videop1 -n1 -N1 --gres=gpu:1 --quotatype auto --async -J "vc-W$winsize-M16-$idx-$duration"  -o ~/outs/W$winsize-M16-$idx-$duration.log\
            python longshort_mgpu.py "$duration" "$idx" 8 $winsize 16
        sleep 5
    done
done
done

