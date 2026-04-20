#!/bin/bash
for duration in "imme"; do
    for idx in {0..7}; do   
        srun -p videop1 -n1 -N1 --gres=gpu:1 --quotatype auto --async -J "$duration-$idx" \
            python offline.py "$duration" "$idx" 8
        sleep 2
    done
done
