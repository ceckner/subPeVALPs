#!/bin/bash

masses='1.0'
GMF='jansson12c'
g_agg='1.0 2.0'

for M in $masses
do  
    for g in $g_agg
    do
        for B in $GMF
        do
            nice -n 5 python3 calculate_GMF_conv_prob.py $M 1 0 $B $g
            wait
            nice -n 5 python3 calculate_GMF_conv_prob.py $M 0 0 $B $g
            wait
            nice -n 5 python3 calculate_GMF_conv_prob.py $M 0 1 $B $g
            rm -rf *_1.fits *_0.fits
        done
    done
done 
