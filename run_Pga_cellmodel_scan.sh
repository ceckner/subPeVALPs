#!/bin/bash

ma_vals='1.0'
gagg_vals='1.0 2.0'

for m in $ma_vals
do  
    for g in $gagg_vals
    do
            nice -n 5 python3 run_ALP_cellmodel_propagation_gammaALPs.py -f 'photon_alp_conv_probability_ma_'$m'neV_gagg_'$g'e-11_upto_z_10_1GeV_10PeVtest_file_new_gammaALPs.dat' -m $m -g $g -ne 0.05 -B 5.0 -sB 3.0 -L 1.0 -R 10.0
            wait
    done
done 
