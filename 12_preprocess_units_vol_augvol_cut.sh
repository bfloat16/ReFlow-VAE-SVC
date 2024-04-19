#!/bin/bash

python 12_preprocess_units_vol_augvol_cut.py --filelist filelist_0.txt &
python 12_preprocess_units_vol_augvol_cut.py --filelist filelist_1.txt &
python 12_preprocess_units_vol_augvol_cut.py --filelist filelist_2.txt &

wait