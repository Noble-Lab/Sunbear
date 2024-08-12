#!/usr/bin/bash

## example 1: temporal infernece and then predict temporal changes of cells in Muscles around E16
rna_h5ad=/net/noble/vol1/home/ranz0/proj/2021_ranz0_sc-time/bin//data/rna_time_sex.h5ad
#rna_h5ad=./data/example_single_rna.h5ad
python ./sunbear.py --domain rna --rna_h5ad $rna_h5ad --batch batch --condition sex --predict temporal --ct_query Muscle_cells --targettime 16

## example 2: compare expression difference between conditions (i.e., sexes) and return sex-difference prediction for each cell type at E16
#rna_h5ad=./data/example_single_rna.h5ad
#python ./sunbear.py --domain rna --rna_h5ad $rna_h5ad --batch batch --condition sex --targettime 16 --predict diffexp_condition --sourcecondition M --targetcondition F 

## example 3: cross-modality temporal inference and return temporal profile changes for cells in Hindbrain trajectory around E8.25
#rna_h5ad=./data/example_multi_rna.h5ad
#atac_h5ad=./data/example_multi_rna.h5ad
rna_h5ad=/net/noble/vol1/home/ranz0/proj/2021_ranz0_sc-time/bin//data/rna_time.h5ad
atac_h5ad=/net/noble/vol1/home/ranz0/proj/2021_ranz0_sc-time/bin//data/atac_time.h5ad
#python ./sunbear.py --domain multi --rna_h5ad $rna_h5ad --atac_h5ad $atac_h5ad --batch batch --predict temporal --time_range 0.1 --targettime 8.25 --ct_query Hindbrain
