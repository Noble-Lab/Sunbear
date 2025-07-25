# Sunbear
Single-cell multi-modal/multi-condition temporal inference model

![sunbear_schema](https://github.com/user-attachments/assets/86b76cb1-f63a-4f44-809d-c64e993f0e70)

## Overview
The Sunbear model performs temporal inference alongside cross-condition and cross-modality inference, which can be applied to:
1. infer how a cellular profile changes along a continuous time frame
2. compare condition-differences on time-series data with few matched conditions
3. jointly infer temporal multimodal profile changes for each cell


## Installation
Install via docker (recommended):
```
apptainer pull docker://bearfam/bears
apptainer shell --nv bears_latest.sif
```

Install via conda:
```
conda env create -f environment.yml
conda activate sunbear
```
## Example run:
```
bash ./example.sh
```

## Input data:
The code takes in [h5ad format](https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html). Here are some [example input files](https://noble.gs.washington.edu/~ranz0/Sunbear/data/).

The h5ad object for scRNA-seq consists of:
- gene expression count matrix (rna_adata.X)
- gene annotation (rna_adata.var)
- cell annotation (rna_adata.obs): rna_adata.obs needs to contain a "time" column. The model also allows for an optional "batch" and/or "condition" column.  
Example input data: 
```
$rna_h5ad=example_single_rna.h5ad
```

In the multimodal setup, Sunbear takes another scATAC-seq input in h5ad format, which consists of
- a binarized peak accessibility matrix (atac_adata.X)
- peak region annotation (atac_adata.var): must include a column indicating the chromosome so that the model can save memory and include all peak regions
- cell annotation (atac_adata.obs), similar to rna_adata.obs needs to contain a "time" column, an optional batch column, and an optional condition column.  
Example input data: 
```
$rna_h5ad=example_multi_rna.h5ad
$atac_h5ad=example_multi_atac.h5ad
```

## Basic usage:
1. temporal inference of gene expression profiles in a specific $celltype around $timepoint:
```
rna_h5ad=data/example_single_rna.h5ad
python bin/sunbear.py --domain rna --rna_h5ad $rna_h5ad --batch batch --condition sex --predict temporal --ct_query Muscle_cells --targettime 16 --celltype major_trajectory
```

2. Cross-condition inference and comparison between biological conditions: 
For cross-condition comparison, at least two conditions need to be included and both conditions should be seen during training.Here is an example to calculate differential expression between conditon F and M:
```
python bin/sunbear.py --domain rna --rna_h5ad $rna_h5ad --batch batch --condition sex --targettime 16 --predict diffexp_condition --sourcecondition M --targetcondition F 
```

3. Cross-modality temporal inference: 
We first train a multimodal temporal model on existing measurements. Then, for cells in $celltype, we jointly predict their gene expression and chromatin accessibility changes around $timepoint:
```
$rna_h5ad=example_multi_rna.h5ad
$atac_h5ad=example_multi_atac.h5ad
python bin/sunbear.py --domain multi --rna_h5ad $rna_h5ad --atac_h5ad $atac_h5ad --batch batch --predict temporal --time_range 0.1 --targettime 8.25 --ct_query Hindbrain
```

## other functions:
1. generate simulation and train on simulated data (take linear trend as an example)
```
Rscript bin/generate_simulation.R
rna_h5ad=data/simulation_linear10.2
python bin/sunbear.py --domain rna --rna_h5ad $rna_h5ad --train_ver simulation --targettime 7.5
```
