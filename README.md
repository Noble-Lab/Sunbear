# Sunbear
Single-cell multi-modal/multi-condition temporal inference model

## Overview
The Sunbear model performs temporal inference alongside cross-condition and cross-modality inference, which can be applied to:
1. infer how a cellular profile changes along a continuous time frame
2. compare condition-differences on time-series data with few matched conditions
3. jointly infer temporal multimodal profile changes for each cell

## Installation
```
conda env create -f environment.yml
conda activate sunbear
```
## Example run:
```
bash ./example.sh
```

## Input data:
The code takes in h5ad format (ref https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html).

The h5ad object for scRNA-seq consists of:
- gene expression count matrix (rna_adata.X)
- gene annotation (rna_adata.var)
- cell annotation (rna_adata.obs): rna_adata.obs needs to contain a "time" column, an optional "batch" column, and a condition column.  
Example input data: ../data/example_single_rna.h5ad

In the multimodal setup, Sunbear takes another scATAC-seq input in h5ad format, which consists of
- a binarized peak accessibility matrix (atac_adata.X)
- peak region annotation (atac_adata.var): must include a column indicating the chromosome so that the model can save memory and include all peak regions
- cell annotation (atac_adata.obs), similar to rna_adata.obs needs to contain a "time" column, an optional batch column, and an optional condition column.  
Example input data: ../data/example_multi_rna.h5ad ../data/example_multi_atac.h5ad

## Basic usage:
1. temporal inference of scRNA-seq profile in a specific $celltype around a $timepoint:
```
python ./sunbear.py --domain rna --rna_h5ad $rna_h5ad --predict temporal --ct_query $celltype --targettime $timepoint
```

3. Cross-condition inference and comparison (e.g., between Female and Male): 
For cross-condition comparison, two conditions need to be specified and should be within the conditions used during training:
```
python ./sunbear.py --domain rna --rna_h5ad $rna_h5ad --condition sex --predict diffexp_condition --sourcecondition M --targetcondition F
```

5. Cross-modality temporal inference: 
Train multimodal temporal model. Then, for cells in $celltype, predict their gene expression and chromatin accessibility changes around $timepoint:
```
python ./sunbear.py --domain multi --rna_h5ad $rna_h5ad --atac_h5ad $atac_h5ad --predict temporal --targettime $timepoint --ct_query $celltype --time_range 0.1 --time_step 0.01
```

