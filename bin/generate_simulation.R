#!/usr/bin/env Rscript

## generate gene expression simulations to check our model's ability to estimate gene expression trend

library(data.table)
library(ggplot2)
library(dplyr)
library(anndata)

cur_dir <- './data/'

generate_distr <- function(distr, a, b, c, t){
  ## generate different types of distributions based on input parameters
  if (distr == 'constant'){
    x = abs(c)
  }else if (distr == 'linear'){
    x = 20*a*(t-7.5+sign(a))
  }else if (distr == 'sine'){
    x = 20*b*(sin(5*a*t+b)+2)
  }else if (distr == 'exp'){
    x = 5*b* exp(2*a*(t-7.5))
    if (sign(b)==-1){
      x = x + abs(5*abs(b)*exp(2*abs(a)))
    }
  }
  return(abs(x))
}


generate_simulation <- function(distr, timepoints, ncells, delta='', error_sd=1){
  ## randomly select a set of parameters to simulate distribution
  rm(exp_mat)
  a_vec = c(-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 1)
  b_vec = c(-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 1)
  a = sample(a_vec,1)
  b = sample(b_vec,1)
  c = runif(min=1, max=30, n=1)
  if (delta==''){
    delta=0
  }
  d = delta/ncells
  for (t in timepoints){
    exp_vec = c()
    ## assume cells within a cell type changes the same, except adding different random noise to different cells and genes
    for (i in 1:ncells){
      ## add random noise
      epsilon = rnorm(1, sd=error_sd)
      ## round up final value
      t_i = t-d*i
      x = generate_distr(distr, a, b, c, t_i)
      exp_vec = c(exp_vec, round(x+epsilon))
    }
    if (exists('exp_mat')){
      exp_mat = rbind(exp_mat, exp_vec)
    }else{
      exp_mat = exp_vec
    }
  }
  return(list("mat" = exp_mat, "para" = c(a,b,c,d)))
}


generate_matrix_simple <- function(distr_spec, vername, delta='', error_sd=1){
  n_cells <- 100
  n_genes <- 500
  n_total_genes <- 2000
  n_celltypes <- 4
  celltype_vec <- letters[1:n_celltypes]
  time_vec <- seq(6.5, 8.5, 0.25/4)
  expression_example_list <- vector(mode = "list", length = length(time_vec))
  gene_para_mat <- c()
  
  for (distr in c('constant')){
    for (i in 1:n_total_genes){
      output <- generate_simulation(distr, time_vec, n_cells, delta=delta, error_sd=error_sd)
      gene_para_mat <- rbind(gene_para_mat, c(distr,output$para))
      for (time_index in 1:length(time_vec)){
        expression_example_list[[time_index]] <- cbind(expression_example_list[[time_index]], output$mat[time_index,])
      }
    }
  }
  print(dim(gene_para_mat))
  print(dim(expression_example_list[[time_index]]))
  
  # for each cell type-specific time-varying changes, calculate a consistent block
  expression_example_list_ctspec <- vector(mode = "list", length = length(time_vec))
  gene_para_mat_ctspec <- c() 
  for (i in 1:n_genes){
    for (distr in c(distr_spec)){
      output <- generate_simulation(distr, time_vec, n_cells, delta=delta, error_sd=error_sd)
      gene_para_mat_ctspec <- rbind(gene_para_mat_ctspec, c(distr,output$para))
      for (time_index in 1:length(time_vec)){
        expression_example_list_ctspec[[time_index]] <- cbind(expression_example_list_ctspec[[time_index]], output$mat[time_index,])
      }
    }
  }
  max_depth <- 0
  min_depth <- 1000000
  for (time_index in 1:length(time_vec)){
    max_depth <- max(c(max_depth), apply(expression_example_list_ctspec[[time_index]], 1, sum))
    min_depth <- min(c(min_depth), apply(expression_example_list_ctspec[[time_index]], 1, sum))
  }
  
  expression_example_mat_combined <- c()
  expression_example_mat_combined_obs <- c()
  for (time_index in 1:length(time_vec)){
    for (i in 1:n_celltypes){
      expression_example_mat_i <- expression_example_list[[time_index]] # pre-set every gene to have constant values
      expression_example_mat_i[,c(n_genes*(i-1)+1):c(n_genes*i)] <- expression_example_list_ctspec[[time_index]] # replace blocks of genes (500) with defined temporal changes in the defined cell type
      expression_example_mat_combined <- rbind(expression_example_mat_combined, expression_example_mat_i)
      expression_example_mat_combined_obs <- rbind(expression_example_mat_combined_obs, cbind(1:n_cells, rep(time_vec[time_index], n_cells), rep(celltype_vec[i], n_cells)))
    }
  }
  expression_example_mat_combined_obs <- data.frame(expression_example_mat_combined_obs)
  colnames(expression_example_mat_combined_obs) <- c('cell','time','cell_type')
  
  
  ## combine the different cell types to make final matrix
  # uniform across cell types
  expression_example_mat_combined_var <- do.call("cbind", rep(list(gene_para_mat), n_celltypes))
  # add cell type specific changes
  for (i in 1:n_celltypes){
    expression_example_mat_combined_var[c(n_genes*(i-1)+1):c(n_genes*i), c((5*(i-1)+1)):c(5*i)] <- gene_para_mat_ctspec
  }
  expression_example_mat_combined_var <- cbind(1:n_total_genes, expression_example_mat_combined_var)
  colnames(expression_example_mat_combined_var) <- c('gene', paste0(rep(c('pattern_ct', 'a_ct', 'b_ct','c_ct','d_ct'), n_celltypes), rep(celltype_vec, each=5)))
  
  expression_example_mat_combined <- pmax(expression_example_mat_combined, 0)
  
  write.table(expression_example_mat_combined, paste0(cur_dir, 'simulation_', vername, error_sd, delta, '_X.txt'), sep = "\t", row.names = FALSE, col.names = FALSE, quote=FALSE)
  write.table(expression_example_mat_combined_obs, paste0(cur_dir, 'simulation_', vername, error_sd, delta, '_obs.txt'), sep = "\t", row.names = FALSE, col.names = TRUE, quote=FALSE)
  write.table(expression_example_mat_combined_var, paste0(cur_dir, 'simulation_', vername, error_sd, delta, '_var.txt'), sep = "\t", row.names = FALSE, col.names = TRUE, quote=FALSE)
  
  if (FALSE){
    ## convert to anndata
    ad <- AnnData(
      X = expression_example_mat_combined,
      obs = expression_example_mat_combined_obs,
      var = expression_example_mat_combined_var
    )
    write_h5ad(ad, "output.h5ad")
    
  }
}


## generate matrix of each cell across different time points
generate_matrix_simple('linear','linear', delta=0.2, error_sd=1) # remodeled so the increasing and decreasing trends are similar
generate_matrix_simple('exp','exp', delta=0.2, error_sd=1) # added flip of exp, so that overall cell size across time should be comparable
generate_matrix_simple('sine','sine', delta=0.2, error_sd=1) # make sine wave to be more frequent

