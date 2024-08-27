#!/usr/bin/env python

"""
This code builds c-vae with time modeled as continuous variable
The output is temporal changes of genes and chromatin accessbility patterns for each cell
"""

import os, sys, argparse, random;
proj_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'));
sys.path.append(proj_dir)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import anndata as ad
import scanpy as sc
from model import *
import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt

import tensorflow as tf;
from tensorflow.python.keras import backend as K

from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.sparse import csc_matrix
from scipy.stats import sem, ranksums, wilcoxon
import math
import statistics


def train(cur_dir, rna_h5ad, atac_h5ad, sim_url, holdouttime, d_time, time_magnitude_float, embed_dim, nlayer, dropout_rate, learning_rate_x, learning_rate_y, discriminator_weight, batch_size, mse_weight, domain, batch, condition, randseed):
    """
    train/load the Sunbear model
    Parameters
    ----------
    outdir: output directory
    
    """
    if os.path.isfile(sim_url+'/checkpoint') is False:
        logging.info('Training model ' + sim_url)
        ## ======================
        ## load and preprocess data to the required format
        ## ======================
        logging.info('Loading data...')
        rna_data, atac_data, nlabel = process_adata(rna_h5ad, atac_h5ad, domain, batch, condition, d_time, time_magnitude_float)
        nsubsample = 2000
        ## =====================================
        ## split train, test, val
        ## val and test would be random subset of cells in later time points that are shared between scATAC and scRNA
        ## =====================================
        ## hold out an entire scATAC timepoint as unseen
        rna_data_test = rna_data[rna_data.obs.time==holdouttime,]
        rna_data_train = rna_data[rna_data.obs.time!=holdouttime,]
        if domain=='multi':
            chr_list = {}
            for chri in atac_data.var.chr.unique():
                chr_list[chri] = [i for i, x in enumerate(atac_data.var['chr']) if x == chri];
            atac_data_test = atac_data[atac_data.obs.time==holdouttime,]
            atac_data_train = atac_data[atac_data.obs.time!=holdouttime,]

        ## sample equal number of cells in each validation timepoint as validation set
        ## only get val and from remaining scATAC time points
        rna_data_val_index = []
        for time_i in rna_data.obs.time.unique():
            random.seed(randseed)
            rna_data_val_index.extend(random.sample(list(rna_data_train.obs.loc[rna_data_train.obs['time']==time_i].index),
            min(int(sum(rna_data_train.obs.time==time_i) * 0.2), nsubsample)))
        rna_data_val = rna_data_train[rna_data_val_index,:]

        ## use all cells that are not assigned to validation set as training set
        rna_data_train = rna_data_train[~rna_data_train.obs.index.isin(rna_data_val_index),:]

        if domain=='multi':
            # use the same set of cells in scRNA validation set
            atac_data_val_index = list(set(atac_data_train.obs.index) & set(rna_data_val_index))
            atac_data_val = atac_data_train[atac_data_val_index,:]
            atac_data_train = atac_data_train[~atac_data_train.obs.index.isin(atac_data_val_index),:]

            ## since the test set can be large too, we subsample the test set
            if atac_data_test.shape[0] > 10000:
                random.seed(randseed)
                atac_data_test = atac_data_test[random.sample(list(atac_data_test.obs.index),10000),:]

        ## input all data and shuffle indices of train and validation
        # RNA
        data_x_train, batch_x_train = randomize_adata(rna_data_train, randseed)
        data_x_val, batch_x_val = randomize_adata(rna_data_val, randseed)

        if domain=='multi':
            # ATAC
            data_y_train, batch_y_train = randomize_adata(atac_data_train, randseed)
            data_y_val, batch_y_val = randomize_adata(atac_data_val, randseed)
            
            ## build co-assay subset of validation set
            random.seed(randseed)
            random.shuffle(atac_data_val_index)
            data_x_val_co = rna_data_val[atac_data_val_index,:].X.tocsr()
            batch_x_val_co = rna_data_val[atac_data_val_index,:].obsm['encoding'].to_numpy()
            data_y_val_co = atac_data_val[atac_data_val_index,:].X.tocsr()
            batch_y_val_co = atac_data_val[atac_data_val_index,:].obsm['encoding'].to_numpy()

            atac_data_train_index = list(set(atac_data_train.obs.index) & set(rna_data_train.obs.index))
            random.seed(randseed)
            random.shuffle(atac_data_train_index)
            data_x_co = rna_data_train[atac_data_train_index,:].X.tocsr()
            batch_x_co = rna_data_train[atac_data_train_index,:].obsm['encoding'].to_numpy()
            data_y_co = atac_data_train[atac_data_train_index,:].X.tocsr()
            batch_y_co = atac_data_train[atac_data_train_index,:].obsm['encoding'].to_numpy()
        
        ## clear some variables to free up memory
        del rna_data_train
        del rna_data
        if domain=='multi':
            del atac_data_train
            del atac_data
        
        logging.info('== data imported ==')
        sys.stdout.flush()

        ## ===================================
        ## train the model
        ## ===================================
        logging.info('Training model...')
    
        ## train the model
        tf.reset_default_graph()
        if domain=='multi':
            autoencoder = AEtimeMulti(input_dim_x=data_x_train.shape[1], batch_dim_x=batch_x_train.shape[1], d_time=d_time, embed_dim=embed_dim, nlayer=nlayer, dropout_rate=dropout_rate, learning_rate_x=learning_rate_x, learning_rate_y=learning_rate_y, input_dim_y=data_y_train.shape[1], batch_dim_y=batch_y_train.shape[1], chr_list=chr_list, nlabel=nlabel, discriminator_weight=discriminator_weight,  mse_weight=mse_weight);
            
            iter_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_reconstr_y_loss_list, val_kl_y_loss_list, val_translator_xy_loss_list, val_translator_yx_loss_list = autoencoder.train(sim_url, data_x_train, batch_x_train, data_x_val, batch_x_val, data_y_train, batch_y_train, data_y_val, batch_y_val, data_x_co, data_y_co, batch_x_co, batch_y_co, data_x_val_co, data_y_val_co, batch_x_val_co, batch_y_val_co, batch_size, nlayer, d_time, dropout_rate)
            
            ## write and plot loss per epoch
            if len(iter_list)>0:
                fout = open(sim_url+'_loss_per_epoch.txt', 'w')
                for i in range(len(iter_list)):
                    fout.write(str(iter_list[i])+'\t'+str(val_reconstr_x_loss_list[i])+'\t'+str(val_kl_x_loss_list[i])+'\t'+str(val_reconstr_y_loss_list[i])+'\t'+str(val_kl_y_loss_list[i])+'\t'+str(val_translator_xy_loss_list[i])+'\t'+str(val_translator_yx_loss_list[i])+'\n')
                    
                fout.close()

                ## plot loss curve
                fig = plt.figure(figsize = (10,5))
                fig.subplots_adjust(hspace=.4, wspace=.4)
                ax = fig.add_subplot(1,2,1)
                ax.plot(iter_list, val_reconstr_x_loss_list, color='orange', marker='.', markersize=0.5, alpha=1, label='reconstr');
                ax.plot(iter_list, val_translator_yx_loss_list, color='blue', marker='.', alpha=1, label='trans');
                plt.legend(loc='upper right')
                plt.title('scRNA loss')

                ax = fig.add_subplot(1,2,2)
                ax.plot(iter_list, val_reconstr_y_loss_list, color='orange', marker='.', alpha=1, label='reconstr');
                ax.plot(iter_list, val_translator_xy_loss_list, color='blue', marker='.', alpha=1, label='trans');
                plt.legend(loc='upper right')
                plt.title('scATAC loss')

                fig.savefig(sim_url+ '_loss.png')

        if domain=='rna':
            autoencoder = AEtimeRNA(input_dim_x=data_x_train.shape[1], batch_dim_x=batch_x_train.shape[1], embed_dim_x=embed_dim, nlayer=nlayer, dropout_rate=dropout_rate, output_model=sim_url, learning_rate_x=learning_rate_x, nlabel=nlabel, discriminator_weight=discriminator_weight);
            iter_list, reconstr_x_loss_list, kl_x_loss_list, discriminator_x_loss_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_discriminator_x_loss_list = autoencoder.train(sim_url, data_x_train, batch_x_train, data_x_val, batch_x_val, batch_size=batch_size, nlayer=nlayer, dropout_rate=dropout_rate);

            ## write and plot loss per epoch
            if len(iter_list)>0:
                fout = open(sim_url+'_loss_per_epoch.txt', 'w')
                for i in range(len(iter_list)):
                    fout.write(str(iter_list[i])+'\t'+str(reconstr_x_loss_list[i])+'\t'+str(kl_x_loss_list[i])+'\t'+str(discriminator_x_loss_list[i])+'\t'+str(val_reconstr_x_loss_list[i])+'\t'+str(val_kl_x_loss_list[i])+'\t'+str(val_discriminator_x_loss_list[i])+'\n')
                fout.close()
            
                ## plot loss per epoch curve
                fig = plt.figure(figsize = (10,5))
                fig.subplots_adjust(hspace=.4, wspace=.4)
                ax = fig.add_subplot(1,2,1)
                ax.plot(iter_list, reconstr_x_loss_list, color='blue', marker='.', markersize=0.5, alpha=1, label='train');
                ax.plot(iter_list, val_reconstr_x_loss_list, color='orange', marker='.', markersize=0.5, alpha=1, label='val');
                plt.legend(loc='upper right')
                plt.title('reconstr loss')

                ax = fig.add_subplot(1,2,2)
                ax.plot(iter_list, kl_x_loss_list, color='blue', marker='.', alpha=1, label='train');
                ax.plot(iter_list, val_kl_x_loss_list, color='orange', marker='.', alpha=1, label='val');
                plt.legend(loc='upper right')
                plt.title('KL loss')

                fig.savefig(sim_url+ '_loss.png')


        ## ===================================
        ## output metrics for model selection based on validation set
        ## ===================================
        if os.path.isfile(sim_url+'_validation.txt') is False:
            ## load the model
            logging.info('Loading model with dropout_rate=0...')
            tf.reset_default_graph()
            if domain=='multi':
                autoencoder = AEtimeMulti(input_dim_x=data_x_train.shape[1], batch_dim_x=batch_x_train.shape[1], d_time=d_time, embed_dim=embed_dim, nlayer=nlayer, dropout_rate=0, learning_rate_x=learning_rate_x, learning_rate_y=learning_rate_y, input_dim_y=data_y_train.shape[1], batch_dim_y=batch_y_train.shape[1], chr_list=chr_list, nlabel=nlabel, discriminator_weight=discriminator_weight, mse_weight=mse_weight);
                iter_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_reconstr_y_loss_list, val_kl_y_loss_list, val_translator_xy_loss_list, val_translator_yx_loss_list = autoencoder.train(sim_url,data_x_train, batch_x_train, data_x_val, batch_x_val, data_y_train, batch_y_train, data_y_val, batch_y_val, data_x_co, data_y_co, batch_x_co, batch_y_co, data_x_val_co, data_y_val_co, batch_x_val_co, batch_y_val_co,  batch_size, nlayer, d_time, dropout_rate)

                ## =================
                ## output evaluation metrics on validation set
                ## =================
                sim_metric_val = []
                ## get translation loss between rna and atac in validation set
                loss_mat = pd.read_csv(sim_url+'_loss_per_epoch.txt', delimiter='\t')
                loss_mat.columns = ['iter','val_rna_reconstr','val_rna_kl','val_atac_reconstr', 'val_atac_kl', 'val_trans_atac', 'val_trans_rna']
                val_translator_xy_loss_list = loss_mat['val_trans_atac'].values.tolist()
                val_translator_yx_loss_list = loss_mat['val_trans_rna'].values.tolist()

                patience = 20
                sim_metric_val.append(val_translator_yx_loss_list[-(patience+2)])
                sim_metric_val.append(val_translator_xy_loss_list[-(patience+2)])

                ## embedding alignment across time, batch and data modalities.
                lisi_score_vec = calc_embedding(autoencoder, rna_data_val, atac_data_val, batch_size, d_time, 'lisi', domain, sim_url)
                
                sim_metric_val = sim_metric_val + lisi_score_vec
                np.savetxt(sim_url+'_validation.txt', sim_metric_val, delimiter='\n', fmt='%1.10f')

            if domain=='rna':
                autoencoder = AEtimeRNA(input_dim_x=data_x_train.shape[1], batch_dim_x=batch_x_train.shape[1], embed_dim_x=embed_dim, nlayer=nlayer, dropout_rate=0, output_model=sim_url, learning_rate_x=learning_rate_x, nlabel=nlabel, discriminator_weight=discriminator_weight);
                iter_list, reconstr_x_loss_list, kl_x_loss_list, discriminator_x_loss_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_discriminator_x_loss_list = autoencoder.train(sim_url, data_x_train, batch_x_train, data_x_val, batch_x_val, batch_size=batch_size, nlayer=nlayer, dropout_rate=0);

                ## =================
                ## output evaluation metrics on validation set
                ## =================
                eval_model_rna(sim_url, rna_data_val, holdouttime)



def pred(cur_dir, rna_h5ad, atac_h5ad, sim_url, holdouttime, d_time, time_magnitude_float, embed_dim, nlayer, dropout_rate, learning_rate_x, learning_rate_y, discriminator_weight, batch_size, mse_weight, domain, batch, condition, randseed, targettime, sourcecondition, targetcondition, ct_query, predict, time_step, time_range):
    """
    downstream applications
    Parameters
    ----------
    outdir: output directory
    
    """
    rna_data, atac_data, nlabel = process_adata(rna_h5ad, atac_h5ad, domain, batch, condition, d_time, time_magnitude_float)
    if os.path.isfile(sim_url+'/checkpoint') is True:
        ## load the model
        logging.info('Loading model '+ sim_url)
        tf.reset_default_graph()
        if domain=='multi':
            chr_list = {}
            for chri in atac_data.var.chr.unique():
                chr_list[chri] = [i for i, x in enumerate(atac_data.var['chr']) if x == chri];
            autoencoder = AEtimeMulti(input_dim_x=rna_data.X.shape[1], batch_dim_x=rna_data.obsm['encoding'].shape[1], d_time=d_time, embed_dim=embed_dim, nlayer=nlayer, dropout_rate=0, learning_rate_x=learning_rate_x, learning_rate_y=learning_rate_y, input_dim_y=atac_data.X.shape[1], batch_dim_y=atac_data.obsm['encoding'].shape[1], chr_list=chr_list, nlabel=nlabel, discriminator_weight=discriminator_weight, mse_weight=mse_weight);
            autoencoder.train(sim_url)

        if domain=='rna':
            rna_data.obs['celltype'] = rna_data.obs.major_trajectory
            autoencoder = AEtimeRNA(input_dim_x=rna_data.X.shape[1], batch_dim_x=rna_data.obsm['encoding'].shape[1], embed_dim_x=embed_dim, nlayer=nlayer, dropout_rate=0, output_model=sim_url, learning_rate_x=learning_rate_x, nlabel=nlabel, discriminator_weight=discriminator_weight);
            autoencoder.train(sim_url)

        if domain == 'rna':
            if predict=='temporal':
                ## ===================================
                ## predict single cell profiles across time
                ## ===================================
                calc_temporal_exp(sim_url, autoencoder, domain, rna_data, atac_data, ct_query, targettime, time_step, time_range, d_time, time_magnitude_float)

            if predict=='diffexp_condition':
                ## query cell
                rna_data_i = rna_data[rna_data.obs.time==targettime, :]
                            
                ## generate swap_encoding
                swap_encoding_target = rna_data_i.obsm['encoding'].to_numpy()
                batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(targetcondition), dataset_list=list(rna_data.obs['condition'].unique())).todense()) # TODO: replace only the condition encoding from the query
                swap_encoding_target[:, (-nlabel - batch_encoding.shape[1]): (-nlabel)] = np.tile(batch_encoding.to_numpy(), (rna_data_i.shape[0],1))

                swap_encoding_source = rna_data_i.obsm['encoding'].to_numpy()
                batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(sourcecondition), dataset_list=list(rna_data.obs['condition'].unique())).todense()) # TODO: replace only the condition encoding from the query
                swap_encoding_source[:, (-nlabel - batch_encoding.shape[1]): (-nlabel)] = np.tile(batch_encoding.to_numpy(), (rna_data_i.shape[0],1))

                calc_condition_diffexp(sim_url + '_' + str(targettime) + sourcecondition+ targetcondition , autoencoder, rna_data_i, swap_encoding_source, swap_encoding_target)

        ## make predictions of dynamic peak accessibility and gene expression changes for scRNA-seq query
        if domain == 'multi':
            if predict=='temporal':
                ## ===================================
                ## calculate multimodal temporal patterns
                ## ===================================
                calc_temporal_exp(sim_url, autoencoder, domain, rna_data, atac_data, ct_query, targettime, time_step, time_range, d_time, time_magnitude_float)
        

        
def main(args):
    cur_dir = args.cur_dir
    rna_h5ad = args.rna_h5ad
    atac_h5ad = args.atac_h5ad
    learning_rate_x = args.learning_rate_x;
    learning_rate_y = args.learning_rate_y;
    embed_dim = args.embed_dim;
    dropout_rate = args.dropout_rate;
    nlayer = args.nlayer;
    batch_size = args.batch_size
    holdouttime = args.holdouttime
    targettime = args.targettime
    d_time = args.d_time
    discriminator_weight = args.discriminator_weight
    mse_weight = args.mse_weight
    time_magnitude = args.time_magnitude
    sourcecondition = args.sourcecondition
    targetcondition = args.targetcondition
    predict = args.predict
    domain = args.domain
    batch = args.batch
    condition = args.condition
    randseed = args.randseed
    ct_query = args.ct_query
    time_step = args.time_step
    time_range = args.time_range
    if time_magnitude.endswith('d'): ## if time is specified to vary according to unit of days
        time_magnitude_float = 2*np.pi/float(time_magnitude[:-1])
    else:
        time_magnitude_float = float(time_magnitude)

    out_dir = cur_dir + '/output/'+ domain+'/'
    os.system('mkdir -p '+ out_dir)
    sim_url = out_dir+ 'model_time'+ str(holdouttime).rstrip('0').rstrip('.')+ '_'+ str(time_magnitude)+ '_'+ str(d_time)+ '_ndim'+str(embed_dim)+ '_mse'+ str(mse_weight)+ '_rand'+ str(randseed)

    ## train the model if it doesn't exist:
    train(cur_dir, rna_h5ad, atac_h5ad, sim_url, holdouttime, d_time, time_magnitude_float, embed_dim, nlayer, dropout_rate, learning_rate_x, learning_rate_y, discriminator_weight, batch_size, mse_weight, domain, batch, condition, randseed)
    
    ## make predictions
    pred(cur_dir, rna_h5ad, atac_h5ad, sim_url, holdouttime, d_time, time_magnitude_float, embed_dim, nlayer, dropout_rate, learning_rate_x, learning_rate_y, discriminator_weight, batch_size, mse_weight, domain, batch, condition, randseed, targettime, sourcecondition, targetcondition, ct_query, predict, time_step, time_range)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--domain', type=str, help='rna or multi', default='multi');
    parser.add_argument('--cur_dir', type=str, help='cur_dir', default='.');
    parser.add_argument('--ct_query', type=str, help='ct_query', default='');
    parser.add_argument('--rna_h5ad', type=str, help='rna_h5ad', default='');
    parser.add_argument('--batch', type=str, help='batch column name in anndata', default='');
    parser.add_argument('--condition', type=str, help='condition column name in anndata', default='');
    parser.add_argument('--atac_h5ad', type=str, help='atac_h5ad', default='');
    parser.add_argument('--holdouttime', type=float, help='hold-out timepoint', default=0);
    parser.add_argument('--targettime', type=float, help='target timepoint', default=0);
    parser.add_argument('--sourcecondition', type=str, help='sourcecondition', default='');
    parser.add_argument('--targetcondition', type=str, help='targetcondition', default='');
    parser.add_argument('--d_time', type=int, help='d_time', default=50);
    parser.add_argument('--nlayer', type=int, help='nlayer', default=2);
    parser.add_argument('--batch_size', type=int, help='batch size', default=8);
    parser.add_argument('--dropout_rate', type=float, help='dropout_rate for hidden layers of autoencoders', default=0.1);
    parser.add_argument('--require_improvement', type=int, help='require_improvement', default=20);
    parser.add_argument('--embed_dim', type=int, help='embed_dim', default=25);
    parser.add_argument('--learning_rate_x', type=float, help='learning_rate_x', default=0.001);
    parser.add_argument('--learning_rate_y', type=float, help='learning_rate_x', default=0.001);
    parser.add_argument('--randseed', type=int, help='randseed', default=101);
    parser.add_argument('--discriminator_weight', type=float, help='discriminator_weight', default=1);
    parser.add_argument('--mse_weight', type=float, help='mse_weight', default=10000);
    parser.add_argument('--time_magnitude', type=str, help='time_magnitude', default='1d');
    parser.add_argument('--time_step', type=float, help='time_step', default=0.02);
    parser.add_argument('--time_range', type=float, help='time_range', default=1);
    parser.add_argument('--predict', type=str, help='make prediction on dynamic correlation ("predict" if yes, "" if no)', default='');

    args = parser.parse_args();
    main(args);
