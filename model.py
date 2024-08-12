#!/usr/bin/env python

import os, sys, argparse, random;
proj_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'));
sys.path.append(proj_dir)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tensorflow as tf
import anndata as ad
import time

from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize, binarize
import scipy
from scipy.sparse import csc_matrix, coo_matrix
from scipy.stats import sem, ttest_rel, ttest_ind, ranksums, wilcoxon, find_repeats, rankdata
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## ============================================================
## NN models
## ============================================================

class AEtimeMulti:
    def __init__(self, input_dim_x, batch_dim_x, d_time, embed_dim, nlayer, dropout_rate, learning_rate_x, learning_rate_y, input_dim_y, batch_dim_y, chr_list, nlabel=2, discriminator_weight=1, mse_weight=1):
        """
        Network architecture and optimization

        Inputs
        ----------
        input_x: scRNA expression, ncell x input_dim_x, float
        input_y: scATAC expression, ncell x input_dim_y, float
        batch_x: scRNA batch factor, ncell x batch_dim_x, int
        batch_y: scATAC batch factor, ncell x batch_dim_y, int
        batch_x_decoder: batch factor to be switch to, same format as batch_x
        batch_y_decoder: batch factor to be switch to, same format as batch_y
        time_x: scRNA time factor, ncell x d_time, float
        time_y: scATAC time factor, ncell x d_time, float
        time_x_decoder: time factor to be switch to, same format as time_x
        time_y_decoder: time factor to be switch to, same format as time_y
        kl_weight_x: kl weight of the scRNA VAE that is increasing with epoch
        kl_weight_y: kl weight of the scATAC VAE that is increasing with epoch
        chr_list: dictionary using chr as keys and corresponding peak index as vals

        Parameters
        ----------
        input_dim_x: #genes, int
        input_dim_y: #peak regions, int
        batch_dim_x: dimension of batch matrix in RNA domain, int
        batch_dim_y: dimension of batch matrix in ATAC domain, int
        embed_dim: embedding dimension in VAEs, int
        learning_rate_x: scRNA VAE learning rate, float
        learning_rate_y: scATAC VAE learning rate, float
        nlayer: number of hidden layers in encoder/decoder, int, >=1
        dropout_rate: dropout rate in VAE, float
        nlabel: nlabel to be predicted by discriminator

        """
        self.nlayer = nlayer;
        self.dropout_rate = dropout_rate;
        self.discriminator_weight = discriminator_weight;
        self.mse_weight = mse_weight;
        self.d_time = d_time;

        self.input_dim_x = input_dim_x;
        self.batch_dim_x = batch_dim_x;
        self.embed_dim = embed_dim;
        self.learning_rate_x = learning_rate_x;

        self.input_dim_y = input_dim_y;
        self.batch_dim_y = batch_dim_y;
        self.chr_list = chr_list;
        self.learning_rate_y = learning_rate_y;
        self.nlabel = nlabel;
        self.hidden_frac = 4;

        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_dim_x]);
        self.batch_x = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x-self.d_time]);
        self.batch_x_decoder = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x-self.d_time]);
        self.time_x = tf.placeholder(tf.float32, shape=[None, self.d_time]);
        self.time_x_decoder = tf.placeholder(tf.float32, shape=[None, self.d_time]);
        self.kl_weight_x = tf.placeholder(tf.float32, None);

        self.input_y = tf.placeholder(tf.float32, shape=[None, self.input_dim_y]);
        self.batch_y = tf.placeholder(tf.float32, shape=[None, self.batch_dim_y-self.d_time]);
        self.batch_y_decoder = tf.placeholder(tf.float32, shape=[None, self.batch_dim_y-self.d_time]);
        self.time_y = tf.placeholder(tf.float32, shape=[None, self.d_time]);
        self.time_y_decoder = tf.placeholder(tf.float32, shape=[None, self.d_time]);
        self.kl_weight_y = tf.placeholder(tf.float32, None);


        def encoder_rna(input_data, nlayer, hidden_frac=4, reuse=tf.AUTO_REUSE):
            """
            scRNA encoder
            Parameters
            ----------
            hidden_frac: used to divide intermediate dimension to shrink the total paramater size to fit into memory
            input_data: generated from tf.concat([self.input_x, self.batch_x], 1), ncells x (input_dim_x + batch_dim_x)
            """
            with tf.variable_scope('encoder_x', reuse=tf.AUTO_REUSE):
                self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x + self.d_time) * self.embed_dim)/hidden_frac) # TODO
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                encoder_output_mean = tf.layers.Dense(self.embed_dim, activation=None, name='encoder_x_mean')(l1)
                encoder_output_var = tf.layers.Dense(self.embed_dim, activation=None, name='encoder_x_var')(l1)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min = -2000000, clip_value_max=15)
                encoder_output_var = tf.math.exp(encoder_output_var) + 0.0001
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.sqrt(encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;
            

        def decoder_rna(encoded_data, nlayer, hidden_frac=4, reuse=tf.AUTO_REUSE):
            """
            scRNA decoder
            Parameters
            ----------
            hidden_frac: intermadiate layer dim, used hidden_frac to shrink the size to fit into memory
            layer_norm_type: how we normalize layer, don't worry about it now
            encoded_data: generated from concatenation of the encoder output self.encoded_x and batch_x: tf.concat([self.encoded_x, self.batch_x], 1), ncells x (embed_dim + batch_dim_x)
            """

            self.intermediate_dim = int(math.sqrt(self.input_dim_x* self.embed_dim )/hidden_frac);
            with tf.variable_scope('decoder_x', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                px = tf.layers.Dense(self.intermediate_dim, activation=tf.nn.relu, name='decoder_x_px')(l1);
                px_scale = tf.layers.Dense(self.input_dim_x, activation=tf.nn.softmax, name='decoder_x_px_scale')(px);
                px_dropout = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_dropout')(px) 
                px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_r')(px)
                    
                return px_scale, px_dropout, px_r


        def encoder_y(input_data, nlayer, chr_list, hidden_frac=4, reuse=tf.AUTO_REUSE):
            """
            scATAC encoder; to save memory, only allow within chromosome connections for the first several layers
            """
            with tf.variable_scope('encoder_y', reuse=tf.AUTO_REUSE):
                dic_intermediate_dim = {}
                dic_l1 = {}
                dic_l2 = {}
                dic_l2_list = []
                
                for chri in chr_list.keys():
                    dic_intermediate_dim[chri] = int(math.sqrt((len(chr_list[chri]) + self.batch_dim_y) * self.embed_dim)/hidden_frac)
                    dic_l1[chri] = tf.layers.Dense(dic_intermediate_dim[chri], activation=None, name='Encoder_y_initial'+str(chri))(tf.gather(input_data, chr_list[chri]+list(range(self.input_dim_y, self.batch_dim_y+ self.input_dim_y)), axis=1)); ## for each chromosome, include time and batch factor in the first hidden layer
                    dic_l1[chri] = tf.contrib.layers.layer_norm(inputs=dic_l1[chri], center=True, scale=True);
                    dic_l1[chri] = tf.nn.leaky_relu(dic_l1[chri])
                    dic_l1[chri] = tf.nn.dropout(dic_l1[chri], rate=self.dropout_rate);

                    for layer_i in range(2, nlayer):
                        dic_l1[chri] = tf.layers.Dense(dic_intermediate_dim[chri], activation=None, name='Encoder_y_'+str(layer_i)+ '_'+ str(chri))(dic_l1[chri]);
                        dic_l1[chri] = tf.contrib.layers.layer_norm(inputs=dic_l1[chri], center=True, scale=True);
                        dic_l1[chri] = tf.nn.leaky_relu(dic_l1[chri])
                        dic_l1[chri] = tf.nn.dropout(dic_l1[chri], rate=self.dropout_rate);

                    dic_l2[chri] = tf.layers.Dense(self.embed_dim, activation=None, name='Encoder_y_end'+str(chri))(dic_l1[chri]);
                    dic_l2[chri] = tf.contrib.layers.layer_norm(inputs=dic_l2[chri], center=True, scale=True);
                    dic_l2[chri] = tf.nn.leaky_relu(dic_l2[chri])
                    dic_l2[chri] = tf.nn.dropout(dic_l2[chri], rate=self.dropout_rate);
                    dic_l2_list.append(dic_l2[chri])
                    
                l2_concatenate = tf.concat(dic_l2_list, 1)
                encoder_output_mean = tf.layers.Dense(self.embed_dim, activation=None, name='Encoder_y_mean')(l2_concatenate)
                encoder_output_var = tf.layers.Dense(self.embed_dim, activation=None, name='Encoder_y_var')(l2_concatenate)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min = -2000000, clip_value_max=15)
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.exp(0.5 * encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;


        def decoder_y(encoded_data, nlayer, chr_list, hidden_frac=4, reuse=tf.AUTO_REUSE):
            """
            scATAC decoder; to save memory, only allow within chromosome connections for the last several layers
            """
            with tf.variable_scope('decoder_y', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(self.embed_dim * 22, activation=None, name='Decoder_y_initial')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                dic_intermediate_dim_decode  = {}
                dic_l1_decode = {}
                dic_l2_decode = {}
                py = []
                for chri in chr_list.keys():
                    dic_intermediate_dim_decode[chri] = int(math.sqrt(len(chr_list[chri]) * (self.embed_dim + self.batch_dim_y))/hidden_frac)
                    dic_l2_decode[chri] = tf.layers.Dense(dic_intermediate_dim_decode[chri], activation=None, name='Decoder_y_1'+str(chri))(l1);
                    dic_l2_decode[chri] = tf.contrib.layers.layer_norm(inputs=dic_l2_decode[chri], center=True, scale=True);
                    dic_l2_decode[chri] = tf.nn.leaky_relu(dic_l2_decode[chri])
                    dic_l2_decode[chri] = tf.nn.dropout(dic_l2_decode[chri], rate=self.dropout_rate);
                    
                    for layer_i in range(2, nlayer):
                        dic_l2_decode[chri] = tf.layers.Dense(dic_intermediate_dim_decode[chri], activation=None, name='Decoder_y_2'+str(chri))(dic_l2_decode[chri]);
                        dic_l2_decode[chri] = tf.contrib.layers.layer_norm(inputs=dic_l2_decode[chri], center=True, scale=True);
                        dic_l2_decode[chri] = tf.nn.leaky_relu(dic_l2_decode[chri])
                        dic_l2_decode[chri] = tf.nn.dropout(dic_l2_decode[chri], rate=self.dropout_rate);

                    dic_l1_decode[chri] = tf.layers.Dense(len(chr_list[chri]), activation=tf.nn.sigmoid, name='Decoder_y_end'+str(chri))(dic_l2_decode[chri]);
                    py.append(dic_l1_decode[chri])
                
                py_concat = tf.concat(py, 1)
                return py_concat;
        

        def projector(encoded_data, nlayer, reuse=tf.AUTO_REUSE):
            """
            projector from cell factor + time factor to the time-encoded cell embedding
            shared across modalities to make sure cell embeddings and projected hidden layer are matched across modalities
            """
            self.intermediate_dim = self.embed_dim *2;
            with tf.variable_scope('projector_x', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='projector_x_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='projector_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
            
            return l1


        def discriminator(input_data, nlayer, nlabel, reuse=tf.AUTO_REUSE):
            """
            discriminator
            Parameters
            ----------
            input_data: the VAE embeddings
            """
            with tf.variable_scope('discriminator_dx', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(int(math.sqrt(self.embed_dim * nlabel)), activation=None, name='discriminator_dx_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(int(math.sqrt(self.embed_dim * nlabel)), activation=None, name='discriminator_dx_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                output = tf.layers.Dense(nlabel, activation=None, name='discriminator_dx_output')(l1)
                return output;
        

        ## ==========================
        ## scRNA-seq reconstruction
        ## ==========================
        self.libsize_x = tf.reduce_sum(self.input_x, 1)
        
        self.px_z_m, self.px_z_v, self.encoded_x = encoder_rna(tf.concat([self.input_x, self.batch_x[:,:-self.nlabel], self.time_x], 1), self.nlayer, self.hidden_frac);

        ## scRNA reconstruction
        self.px_projector = projector(tf.concat([self.encoded_x, self.time_x_decoder], 1), 1, self.hidden_frac)
        self.px_scale, self.px_dropout, self.px_r = decoder_rna(tf.concat([self.px_projector, self.batch_x_decoder[:,:-self.nlabel]], 1), self.nlayer, self.hidden_frac);
        
        self.px_r = tf.clip_by_value(self.px_r, clip_value_min = -2000000, clip_value_max=15)
        self.px_r = tf.math.exp(self.px_r)
        self.reconstr_x = tf.transpose(tf.transpose(self.px_scale) *self.libsize_x)
        
        ## scRNA reconstruction (from mean)
        self.px_projector_mean = projector(tf.concat([self.px_z_m, self.time_x_decoder], 1), 1, self.hidden_frac)
        self.px_scale_mean, self.px_dropout_mean, self.px_r_mean = decoder_rna(tf.concat([self.px_projector_mean, self.batch_x_decoder[:,:-self.nlabel]], 1), self.nlayer, self.hidden_frac);
        
        self.reconstr_x_mean = tf.transpose(tf.transpose(self.px_scale_mean) *self.libsize_x)
        
        ## scRNA loss
        # reconstr loss
        self.reconstr_loss_x = calc_zinb_loss(self.px_dropout, self.px_r, self.px_scale, self.input_x, self.reconstr_x)

        # KL loss
        self.kld_loss_x = tf.reduce_mean(0.5*(tf.reduce_sum(-tf.math.log(self.px_z_v) + self.px_z_v + tf.math.square(self.px_z_m) -1, axis=1)))


        ## ==========================
        ## scATAC-seq reconstruction
        ## ==========================
        self.libsize_y = tf.reduce_sum(self.input_y, 1) / 10000
        self.libsize_y = tf.clip_by_value(self.libsize_y, clip_value_min=0, clip_value_max=1)

        self.py_z_m, self.py_z_v, self.encoded_y = encoder_y(tf.concat([self.input_y, self.batch_y, self.time_y], 1), self.nlayer, self.chr_list, self.hidden_frac, batch_dim_y);
        
        ## scATAC reconstruction
        self.py_projector = projector(tf.concat([self.encoded_y, self.time_y_decoder], 1), 1, self.hidden_frac)
        self.py = decoder_y(tf.concat([self.py_projector, self.batch_y_decoder], 1), self.nlayer, self.chr_list, self.hidden_frac);
        self.py_projector_mean = projector(tf.concat([self.py_z_m, self.time_y_decoder], 1), 1, self.hidden_frac)
        self.py_mean = decoder_y(tf.concat([self.py_projector_mean, self.batch_y_decoder], 1), self.nlayer, self.chr_list, self.hidden_frac);
        self.reconstr_y = tf.transpose(tf.transpose(self.py) * self.libsize_y)

        ## scATAC loss
        # reconstruction
        bce = tf.keras.losses.BinaryCrossentropy()
        self.reconstr_loss_y = bce(self.input_y, self.reconstr_y) * self.input_dim_y
        self.kld_loss_y = tf.reduce_mean(0.5*(tf.reduce_sum(-self.py_z_v + tf.math.exp(self.py_z_v) + tf.math.square(self.py_z_m)-1, axis=1)))


        ## ==========================
        ## translation on co-assays
        ## ==========================
        ## translate to scRNA
        self.px_projector_translator = projector(tf.concat([self.py_z_m, self.time_y_decoder], 1), 1, self.hidden_frac)
        self.px_scale_translator, self.px_dropout_translator, self.px_r_translator = decoder_rna(tf.concat([self.px_projector_translator, self.batch_y_decoder], 1), self.nlayer, self.hidden_frac);
        
        self.px_r_translator = tf.clip_by_value(self.px_r_translator, clip_value_min = -2000000, clip_value_max=15)
        self.px_r_translator = tf.math.exp(self.px_r_translator)
        self.translator_reconstr_x = tf.transpose(tf.transpose(self.px_scale_translator) *self.libsize_x)
        
        self.translator_loss_x = calc_zinb_loss(self.px_dropout_translator, self.px_r_translator, self.px_scale_translator, self.input_x, self.translator_reconstr_x)

        ## translate to scATAC
        self.py_projector_translator = projector(tf.concat([self.px_z_m, self.time_x_decoder], 1), 1, self.hidden_frac)
        self.py_translator = decoder_y(tf.concat([self.py_projector_translator, self.batch_x_decoder[:,:-self.nlabel]], 1), self.nlayer, self.chr_list, self.hidden_frac);
        self.translator_reconstr_y = tf.transpose(tf.transpose(self.py_translator) *self.libsize_y)
        
        self.translator_loss_y = bce(self.input_y, self.translator_reconstr_y)* self.input_dim_y;

        # add MSE loss between cell embeddings
        self.mse_embedding = self.mse_weight *tf.compat.v1.losses.mean_squared_error(labels=self.px_z_m, predictions=self.py_z_m)

        ## ==========================
        ## discriminator across time
        ## ==========================
        self.input_label = self.batch_x[:,-self.nlabel:]
        self.output_label = discriminator(self.px_z_m, 1, self.nlabel)
        if self.nlabel==2:
            cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.discriminator_loss = cce(self.input_label, self.output_label) * self.discriminator_weight
        else:
            self.discriminator_loss = tf.compat.v1.losses.softmax_cross_entropy(self.input_label, self.output_label) * self.discriminator_weight
        ## optimizers
        self.train_vars_x = [var for var in tf.trainable_variables() if '_x' in var.name];
        self.train_vars_y = [var for var in tf.trainable_variables() if '_y' in var.name];
        self.train_vars_dx = [var for var in tf.trainable_variables() if '_dx' in var.name];

        self.loss_x = self.reconstr_loss_x + self.kl_weight_x * self.kld_loss_x
        self.loss_y = self.reconstr_loss_y + self.kl_weight_y * self.kld_loss_y
        self.optimizer_x = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=0.01).minimize(self.loss_x, var_list=self.train_vars_x );
        
        self.loss_y_generator = self.loss_y + self.translator_loss_x + self.translator_loss_y + self.mse_embedding
        self.optimizer_y = tf.train.AdamOptimizer(learning_rate=self.learning_rate_y, epsilon=0.00001).minimize(self.loss_y, var_list=self.train_vars_y);
        self.optimizer_y_generator = tf.train.AdamOptimizer(learning_rate=self.learning_rate_y, epsilon=0.00001).minimize(self.loss_y_generator, var_list=self.train_vars_y);
        
        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        self.sess.run(tf.global_variables_initializer());
        

    def train(self, output_model, data_x='', batch_x='', data_x_val='', batch_x_val='', data_y='', batch_y='', data_y_val='', batch_y_val='', data_x_co='', data_y_co='', batch_x_co='', batch_y_co='', data_x_val_co='', data_y_val_co='', batch_x_val_co='', batch_y_val_co='', batch_size='', nlayer='', d_time='', dropout_rate=0):
        """
        train in four steps, in each step, part of neural network is optimized meanwhile other layers are frozen.
        early stopping based on tolerance (patience) and maximum epochs defined in each step
        sep_train_index: 1: train scRNA autoencoder; 2: train scATAC autoencoder; 3: minimize discriminator of scRNA and scATAC

        """
        iter_list = []
        val_reconstr_x_loss_list = [];
        val_kl_x_loss_list = [];
        val_reconstr_y_loss_list = [];
        val_kl_y_loss_list = [];
        val_translator_xy_loss_list = [];
        val_translator_yx_loss_list = [];
        saver = tf.train.Saver()
        if os.path.exists(output_model+'/mymodel.meta'):
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
        else:
            patience=20
            nepoch_klstart_y=20
            last_improvement=0
            n_iter_step1 = 0 # keep track of the number of epochs optimized for RNA
            n_iter_step2 = 0 # keep track of the number of epochs optimized for ATAC

            loss_val_check_list = []

            nbatch_train = data_x.shape[0]//batch_size
            nbatch_val = data_x_val.shape[0]//batch_size

            nbatch_train_y = data_y.shape[0]//batch_size
            nbatch_val_y = data_y_val.shape[0]//batch_size
            nbatch_co = data_x_co.shape[0]//batch_size

            ## subset of validation samples with matched size across domains
            for iter in range(1, 5000):
                iter_list.append(iter)
                print('iter '+str(iter))
                sys.stdout.flush()

                ## shuffle dataset (updated)
                p = np.random.permutation(data_x.shape[0])
                data_x = data_x[p,]
                batch_x = batch_x[p,]
                p = np.random.permutation(data_x_co.shape[0])
                data_x_co = data_x_co[p,]
                data_y_co = data_y_co[p,]
                batch_x_co = batch_x_co[p,]
                batch_y_co = batch_y_co[p,]

                ## scRNA-seq model training
                n_iter_step1 +=1
                kl_weight_x_update = min(1.0, n_iter_step1/float(400))
                
                ## scATAC-seq model training
                n_iter_step2 +=1
                if n_iter_step2 < nepoch_klstart_y:
                    kl_weight_y_update = 0
                else:
                    kl_weight_y_update = min(1.0, (n_iter_step2-nepoch_klstart_y)/float(400))

                ## scRNA-seq reconstruction
                for batch_id in range(nbatch_train):
                    data_x_i = data_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),].todense()
                    batch_x_i = batch_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),]
                    self.sess.run(self.optimizer_x, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i[:,d_time:], self.time_x: batch_x_i[:,:d_time], self.batch_x_decoder: batch_x_i[:,d_time:], self.time_x_decoder: batch_x_i[:,:d_time], self.kl_weight_x: kl_weight_x_update, self.input_y: np.zeros(shape=(data_x_i.shape[0], data_y.shape[1]), dtype=np.int32), self.batch_y: np.zeros(shape=(data_x_i.shape[0], batch_y.shape[1]-d_time), dtype=np.int32), self.time_y: np.zeros(shape=(data_x_i.shape[0], d_time), dtype=np.int32), self.batch_y_decoder: np.zeros(shape=(data_x_i.shape[0], batch_y.shape[1]-d_time), dtype=np.int32), self.time_y_decoder: np.zeros(shape=(data_x_i.shape[0], d_time)), self.kl_weight_y: kl_weight_y_update});

                ## scATAC-seq reconstruction and translation
                for batch_id in range(nbatch_co): #no oversampling compared to the previous version, since now the model converges very fast!
                    data_y_i = data_y_co[(batch_size*batch_id) : batch_size*(batch_id+1),].todense()
                    batch_y_i = batch_y_co[(batch_size*batch_id) : batch_size*(batch_id+1),]
                    data_x_i = data_x_co[(batch_size*batch_id) : batch_size*(batch_id+1),].todense()
                    batch_x_i = batch_x_co[(batch_size*batch_id) : batch_size*(batch_id+1),]
                    self.sess.run(self.optimizer_y_generator, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i[:,d_time:], self.time_x: batch_x_i[:,:d_time], self.batch_x_decoder: batch_x_i[:,d_time:], self.time_x_decoder: batch_x_i[:,:d_time], self.kl_weight_x: kl_weight_x_update, self.input_y: data_y_i, self.batch_y: batch_y_i[:,d_time:], self.time_y: batch_y_i[:,:d_time], self.batch_y_decoder: batch_y_i[:,d_time:], self.time_y_decoder: batch_y_i[:,:d_time], self.kl_weight_y: kl_weight_y_update});

                ## generator validation loss
                loss_reconstruct_x_val = []
                loss_kl_x_val = []
                loss_y_val = []
                loss_reconstruct_y_val = []
                loss_kl_y_val = []
                loss_translator_xy_val = []
                loss_translator_yx_val = []
                loss_mse_embedding = []
                for batch_id in range(0, nbatch_val_y):
                    data_x_i = data_x_val_co[(batch_size*batch_id) : batch_size*(batch_id+1),].todense()
                    batch_x_i = batch_x_val_co[(batch_size*batch_id) : batch_size*(batch_id+1),]
                    data_y_i = data_y_val_co[(batch_size*batch_id) : batch_size*(batch_id+1),].todense()
                    batch_y_i = batch_y_val_co[(batch_size*batch_id) : batch_size*(batch_id+1),]
                    loss_val_x_i, loss_reconstruct_x_val_i, loss_kl_x_val_i, loss_val_y_i, loss_reconstruct_y_val_i, loss_kl_y_val_i, loss_discriminator_val_i, loss_translator_xy_i, loss_translator_yx_i, loss_mse_embedding_i = self.get_losses_all(data_x_i, batch_x_i, kl_weight_y_update, data_y_i, batch_y_i, kl_weight_y_update, d_time);
                    
                    loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                    loss_kl_x_val.append(loss_kl_x_val_i)
                    loss_y_val.append(loss_val_y_i-loss_discriminator_val_i)
                    loss_reconstruct_y_val.append(loss_reconstruct_y_val_i)
                    loss_kl_y_val.append(loss_kl_y_val_i)
                    loss_translator_xy_val.append(loss_translator_xy_i)
                    loss_translator_yx_val.append(loss_translator_yx_i)
                    loss_mse_embedding.append(loss_mse_embedding_i)

                loss_val_check = np.nanmean(np.array(loss_y_val)) + np.nanmean(np.array(loss_translator_xy_val)) + np.nanmean(np.array(loss_translator_yx_val)) + np.nanmean(np.array(loss_mse_embedding)) #early stopping based on generator loss validation set
                val_reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x_val)))
                val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))
                val_reconstr_y_loss_list.append(np.nanmean(np.array(loss_reconstruct_y_val)))
                val_kl_y_loss_list.append(np.nanmean(np.array(loss_kl_y_val)))
                val_translator_xy_loss_list.append(np.nanmean(np.array(loss_translator_xy_val)))
                val_translator_yx_loss_list.append(np.nanmean(np.array(loss_translator_yx_val)))
                
                if np.isnan(loss_reconstruct_x_val).any():
                    break

                print('loss_val_check: '+str(loss_val_check))
                loss_val_check_list.append(loss_val_check)
                try:
                    loss_val_check_best
                except NameError:
                    loss_val_check_best = loss_val_check
                if loss_val_check <= loss_val_check_best:
                    saver.save(self.sess, output_model+'/mymodel')
                    loss_val_check_best = loss_val_check
                    last_improvement = 0
                else:
                    last_improvement +=1

                ## decide on early stopping 
                stop_decision = last_improvement > patience
                if stop_decision:
                    tf.reset_default_graph()
                    saver = tf.train.import_meta_graph(output_model+ '/mymodel.meta')
                    saver.restore(self.sess, tf.train.latest_checkpoint(output_model +'/'))
                    print('model reached minimum, switching to next')
                    last_improvement = 0
                    loss_val_check_list = []
                    del loss_val_check_best
                    break

        return iter_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_reconstr_y_loss_list, val_kl_y_loss_list, val_translator_xy_loss_list, val_translator_yx_loss_list

    def predict_embedding(self, data_x, batch_x, data_y, batch_y, d_time):
        """
        return scRNA and scATAC projections on VAE embedding layers 
        """
        return self.sess.run([self.px_z_m, self.py_z_m], feed_dict={self.input_x: data_x, self.batch_x: batch_x[:,d_time:], self.time_x: batch_x[:,:d_time], self.batch_x_decoder: batch_x[:,d_time:], self.time_x_decoder: batch_x[:,:d_time], self.kl_weight_x: 1, self.input_y: data_y, self.batch_y: batch_y[:,d_time:], self.time_y: batch_y[:,:d_time], self.batch_y_decoder: batch_y[:,d_time:], self.time_y_decoder: batch_y[:,:d_time], self.kl_weight_y: 1});

    def predict_rnanorm(self, data_x, batch_x, batch_x_decoder, data_y, batch_y, batch_y_decoder, d_time):
        """
        return scRNA rescaled profile (normalized) based on scRNA input
        """
        return self.sess.run(self.px_scale_mean, feed_dict={self.input_x: data_x, self.batch_x: batch_x[:,d_time:], self.time_x: batch_x[:,:d_time], self.batch_x_decoder: batch_x_decoder[:,d_time:], self.time_x_decoder: batch_x_decoder[:,:d_time], self.kl_weight_x: 1, self.input_y: data_y, self.batch_y: batch_y[:,d_time:], self.time_y: batch_y[:,:d_time], self.batch_y_decoder: batch_y_decoder[:,d_time:], self.time_y_decoder: batch_y_decoder[:,:d_time], self.kl_weight_y: 1});
        
    def predict_atacnorm(self, data_x, batch_x, batch_x_decoder, data_y, batch_y, batch_y_decoder, d_time):
        """
        return scATAC rescaled profile (normalized) by varying time factor, predicted based on scATAC input
        """
        return self.sess.run(self.py_mean, feed_dict={self.input_x: data_x, self.batch_x: batch_x[:,d_time:], self.time_x: batch_x[:,:d_time], self.batch_x_decoder: batch_x_decoder[:,d_time:], self.time_x_decoder: batch_x_decoder[:,:d_time], self.kl_weight_x: 1, self.input_y: data_y, self.batch_y: batch_y[:,d_time:], self.time_y: batch_y[:,:d_time], self.batch_y_decoder: batch_y_decoder[:,d_time:], self.time_y_decoder: batch_y_decoder[:,:d_time], self.kl_weight_y: 1});
        
    def predict_atacnorm_trans(self, data_x, batch_x, batch_x_decoder, data_y, batch_y, batch_y_decoder, d_time):
        """
        return scATAC rescaled profile (normalized) by varying time factor, predicted based on scRNA input
        """
        return self.sess.run(self.py_translator, feed_dict={self.input_x: data_x, self.batch_x: batch_x[:,d_time:], self.time_x: batch_x[:,:d_time], self.batch_x_decoder: batch_x_decoder[:,d_time:], self.time_x_decoder: batch_x_decoder[:,:d_time], self.kl_weight_x: 1, self.input_y: data_y, self.batch_y: batch_y[:,d_time:], self.time_y: batch_y[:,:d_time], self.batch_y_decoder: batch_y_decoder[:,d_time:], self.time_y_decoder: batch_y_decoder[:,:d_time], self.kl_weight_y: 1});
        
    def get_losses_all(self, data_x, batch_x, kl_weight_x, data_y, batch_y, kl_weight_y, d_time):
        """
        return various losses
        """
        return self.sess.run([self.loss_x, self.reconstr_loss_x, self.kld_loss_x, self.loss_y, self.reconstr_loss_y, self.kld_loss_y, self.discriminator_loss, self.translator_loss_y, self.translator_loss_x, self.mse_embedding], feed_dict={self.input_x: data_x, self.batch_x: batch_x[:,d_time:], self.time_x: batch_x[:,:d_time], self.batch_x_decoder: batch_x[:,d_time:], self.time_x_decoder: batch_x[:,:d_time], self.kl_weight_x: kl_weight_x, self.input_y: data_y, self.batch_y: batch_y[:,d_time:], self.time_y: batch_y[:,:d_time], self.batch_y_decoder: batch_y[:,d_time:], self.time_y_decoder: batch_y[:,:d_time], self.kl_weight_y: kl_weight_y});

    def restore(self, restore_folder):
        """
        Restore the tensorflow graph stored in restore_folder.
        """
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(restore_folder+'/mymodel.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(restore_folder+'/'))


class AEtimeRNA:
    def __init__(self, input_dim_x, batch_dim_x, embed_dim_x, nlayer, dropout_rate, output_model, learning_rate_x, nlabel=1, discriminator_weight=1):
        """
        Network architecture and optimization

        Inputs
        ----------
        input_x: scRNA expression, ncell x input_dim_x, float
        batch_x: scRNA batch factor, ncell x batch_dim_x, int
        batch_x_decoder: batch factor to be switch to, same format as batch_x
        time_x: scRNA time factor, ncell x d_time, float
        time_x_decoder: time factor to be switch to, same format as time_x
        kl_weight_x: kl weight of the scRNA VAE that is increasing with epoch

        Parameters
        ----------
        input_dim_x: #genes, int
        batch_dim_x: dimension of batch matrix in RNA domain, int
        embed_dim_x: embedding dimension in RNA VAE, int
        learning_rate_x: scRNA VAE learning rate, float
        nlayer: number of hidden layers in encoder/decoder, int, >=1
        dropout_rate: dropout rate in VAE, float
        nlabel: nlabel to be predicted by discriminator
        discriminator_weight: discriminator weight in loss

        """
        self.input_dim_x = input_dim_x;
        self.batch_dim_x = batch_dim_x;
        self.embed_dim_x = embed_dim_x;
        self.learning_rate_x = learning_rate_x;
        self.nlayer = nlayer;
        self.dropout_rate = dropout_rate;
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_dim_x]);
        self.batch_x = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x]);
        self.batch_x_decoder = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x]);
        self.kl_weight_x = tf.placeholder(tf.float32, None);
        self.discriminator_weight = discriminator_weight;
        self.nlabel = nlabel;

        def encoder_rna(input_data, nlayer, hidden_frac=2, reuse=tf.AUTO_REUSE):
            """
            scRNA encoder
            Parameters
            ----------
            hidden_frac: used to divide intermediate dimension to shrink the total paramater size to fit into memory
            input_data: generated from tf.concat([self.input_x, self.batch_x], 1), ncells x (input_dim_x + batch_dim_x)
            """
            with tf.variable_scope('encoder_x', reuse=tf.AUTO_REUSE):
                self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x)/hidden_frac)
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                encoder_output_mean = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_mean')(l1)
                encoder_output_var = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_var')(l1)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min = -2000000, clip_value_max=15)
                encoder_output_var = tf.math.exp(encoder_output_var) + 0.0001
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim_x), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.sqrt(encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;
            

        def decoder_rna(encoded_data, nlayer, hidden_frac=2, reuse=tf.AUTO_REUSE):
            """
            scRNA decoder
            Parameters
            ----------
            encoded_data: generated from concatenation of the encoder output self.encoded_x and batch_x: tf.concat([self.encoded_x, self.batch_x], 1), ncells x (embed_dim_x + batch_dim_x)
            """

            self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x)/hidden_frac);
            with tf.variable_scope('decoder_x', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                px = tf.layers.Dense(self.intermediate_dim, activation=tf.nn.relu, name='decoder_x_px')(l1);
                px_scale = tf.layers.Dense(self.input_dim_x, activation=tf.nn.softmax, name='decoder_x_px_scale')(px);
                px_dropout = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_dropout')(px) 
                px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_r')(px)
                    
                return px_scale, px_dropout, px_r


        def discriminator(input_data, nlayer, nlabel, reuse=tf.AUTO_REUSE):
            """
            discriminator
            Parameters
            ----------
            input_data: the VAE embeddings
            """
            with tf.variable_scope('discriminator_dx', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(int(math.sqrt(self.embed_dim_x * nlabel)), activation=None, name='discriminator_dx_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(int(math.sqrt(self.embed_dim_x * nlabel)), activation=None, name='discriminator_dx_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                output = tf.layers.Dense(nlabel, activation=None, name='discriminator_dx_output')(l1)
                return output;
            

        self.libsize_x = tf.reduce_sum(self.input_x, 1)
        
        self.px_z_m, self.px_z_v, self.encoded_x = encoder_rna(tf.concat([self.input_x, self.batch_x[:,:-self.nlabel]], 1), self.nlayer);

        ## scRNA reconstruction
        self.px_scale, self.px_dropout, self.px_r = decoder_rna(tf.concat([self.encoded_x, self.batch_x_decoder[:,:-self.nlabel]], 1), self.nlayer);

        self.px_r = tf.clip_by_value(self.px_r, clip_value_min = -2000000, clip_value_max=15)
        self.px_r = tf.math.exp(self.px_r)
        self.reconstr_x = tf.transpose(tf.transpose(self.px_scale) *self.libsize_x)
        
        ## scRNA reconstruction
        self.px_scale_mean, self.px_dropout_mean, self.px_r_mean = decoder_rna(tf.concat([self.px_z_m, self.batch_x_decoder[:,:-self.nlabel]], 1), self.nlayer);
        self.reconstr_x_mean = tf.transpose(tf.transpose(self.px_scale_mean) *self.libsize_x)
        
        ## scRNA loss
        # reconstr loss
        self.reconstr_loss_x = calc_zinb_loss(self.px_dropout, self.px_r, self.px_scale, self.input_x, self.reconstr_x)

        # discriminator loss
        self.input_label = self.batch_x[:,-self.nlabel:]
        self.output_label = discriminator(self.px_z_m, 1, self.nlabel)
        if self.nlabel==2:
            cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.discriminator_loss = cce(self.input_label, self.output_label) * self.discriminator_weight
        else:
            self.discriminator_loss = tf.compat.v1.losses.softmax_cross_entropy(self.input_label, self.output_label) * self.discriminator_weight

        # KL loss
        self.kld_loss_x = tf.reduce_mean(0.5*(tf.reduce_sum(-tf.math.log(self.px_z_v) + self.px_z_v + tf.math.square(self.px_z_m) -1, axis=1)))

        ## optimizers
        self.train_vars_x = [var for var in tf.trainable_variables() if '_x' in var.name];
        self.train_vars_dx = [var for var in tf.trainable_variables() if '_dx' in var.name];
        self.loss_x = self.reconstr_loss_x + self.kl_weight_x * self.kld_loss_x
        self.loss_x_generator = self.loss_x - self.discriminator_loss
        self.optimizer_x = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=0.01).minimize(self.loss_x, var_list=self.train_vars_x );
        self.optimizer_dx_discriminator = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=0.01).minimize(self.discriminator_loss, var_list=self.train_vars_dx );
        self.optimizer_dx_generator = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=0.01).minimize(self.loss_x_generator, var_list=self.train_vars_x );

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        self.sess.run(tf.global_variables_initializer());

    def train(self, output_model, data_x='', batch_x='', data_x_val='', batch_x_val='', batch_size='', nlayer='', dropout_rate=0):
        """
        train in two steps, in each step, part of neural network is optimized meanwhile other layers are frozen.
        early stopping based on tolerance (patience) and maximum epochs defined in each step
        iter: document the niter for scATAC autoencoder, once it reaches nepoch_klstart_y, KL will start to warm up
        n_iter_step2: document the niter for scRNA autoencoder, once it reaches nepoch_klstart_x, KL will start to warm up

        """
        saver = tf.train.Saver()
        if os.path.exists(output_model+'/mymodel.meta'):
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
        else:
            val_reconstr_x_loss_list = [];
            val_kl_x_loss_list = [];
            val_discriminator_x_loss_list = [];
            reconstr_x_loss_list = [];
            kl_x_loss_list = [];
            discriminator_x_loss_list = [];
            last_improvement=0
            iter_list = []
            loss_val_check_list = []
            patience=20

            sub_index = random.sample(range(data_x.shape[0]), data_x_val.shape[0])
            data_x_sub = data_x[sub_index,:]
            batch_x_sub = batch_x[sub_index,:]

            if data_x.shape[0] % batch_size >0:
                nbatch_train = data_x.shape[0]//batch_size +1
            else:
                nbatch_train = data_x.shape[0]//batch_size
            if data_x_val.shape[0] % batch_size >0:
                nbatch_val = data_x_val.shape[0]//batch_size +1
            else:
                nbatch_val = data_x_val.shape[0]//batch_size
            for iter in range(1, 5000):
                print('iter '+str(iter))
                iter_list.append(iter)
                sys.stdout.flush()

                sub_index = []
                for batch_value in list(range(self.nlabel)):
                    sub_index_batch = list(np.where(batch_x[:,-batch_value-1] == 1)[0])
                    sub_index.extend(random.sample(sub_index_batch, min(len(sub_index_batch), 1000)))
                random.shuffle(sub_index)
                data_x_dis = data_x[sub_index,:]
                batch_x_dis = batch_x[sub_index,:]
                kl_weight_x_update = min(1.0, iter/float(400))
                
                ## discriminator
                for batch_id in range(0, data_x_dis.shape[0]//batch_size +1):
                    data_x_i = data_x_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_dis.shape[0]),].todense()
                    batch_x_i = batch_x_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_dis.shape[0]),]
                    self.sess.run(self.optimizer_dx_discriminator, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i, self.batch_x_decoder: batch_x_i, self.kl_weight_x: kl_weight_x_update});
                
                loss_reconstruct_x = []
                loss_kl_x = []
                loss_discriminator_x = []
                for batch_id in range(0, data_x_sub.shape[0]//batch_size +1):
                    data_x_i = data_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),].todense()
                    batch_x_i = batch_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),]
                    loss_i, loss_reconstruct_x_i, loss_kl_x_i, loss_discriminator_i = self.get_losses_rna(data_x_i, batch_x_i, batch_x_i, kl_weight_x_update);
                    loss_reconstruct_x.append(loss_reconstruct_x_i)
                    loss_kl_x.append(loss_kl_x_i)
                    loss_discriminator_x.append(loss_discriminator_i)
                
                reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x)))
                kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x)))
                discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x)))

                loss_x_val = []
                loss_reconstruct_x_val = []
                loss_kl_x_val = []
                loss_discriminator_x_val = []
                for batch_id in range(0, nbatch_val):
                    data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),]
                    loss_val_i, loss_reconstruct_x_val_i, loss_kl_x_val_i, loss_discriminator_val_i = self.get_losses_rna(data_x_vali, batch_x_vali, batch_x_vali, kl_weight_x_update);
                    loss_x_val.append(loss_val_i) #early stopping based on VAE loss on validation set
                    loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                    loss_kl_x_val.append(loss_kl_x_val_i)
                    loss_discriminator_x_val.append(loss_discriminator_val_i)

                loss_val_check = np.nanmean(np.array(loss_x_val))
                val_reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x_val)))
                val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))
                val_discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x_val)))

                ## reconstructor
                train_index = random.sample(list(range(data_x.shape[0])), data_x.shape[0])
                data_x = data_x[train_index,:]
                batch_x = batch_x[train_index,:]
                for batch_id in range(0, data_x.shape[0]//batch_size):
                    data_x_i = data_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),].todense()
                    batch_x_i = batch_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),]
                    self.sess.run(self.optimizer_x, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i, self.batch_x_decoder: batch_x_i, self.kl_weight_x: kl_weight_x_update});
                
                ## generators
                for batch_id in range(0, data_x_dis.shape[0]//batch_size +1):
                    data_x_i = data_x_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_dis.shape[0]),].todense()
                    batch_x_i = batch_x_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_dis.shape[0]),]
                    self.sess.run(self.optimizer_dx_generator, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i, self.batch_x_decoder: batch_x_i, self.kl_weight_x: kl_weight_x_update});
                
                print('== generator done')
                loss_reconstruct_x = []
                loss_kl_x = []
                loss_discriminator_x = []
                for batch_id in range(0, data_x_sub.shape[0]//batch_size +1):
                    data_x_i = data_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),].todense()
                    batch_x_i = batch_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),]
                    loss_i, loss_reconstruct_x_i, loss_kl_x_i, loss_discriminator_i = self.get_losses_rna(data_x_i, batch_x_i, batch_x_i, kl_weight_x_update);
                    loss_reconstruct_x.append(loss_reconstruct_x_i)
                    loss_kl_x.append(loss_kl_x_i)
                    loss_discriminator_x.append(loss_discriminator_i)
                
                iter_list.append(iter+0.5)
                reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x)))
                kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x)))
                discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x)))

                loss_x_val = []
                loss_reconstruct_x_val = []
                loss_kl_x_val = []
                loss_discriminator_x_val = []
                for batch_id in range(0, nbatch_val):
                    data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),]
                    loss_val_i, loss_reconstruct_x_val_i, loss_kl_x_val_i, loss_discriminator_val_i = self.get_losses_rna(data_x_vali, batch_x_vali, batch_x_vali, kl_weight_x_update);
                    loss_x_val.append(loss_val_i) #early stopping based on VAE loss on validation set
                    loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                    loss_kl_x_val.append(loss_kl_x_val_i)
                    loss_discriminator_x_val.append(loss_discriminator_val_i)

                loss_val_check = np.nanmean(np.array(loss_x_val))
                val_reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x_val)))
                val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))
                val_discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x_val)))

                print('loss_val_check: '+str(loss_val_check))
                loss_val_check_list.append(loss_val_check)
                try:
                    loss_val_check_best
                except NameError:
                    loss_val_check_best = loss_val_check
                if loss_val_check < loss_val_check_best:
                    saver.save(self.sess, output_model+'/mymodel')
                    loss_val_check_best = loss_val_check
                    last_improvement = 0
                else:
                    last_improvement +=1
                
                ## decide on early stopping 
                stop_decision = last_improvement > patience
                if stop_decision:
                    last_improvement = 0
                    tf.reset_default_graph()
                    saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
                    print("No improvement found during the ( patience) last iterations, stopping optimization.")
                    saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
                    break

            return iter_list, reconstr_x_loss_list, kl_x_loss_list, discriminator_x_loss_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_discriminator_x_loss_list

    def predict_embedding(self, data_x, batch_x):
        """
        return scRNA VAE embeddings
        """
        return self.sess.run(self.px_z_m, feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x});

    def predict_rnanorm(self, data_x, batch_x, batch_x_decoder):
        """
        return scRNA rescaled profile (normalized) with new time or condition specified in batch_x_decoder
        """
        return self.sess.run(self.px_scale_mean, feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x_decoder});
        
    def get_losses_rna(self, data_x, batch_x, batch_x_decoder, kl_weight_x):
        """
        return various losses
        """
        return self.sess.run([self.loss_x, self.reconstr_loss_x, self.kld_loss_x, self.discriminator_loss], feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x_decoder, self.kl_weight_x: kl_weight_x});

    def restore(self, restore_folder):
        """
        Restore the tensorflow graph stored in restore_folder.
        """
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(restore_folder+'/mymodel.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(restore_folder+'/'))


## ============================================================
## other functions for data/model preparation
## ============================================================

## Zero-inflated negative binomial loss
def calc_zinb_loss(px_dropout, px_r, px_scale, input_x, reconstr_x):
    softplus_pi = tf.nn.softplus(-px_dropout)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = tf.log(px_r + 1e-8)
    log_theta_mu_eps = tf.log(px_r + reconstr_x + 1e-8)
    pi_theta_log = -px_dropout + tf.multiply(px_r, (log_theta_eps - log_theta_mu_eps))

    case_zero = tf.nn.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = tf.multiply(tf.dtypes.cast(input_x < 1e-8, tf.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + tf.multiply(input_x, (tf.log(reconstr_x + 1e-8) - log_theta_mu_eps))
        + tf.lgamma(input_x + px_r)
        - tf.lgamma(px_r)
        - tf.lgamma(input_x + 1)
    )
    mul_case_non_zero = tf.multiply(tf.dtypes.cast(input_x > 1e-8, tf.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    translator_loss_x = - tf.reduce_mean(tf.reduce_sum(res, axis=1))
    return(translator_loss_x)


## positional encoding
# ref https://www.tensorflow.org/text/tutorials/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(time, d_model, time_magnitude=1):
    angle_rads = get_angles(time * time_magnitude,
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pd.DataFrame(pos_encoding[0])


def convert_batch_to_onehot(input_dataset_list, dataset_list):
    dic_dataset_index = {}
    for i in range(len(dataset_list)):
        dic_dataset_index[dataset_list[i]] = i

    indices = np.vectorize(dic_dataset_index.get)(np.array(input_dataset_list))
    indptr = range(len(indices)+1)
    data = np.ones(len(indices))
    matrix_dataset = scipy.sparse.csr_matrix((data, indices, indptr), shape=(len(input_dataset_list), len(dataset_list)))
    return(coo_matrix(matrix_dataset))


def randomize_adata(rna_data, randseed):
    data_index = list(range(rna_data.shape[0]))
    random.seed(randseed)
    random.shuffle(data_index)
    data_mat = rna_data[data_index,:].X.tocsr()
    batch_mat = rna_data[data_index,:].obsm['encoding'].to_numpy()
    return data_mat, batch_mat;


def process_adata(rna_h5ad, atac_h5ad, domain, batch, condition, d_time, time_magnitude_float):
    ## identify batch and condition column
    rna_data = ad.read_h5ad(rna_h5ad)
    rna_data.obs['time'] = rna_data.obs['time'].astype(float)

    if domain=='multi':
        atac_data = ad.read_h5ad(atac_h5ad)
        rna_data_tmp = rna_data[random.sample(range(rna_data.shape[0]), 10000),:]
        rna_data_tmp.write(filename='/net/noble/vol2/user/ranz0/2021_ranz0_sc-time/Sunbear/data/example_multi_rna.h5ad', compression=None, compression_opts=None, force_dense=None)
        atac_data_tmp = atac_data[random.sample(range(atac_data.shape[0]), 10000),:]
        atac_data_tmp.write(filename='data/example_multi_atac.h5ad', compression=None, compression_opts=None, force_dense=None)
        atac_data.obs['time'] = atac_data.obs['time'].astype(float)
        chr_list = {}
        for chri in atac_data.var.chr.unique():
            chr_list[chri] = [i for i, x in enumerate(atac_data.var['chr']) if x == chri];
    else:
        rna_data_tmp = rna_data[random.sample(range(rna_data.shape[0]), 10000),:]
        rna_data_tmp.write(filename='data/example_single_rna.h5ad', compression=None, compression_opts=None, force_dense=None)
    if batch != '':
        rna_data.obs['batch'] = rna_data.obs[batch]
        if domain=='multi':
            atac_data.obs['batch'] = atac_data.obs[batch]
            batch_list = list(set(rna_data.obs.batch.unique()) | set(atac_data.obs.batch.unique()))
        else:
            batch_list = list(set(rna_data.obs.batch.unique()))
            
    if condition != '':
        rna_data.obs['condition'] = rna_data.obs[condition]
        if domain=='multi':
            atac_data.obs['condition'] = atac_data.obs[condition]
            condition_list = list(set(rna_data.obs.condition.unique()) | set(atac_data.obs.condition.unique()))
        else:
            condition_list = list(set(rna_data.obs.condition.unique()))
    
    ## build batch, condition and time encoding
    time_encoding = positional_encoding(np.reshape(np.array(rna_data.obs.time), (rna_data.shape[0], 1)), d_time, time_magnitude_float)
    time_encoding.index = rna_data.obs.index
    rna_data.obsm['encoding'] = time_encoding
    if domain=='multi':
        time_encoding = positional_encoding(np.reshape(np.array(atac_data.obs.time), (atac_data.shape[0], 1)), d_time, time_magnitude_float)
        time_encoding.index = atac_data.obs.index
        atac_data.obsm['encoding'] = time_encoding
    else:
        atac_data = ''
    if batch != '':
        batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(rna_data.obs.batch), dataset_list=batch_list).todense())
        batch_encoding.index = rna_data.obs.index
        rna_data.obsm['encoding'] = pd.concat([rna_data.obsm['encoding'], batch_encoding], axis=1)
        if domain=='multi':
            batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(atac_data.obs.batch), dataset_list=batch_list).todense())
            batch_encoding.index = atac_data.obs.index
            atac_data.obsm['encoding'] = pd.concat([atac_data.obsm['encoding'], batch_encoding], axis=1)

    if condition != '':
        batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(rna_data.obs.condition), dataset_list=condition_list).todense())
        batch_encoding.index = rna_data.obs.index
        rna_data.obsm['encoding'] = pd.concat([rna_data.obsm['encoding'], batch_encoding], axis=1)
        if domain=='multi':
            batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(atac_data.obs.condition), dataset_list=condition_list).todense())
            batch_encoding.index = atac_data.obs.index
            atac_data.obsm['encoding'] = pd.concat([atac_data.obsm['encoding'], batch_encoding], axis=1)
    
    ## append time as a label so that we build discriminator to correct time from cell identity factors in the reference domain
    append_encoding = pd.DataFrame(convert_batch_to_onehot(list(rna_data.obs.time), dataset_list=list(rna_data.obs.time.unique())).todense())
    append_encoding.index = rna_data.obs.index
    rna_data.obsm['encoding'] = pd.concat([rna_data.obsm['encoding'], append_encoding], axis=1)
    nlabel = append_encoding.shape[1]
    
    return rna_data, atac_data, nlabel


## ============================================================
## other functions for output evaluation and application
## ============================================================


def normalize_raw(x, logscale=True, norm = 'norm', bulk=False, scale=10000):
    """
    return normalized profile
    """
    x = np.asarray(x)
    if bulk:
        x = np.sum(x, axis=0)
        lib = x.sum()
        x = x/lib
        if logscale:
            x = np.log1p(x*scale)
    else:
        if norm == 'norm':
            ## compare with normalized true profile
            lib = x.sum(axis=1, keepdims=True)
            x = x / lib
        x = np.mean(x, axis=0)
        if logscale:
            x = np.log1p(x*scale)
    return(x.tolist())


def compute_lisi(X, label, perplexity = 30):
    """
    adapted from https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
    Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.
    
    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    n_cells = len(label)
    n_labels = len(label)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(n_neighbors = perplexity * 3, algorithm = 'kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:,1:]
    distances = distances[:,1:]
    # Save the result
    labels = pd.Categorical(label)
    n_categories = len(labels.categories)
    simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
    lisi_df = np.mean(1 / simpson)
    return lisi_df


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float=1e-5
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:,i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:,i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:,i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson
    

def isNaN(num):
    return num!= num


def lmean(x):
    # calculate mean of list, if na, return na
    if isNaN(x):
        return float('nan')
    else:
        return sum(x)/len(x)


def get_neighbor_timepoints(timepoints, timepoint, nearest='close'):
    timepoints = set(timepoints)
    timepoints.add(timepoint)
    timepoints = list(timepoints)
    timepoints.sort()
    timepoint_index = timepoints.index(timepoint)
    if timepoint_index == 0:
        prev_timepoint = float("nan")
    else:
        prev_timepoint = timepoints[timepoint_index-1]
    if timepoint_index == len(timepoints)-1:
        next_timepoint = float("nan")
    else:
        next_timepoint = timepoints[timepoint_index+1]
    if nearest == 'close':
        neighboring_timepoints = [prev_timepoint, next_timepoint]
    else:
        neighboring_timepoints = [timepoints[1],timepoints[-2]]
    return neighboring_timepoints


def compute_pairwise_distances(x, y):
    """
    compute pairwise distance for x and y, used for FOSCTTM distance calculation
    """
    x = np.expand_dims(x, 2)
    y = np.expand_dims(y.T, 0)
    diff = np.sum(np.square(x - y), 1)
    return diff


def NestedDictValues(d, dic, stage):
    for v in dic[d]:
        if v in set(dic.keys()):
            #stagei = dic[v][0].split(':')[0][1:]
            stagei = v.split(':')[0]
            stagei_float = convert_stage_float(stagei)
            stage_float = convert_stage_float(stage)
            if stagei_float<stage_float:
                yield from NestedDictValues(v, dic, stage)
            else:
                yield v


def normalize_scrna(x, logscale=True, norm = 'norm', bulk=False):
    """
    return seq-depth normalized scRNA-seq matrices or pseudobulk
    """
    x = np.asarray(x)
    if bulk:
        x = np.sum(x, axis=0)
        lib = x.sum()
        x = x/lib
        if logscale:
            x = np.log1p(x*10000)
    else:
        if norm == 'norm':
            ## compare with normalized true profile
            lib = x.sum(axis=1, keepdims=True)
            x = x / lib
        if logscale:
            x = np.log1p(x*10000)
    return(x)


def normalize_scrna_gene(x, index):
    """
    return seq-depth normalized query gene expression vector across cells
    """
    lib = x.sum(axis=1)
    x = x[:,index] / lib
    return(x)


def crosscorr_fw(datax, datay, index_list, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    Reference
    ----------
    # https://www.kaggle.com/code/adepvenugopal/time-series-correlation-pearson-tlcc-dtw
    """
    return datax[index_list].corr(datay.shift(lag)[index_list])


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))


def pred_expression(autoencoder, data_x_seed, timepoint_seed, time_step, time_range, d_time, time_magnitude, dataset):
    allArrays = np.empty((0, data_x_seed.shape[1]))
    timepoint_pred_list = []
    for i in range(0, 10000):
        timepoint_pred = i*time_step + timepoint_seed
        if timepoint_pred <= timepoint_seed + time_range:
            timepoint_pred_list.append(timepoint_pred)
            time_encoding = positional_encoding(timepoint_pred, d_time, time_magnitude)
            data_rna_seed_pred_i = autoencoder.predict_rnanorm(data_x_seed.X.tocsr().todense(), data_x_seed.obsm['encoding'].to_numpy(), np.hstack((np.tile(time_encoding, (data_x_seed.shape[0],1)), data_x_seed.obsm['encoding'].to_numpy()[:,d_time:])))
            allArrays = np.append(allArrays, np.log1p(data_rna_seed_pred_i), axis = 0)
        else:
            break
    return timepoint_pred_list, allArrays


def calc_temporal_exp(sim_url, autoencoder, domain, rna_data, atac_data, ct_query, timepoint_seed, time_step, time_range, d_time, time_magnitude_float):
    '''
    derive temporal gene expression trend for a query cell
    '''
    data_x_seed = rna_data[(rna_data.obs.time==timepoint_seed) & (rna_data.obs.celltype==ct_query),]
    if data_x_seed.shape[0]>50:
        data_x_seed = data_x_seed[random.sample(range(data_x_seed.shape[0]), 50),]
    if domain == 'multi':
        data_y_seed = atac_data[range(data_x_seed.shape[0]),]
    dic_gene = {}
    dic_peak = {}
    for i in range(0, 10000):
        timepoint_pred = i*time_step + timepoint_seed - time_range
        if timepoint_pred <= timepoint_seed + time_range:
            time_encoding = positional_encoding(timepoint_pred, d_time, time_magnitude_float)
            if domain == 'rna':
                data_rna_seed_pred_i = autoencoder.predict_rnanorm(data_x_seed.X.tocsr().todense(), data_x_seed.obsm['encoding'].to_numpy(), np.hstack((np.tile(time_encoding, (data_x_seed.shape[0],1)), data_x_seed.obsm['encoding'].to_numpy()[:,d_time:])))
                ## normalize
                dic_gene[timepoint_pred] = normalize(data_rna_seed_pred_i, axis=1, norm='l1')
            elif domain=='multi':
                data_rna_seed_pred_i = autoencoder.predict_rnanorm(data_x_seed.X.tocsr().todense(), 
                                                                data_x_seed.obsm['encoding'].to_numpy(), 
                                                                np.hstack((np.tile(time_encoding, (data_x_seed.shape[0],1)), 
                                                                            data_x_seed.obsm['encoding'].to_numpy()[:,d_time:])),
                                                                data_y_seed.X.tocsr().todense(), 
                                                                data_y_seed.obsm['encoding'].to_numpy(), 
                                                                np.hstack((np.tile(time_encoding, (data_y_seed.shape[0],1)), 
                                                                            data_y_seed.obsm['encoding'].to_numpy()[:,d_time:])),
                                                                d_time)
                dic_gene[timepoint_pred] = normalize(data_rna_seed_pred_i, axis=1, norm='l1')
                data_atac_seed_pred_i = autoencoder.predict_atacnorm_trans(data_x_seed.X.tocsr().todense(), 
                                                            data_x_seed.obsm['encoding'].to_numpy(), 
                                                            np.hstack((np.tile(time_encoding, (data_x_seed.shape[0],1)), 
                                                                        data_x_seed.obsm['encoding'].to_numpy()[:,d_time:])),
                                                            data_y_seed.X.tocsr().todense(), 
                                                            data_y_seed.obsm['encoding'].to_numpy(), 
                                                            np.hstack((np.tile(time_encoding, (data_y_seed.shape[0],1)), 
                                                                        data_y_seed.obsm['encoding'].to_numpy()[:,d_time:])),
                                                                        d_time)
                dic_peak[timepoint_pred] = normalize(data_atac_seed_pred_i, axis=1, norm='l1')

    np.save(sim_url + ct_query + 'time' + str(timepoint_seed) + '_'+ str(time_step) + '_' + str(time_range) + '_exp.npy', dic_gene)
    if domain=='multi':
        np.save(sim_url + ct_query + 'time' + str(timepoint_seed) + '_'+ str(time_step) + '_' + str(time_range) + '_acc.npy', dic_peak)


def calc_embedding(autoencoder, rna_data, atac_data, batch_size, d_time, method, domain, sim_url, celltype_i='', nk=25):
    """
    predict cell embeddings
    return LISI score or cell embeddings for UMAP
    """
    id_label = []
    batch_label = []
    domain_label = []
    timepoint_label = []
    celltype_label = []
    ## from time points that have both single and co-assay, then we just retrieve the embedding together
    ## RNA
    if rna_data.shape[0] % batch_size >0:
        nbatch = rna_data.shape[0]//batch_size +1
    else:
        nbatch = rna_data.shape[0]//batch_size
        
    timepoint_label.extend(list(rna_data.obs['time']))
    id_label.extend(list(rna_data.obs.index))
    batch_label.extend(list(rna_data.obs['batch']))
    if 'celltype' in rna_data.obs.columns:
        celltype_label.extend(list(rna_data.obs['celltype']))
    else:
        celltype_label.extend(['']*rna_data.shape[0])
    domain_label.extend(['RNA']*rna_data.shape[0])
    for batch_id in range(0, nbatch):
        batch_index = range((batch_size*batch_id), min(batch_size*(batch_id+1), rna_data.shape[0]))
        if domain=='multi':
            rna_embedding, atac_embedding = autoencoder.predict_embedding(rna_data.X[batch_index, ].todense(), rna_data[batch_index, ].obsm['encoding'].to_numpy(), atac_data.X[:len(batch_index),].todense(), atac_data[:len(batch_index),].obsm['encoding'].to_numpy(), d_time)
        else:
            rna_embedding = autoencoder.predict_embedding(rna_data.X[batch_index, ].todense(), rna_data[batch_index, ].obsm['encoding'].to_numpy())
        if batch_id==0:
            sc_rna_combined_embedding = rna_embedding
        else:
            sc_rna_combined_embedding = np.concatenate((sc_rna_combined_embedding, rna_embedding), axis = 0)
    
    ## ATAC
    if domain == 'multi' and atac_data.shape[0]>0:
        print('calc embedding of multi')
        if atac_data.shape[0] % batch_size >0:
            nbatch = atac_data.shape[0]//batch_size +1
        else:
            nbatch = atac_data.shape[0]//batch_size
            
        timepoint_label.extend(list(atac_data.obs['time']))
        id_label.extend(list(atac_data.obs.index))
        batch_label.extend(list(atac_data.obs['batch']))
        if 'celltype' in atac_data.obs.columns:
            celltype_label.extend(list(atac_data.obs['celltype']))
        else:
            celltype_label.extend(['']*atac_data.shape[0])
        domain_label.extend(['ATAC']*atac_data.shape[0])
        for batch_id in range(0, nbatch):
            batch_index = range((batch_size*batch_id), min(batch_size*(batch_id+1), atac_data.shape[0]))
            rna_embedding, atac_embedding = autoencoder.predict_embedding(rna_data.X[:len(batch_index), ].todense(), rna_data[:len(batch_index), ].obsm['encoding'].to_numpy(), atac_data.X[batch_index, ].todense(), atac_data[batch_index, ].obsm['encoding'].to_numpy(), d_time)
            if batch_id==0:
                sc_atac_combined_embedding = atac_embedding
            else:
                sc_atac_combined_embedding = np.concatenate((sc_atac_combined_embedding, atac_embedding), axis = 0)
        
        sc_combined_embedding = np.concatenate((sc_rna_combined_embedding, sc_atac_combined_embedding), axis = 0)
    else:
        sc_combined_embedding = sc_rna_combined_embedding

    ## calculate LISI score for time, batch and data domain
    if method=='lisi':
        lisi_score_vec = []
        lisi_score_vec.append(compute_lisi(sc_combined_embedding, timepoint_label, perplexity = 30))
        lisi_score_vec.append(compute_lisi(sc_combined_embedding, batch_label, perplexity = 30))
        lisi_score_vec.append(compute_lisi(sc_combined_embedding, domain_label, perplexity = 30))

        return(lisi_score_vec)
    

def plot_auroc_pergene(matrix_true, matrix_pred):
    matrix_true = binarize(matrix_true)
    ## overall
    fpr, tpr, _thresholds = metrics.roc_curve(matrix_true.flatten(), matrix_pred.flatten())
    auc_flatten = metrics.auc(fpr, tpr)
    print(auc_flatten)

    auc_list = []
    npos = []
    for i in range(matrix_true.shape[1]):
        pp = np.sum(matrix_true[:,i])
        npos.append(pp)
        if pp >= 1:
            fpr, tpr, _thresholds = metrics.roc_curve(matrix_true[:,i], matrix_pred[:,i])
            auc = metrics.auc(fpr, tpr)
            auc_list.append(auc)
        else:
            auc_list.append(np.nan)
            
    return np.array(auc_list), auc_flatten, np.array(npos)



def calc_wilcoxon(x, y, alternative='two-sided'):
    if alternative=='norm':
        ttest = wilcoxon(x, y)
        pval = ttest.pvalue
        ttest = wilcoxon(x, y, alternative='greater')
        stat_norm = ttest.statistic *2 / (len(x)*(len(x)+1))
        output = np.array([pval, stat_norm])
    elif alternative=='normz':
        # z-statistic: https://github.com/scipy/scipy/blob/v1.11.2/scipy/stats/_morestats.py#L3831-L4184
        ttest = wilcoxon(x, y)
        pval = ttest.pvalue
        #ttest = wilcoxon(x, y, alternative='greater')
        d = x-y
        d = compress(np.not_equal(d, 0), d)

        count = len(d)
        r = rankdata(abs(d))
        r_plus = np.sum((d > 0) * r)
        T = r_plus

        mn = count * (count + 1.) * 0.25
        se = count * (count + 1.) * (2. * count + 1.)
        replist, repnum = find_repeats(r)
        if repnum.size != 0:
            # Correction for repeated elements.
            se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

        se = sqrt(se / 24)
        # apply continuity correction if applicable
        d = 0.5
        stat_norm = (T - mn - d) / se
        output = np.array([pval, stat_norm])
    else:
        ttest = wilcoxon(x, y, alternative=alternative)
        output = np.array([ttest.pvalue, ttest.statistic])
    return(output)


def calc_metric(x1, x2, metric='mse', logscale=True):
    """
    return comparison statistics between x1 and x2
    """
    if logscale:
        x1 = np.log1p(x1)
        x2 = np.log1p(x2)
    if metric=='mse':
        output = mean_squared_error(x1, x2)
    if metric == 'pseudocor':
        output, tmp = scipy.stats.pearsonr(x1, x2)
    return(output)


def eval_temporal_rna(sim_url, autoencoder, rna_data, timepoint):
    """
    evaluate cross-time and cross-condition prediction on the held-out timepoint
    return pseudobulk pearson correlation per cell type
    """
    celltype_list = rna_data.obs.major_trajectory.value_counts().index.tolist()
    for neighbor_ver in ['_prev', '_next']:
        fout = open(sim_url+ '_pseudobulk_eval' + neighbor_ver + '.txt', 'w')
        target_sex = list(rna_data[rna_data.obs.time==timepoint,:].obs.sex.unique())[0]
        if target_sex =='M':
            from_sex = 'F'
        if target_sex == 'F':
            from_sex = 'M'
        neighboring_timepoints_target = get_neighbor_timepoints(list(rna_data[rna_data.obs.sex==target_sex,:].obs.time.unique()), timepoint)
        neighboring_timepoints_from = get_neighbor_timepoints(list(rna_data[rna_data.obs.sex==from_sex,:].obs.time.unique()), timepoint)

        if neighbor_ver=='_prev':
            nearest_time_index = 0
        elif neighbor_ver=='_next':
            nearest_time_index = 1
        neighboring_timepoints_from = neighboring_timepoints_from[nearest_time_index]
        neighboring_timepoints_target = neighboring_timepoints_target[nearest_time_index]

        if not math.isnan(neighboring_timepoints_from) and not math.isnan(neighboring_timepoints_target):
            for ct in celltype_list:
                rna_data_i = rna_data[(rna_data.obs.major_trajectory==ct) & (rna_data.obs.time==timepoint),]
                rna_data_i_neighbor = rna_data[(rna_data.obs.time == neighboring_timepoints_from) & (rna_data.obs.major_trajectory==ct) & (rna_data.obs.sex==from_sex),]
                
                if rna_data_i.shape[0]>=10 and rna_data_i_neighbor.shape[0]>=10:
                    rna_data_i_pseudo = normalize_raw(rna_data_i.X.todense())
                    
                    ## predict using sum of neighboring cell types (baseline prediction)
                    rna_data_i_target_neighbor = rna_data[(rna_data.obs.time == neighboring_timepoints_target) & (rna_data.obs.major_trajectory==ct) & (rna_data.obs.sex==target_sex),]
                    rna_data_i_target_neighbor_pseudo = normalize_raw(rna_data_i_target_neighbor.X.todense())
                    
                    ## predict missing time point by swapping sex factor (Sunbear prediction)
                    time_encoding = positional_encoding(timepoint, d_time, time_magnitude_float)
                    batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(target_sex), dataset_list=list(rna_data.obs['batch'].unique())).todense())
                    append_encoding = pd.DataFrame(convert_batch_to_onehot([timepoint], dataset_list=list(rna_data.obs.time.unique())).todense())
                    swap_encoding = pd.concat([time_encoding, batch_encoding, append_encoding], axis=1)

                    rna_data_i_swapsex_pred = autoencoder.predict_rnanorm(rna_data_i_neighbor.X.todense(), rna_data_i_neighbor.obsm['encoding'].to_numpy(), np.tile(swap_encoding.to_numpy(), (rna_data_i_neighbor.shape[0],1)))
                    rna_data_i_swapsex_pred_pseudo = normalize_raw(rna_data_i_swapsex_pred)
                
                    rna_data_i_target_neighbor_pred = autoencoder.predict_rnanorm(rna_data_i_target_neighbor.X.todense(), rna_data_i_target_neighbor.obsm['encoding'].to_numpy(), np.tile(swap_encoding.to_numpy(), (rna_data_i_target_neighbor.shape[0],1)))
                    rna_data_i_target_neighbor_pred_pseudo = normalize_raw(rna_data_i_target_neighbor_pred)
                
                    cor_baseline = np.corrcoef(rna_data_i_pseudo, rna_data_i_target_neighbor_pseudo)[0,1]
                    cor_swapsex_pred = np.corrcoef(rna_data_i_pseudo, rna_data_i_swapsex_pred_pseudo)[0,1]
                    cor_target_pred = np.corrcoef(rna_data_i_pseudo, rna_data_i_target_neighbor_pred_pseudo)[0,1]
                    fout.write(str(timepoint)+'\t'+ ct+ '\t'+ str(rna_data_i.shape[0])+ '\tcor\t'+ str(cor_baseline)+ '\t'+ str(cor_swapsex_pred)+ '\t'+ str(cor_target_pred)+ '\n')

                    if False:
                        ## calculate pseudobulk MSE
                        mse_baseline = mean_squared_error(rna_data_i_pseudo,rna_data_i_target_neighbor_pseudo)
                        mse_swap = mean_squared_error(rna_data_i_pseudo,rna_data_i_swapsex_pred_pseudo)
                        mse_cont = mean_squared_error(rna_data_i_pseudo,rna_data_i_target_neighbor_pred_pseudo)
                        fout.write(str(timepoint)+'\t'+ ct+ '\t'+ str(rna_data_i.shape[0])+ '\tmse\t'+ str(mse_baseline)+ '\t'+ str(mse_swap)+ '\t'+ str(mse_cont)+ '\n')

        fout.close()


def eval_model_rna(sim_url, autoencoder, rna_data_val, timepoint):
    """
    evaluate RNA model on validation set, for downstream use of selecting hyperparameters
    """
    sim_metric_val = []
    sorted_timepoints = list(rna_data_val.obs.time.unique())
    sorted_timepoints.sort()
    neighboring_timepoints = get_neighbor_timepoints(sorted_timepoints, timepoint)
    neighboring_timepoints = [x for x in neighboring_timepoints if not math.isnan(x)]
    if len(neighboring_timepoints) < 2:
        neighboring_timepoints_additional = get_neighbor_timepoints(sorted_timepoints, neighboring_timepoints[0])
        neighboring_timepoints = neighboring_timepoints + neighboring_timepoints_additional.remove(timepoint)
        
    ## all neighbor
    pearson_r_meanbulk_val = []
    for i in range(len(sorted_timepoints)-1):
        rna_data_val_time_i = rna_data_val[rna_data_val.obs.time==sorted_timepoints[i],]
        rna_data_val_time_j = rna_data_val[rna_data_val.obs.time==sorted_timepoints[i+1],]
        data_rna_data_val_time_ij_pred_norm = autoencoder.predict_rnanorm(rna_data_val_time_i.X.todense(), rna_data_val_time_i.obsm['encoding'].to_numpy(), np.tile(rna_data_val_time_j.obsm['encoding'].to_numpy()[1,], (rna_data_val_time_i.shape[0],1)))
        pearson_r_meanbulk, pearson_p_meanbulk = scipy.stats.pearsonr(normalize_raw(data_rna_data_val_time_ij_pred_norm), normalize_raw(rna_data_val_time_j.X.todense()))
        pearson_r_meanbulk_val.append(pearson_r_meanbulk)
        
    sim_metric_val.append(sum(pearson_r_meanbulk_val)/len(pearson_r_meanbulk_val))

    ## closest neighbor
    pearson_r_meanbulk_val = []
    for i in range(len(neighboring_timepoints)-1):
        rna_data_val_time_i = rna_data_val[rna_data_val.obs.time==neighboring_timepoints[i],]
        rna_data_val_time_j = rna_data_val[rna_data_val.obs.time==neighboring_timepoints[i+1],]
        data_rna_data_val_time_ij_pred_norm = autoencoder.predict_rnanorm(rna_data_val_time_i.X.todense(), rna_data_val_time_i.obsm['encoding'].to_numpy(), np.tile(rna_data_val_time_j.obsm['encoding'].to_numpy()[1,], (rna_data_val_time_i.shape[0],1)))
        pearson_r_meanbulk, pearson_p_meanbulk = scipy.stats.pearsonr(normalize_raw(data_rna_data_val_time_ij_pred_norm), normalize_raw(rna_data_val_time_j.X.todense()))
        pearson_r_meanbulk_val.append(pearson_r_meanbulk)
        
    sim_metric_val.append(sum(pearson_r_meanbulk_val)/len(pearson_r_meanbulk_val))

    ## get LISI term based on embeddings across time in the validation set, let data itself define sigma
    if rna_data_val.shape[0] % batch_size >0:
        nbatch_val = rna_data_val.shape[0]//batch_size +1
    else:
        nbatch_val = rna_data_val.shape[0]//batch_size
    for batch_id in range(0, nbatch_val):
        rna_data_val_i = rna_data_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), rna_data_val.shape[0]),]
        if rna_data_val_i.shape[0]>1:
            sc_val_embedding_i = autoencoder.predict_embedding(rna_data_val_i.X.todense(), rna_data_val_i.obsm['encoding'].to_numpy());
        else:
            sc_val_embedding_i = autoencoder.predict_embedding(np.reshape(rna_data_val_i.X, (1, -1)), rna_data_val_i.obsm['encoding'].to_numpy());
        if batch_id == 0:
            sc_val_embedding = sc_val_embedding_i
        else:
            sc_val_embedding = np.concatenate((sc_val_embedding, sc_val_embedding_i))

    ## get LISI term based on embeddings across neighboring validation time points
    ## all neighbor
    label = rna_data_val.obs.time
    lisi_score_vec = []
    for i in range(len(sorted_timepoints)-1):
        sc_val_embedding_i = sc_val_embedding[rna_data_val.obs.time.isin(sorted_timepoints[i:(i+2)]),:]
        label_i = label[rna_data_val.obs.time.isin(sorted_timepoints[i:(i+2)])]
        lisi_score = compute_lisi(sc_val_embedding_i, label_i.tolist(), perplexity = 5)
        lisi_score_vec.append(lisi_score)

    sim_metric_val.append(sum(lisi_score_vec)/len(lisi_score_vec))

    ## closest neighbor
    label = rna_data_val.obs.time
    lisi_score_vec = []
    for i in range(len(neighboring_timepoints)-1):
        sc_val_embedding_i = sc_val_embedding[rna_data_val.obs.time.isin(neighboring_timepoints[i:(i+2)]),:]
        label_i = label[rna_data_val.obs.time.isin(neighboring_timepoints[i:(i+2)])]
        lisi_score = compute_lisi(sc_val_embedding_i, label_i.tolist(), perplexity = 5)
        lisi_score_vec.append(lisi_score)

    sim_metric_val.append(sum(lisi_score_vec)/len(lisi_score_vec))

    print('== val data evaluated ==')
    sys.stdout.flush()
    np.savetxt(sim_url+'_validation.txt', sim_metric_val, delimiter='\n', fmt='%1.10f')


def calc_condition_diffexp(sim_url, autoencoder, rna_data_i, swap_encoding_source, swap_encoding_target):
    """
    predict differences between conditions at any time point query
    """

    ## TODO - ref calc_new_diffexp.py and sexdiff_eval.R
    rna_data_i_reconstr = autoencoder.predict_rnanorm(rna_data_i.X.todense(), rna_data_i.obsm['encoding'].to_numpy(), swap_encoding_source)
    rna_data_i_swap = autoencoder.predict_rnanorm(rna_data_i.X.todense(), rna_data_i.obsm['encoding'].to_numpy(), swap_encoding_target)

    ## normalize with total depth
    rna_data_i_reconstr_norm = (rna_data_i_reconstr.T / np.mean(rna_data_i_reconstr, axis=1)).T
    rna_data_i_swap_norm = (rna_data_i_swap.T / np.mean(rna_data_i_swap, axis=1)).T
    
    for celltype_i in rna_data_i.obs.celltype.unique():
        index = rna_data_i.obs.celltype==celltype_i
        if np.sum(index)>=50:
            ttest_output = np.empty((0, 2))
            for i in range(rna_data_i.shape[1]):
                ttest_output = np.vstack([ttest_output, calc_wilcoxon(
                    rna_data_i_swap_norm[index, i], rna_data_i_reconstr_norm[index, i], alternative='norm')])
                
            d = {'gene': rna_data_i.var['gene_short_name'].tolist(), 
                    #'pval': ttest_output[:,0].tolist(), 
                    'statistic': ttest_output[:,1].tolist()}
            output_df = pd.DataFrame(data=d)
            output_df.to_csv(sim_url+ 'diff'+ '_'+ celltype_i+ '.txt', index=False, sep='\t', header=True)

