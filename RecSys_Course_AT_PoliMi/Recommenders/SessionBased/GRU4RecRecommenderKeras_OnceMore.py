from multiprocessing.sharedctypes import Value
import DressipiChallenge.Recommenders.Neural.GRU4Rec.model.gru4rec_embeddings as ge

from DressipiChallenge.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from DressipiChallenge.Recommenders.BaseTempFolder import BaseTempFolder
from DressipiChallenge.Recommenders.DataIO import DataIO
from DressipiChallenge.Recommenders.SessionBased.GRU4RecRecommenderBase import GRU4RecRecommenderNextBase

from tqdm import tqdm
import os
import numpy as np
import time
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, GRU
from tensorflow.python.ops import rnn_cell
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.utils import to_categorical

class GRU4RecRecommenderKerasAlternative(GRU4RecRecommenderNextBase, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "GRU4RecRecommenderKerasAlternative"

    def __init__(self, *args, **kwargs):

        super(GRU4RecRecommenderKerasAlternative, self).__init__(*args, **kwargs)

        gpu_config = tf.compat.v1.ConfigProto()
        tf.compat.v1.disable_eager_execution()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=gpu_config)
        self.is_training = True

        self.last_view_df = self.test_sessions_df.drop_duplicates(['session_id'], keep='last')
        self.model = None

        self.max_items = 23691
        self.session_number = len(self.train_df.session_id.unique())

    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self.model.save(self.best_model_file)

        if self.save_weights:
            self.model.save_weights(self.weights_file)

    ########################ACTIVATION FUNCTIONS#########################
    def linear(self, X):
        return X
    def tanh(self, X):
        return tf.nn.tanh(X)
    def softmax(self, X):
        return tf.nn.softmax(X)
    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))
    def relu(self, X):
        return tf.nn.relu(X)
    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    ############################LOSS FUNCTIONS######################
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat)+1e-24))
    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat)-yhatT)))
    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def create_model(self,
                    # Network Arguments
                    epochs = 3,
                    rnn_size = 100,
                    batch_size = 10,
                    layers = 1,
                    dropout_p_hidden=1,
                    decay = 0.96, 
                    decay_steps = 1e4,
                    sigma = 0,
                    init_as_normal = False,
                    final_act = 'softmax',
                    hidden_act = 'tanh',
                    loss = "bpr",
                    grad_cap = 0,
                    # Opt arguments
                    learning_rate=0.001,
                    # Recommender and Evaluatore args
                    ):

        self.X = tf.compat.v1.placeholder(tf.int32, [batch_size], name='input')
        self.Y = tf.compat.v1.placeholder(tf.int32, [batch_size], name='output')
        self.state = [tf.compat.v1.placeholder(tf.float32, [batch_size, rnn_size], name='rnn_state') for _ in range(layers)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.compat.v1.variable_scope('gru_layer'):
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.max_items + rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.compat.v1.get_variable('embedding', [self.max_items, rnn_size], initializer=initializer)
            softmax_W = tf.compat.v1.get_variable('softmax_w', [self.max_items, rnn_size], initializer=initializer)
            softmax_b = tf.compat.v1.get_variable('softmax_b', [self.max_items], initializer=tf.constant_initializer(0.0))

            cell = rnn_cell.GRUCell(rnn_size, activation=hidden_act)
            drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_p_hidden)
            stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * layers)
            
            inputs = tf.nn.embedding_lookup(embedding, self.X)
            output, state = stacked_cell(inputs, tuple(self.state))
            self.final_state = state

        if self.is_training:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)
        else:
            logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            self.yhat = self.final_activation(logits)

        if not self.is_training:
            return

        self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True)) 
        
        '''
        Try different optimizers.
        '''
        #optimizer = tf.train.AdagradOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        #optimizer = tf.train.AdadeltaOptimizer(self.lr)
        #optimizer = tf.train.RMSPropOptimizer(self.lr)

        tvars = tf.compat.v1.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs 
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def fit(self,
            # Network Arguments
            epochs = 3,
            rnn_size = 100,
            batch_size = 10,
            layers = 1,
            dropout_p_hidden=1,
            decay = 0.96, 
            decay_steps = 1e4,
            sigma = 0,
            init_as_normal = False,
            final_act = 'softmax',
            hidden_act = 'tanh',
            loss = "bpr",
            grad_cap = 0,
            # Opt arguments
            learning_rate=0.001,
            # Recommender and Evaluatore args
            save_weights=False,
            temp_file_folder="./GRU4RecKeras/",
            **earlystopping_kwargs
            ):

        self.layers = layers
        self.rnn_size = rnn_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.decay = decay
        self.decay_steps = decay_steps
        self.sigma = sigma
        self.init_as_normal = init_as_normal
        self.grad_cap = grad_cap
        self.batch_size = batch_size
        
        if hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if loss == 'cross-entropy':
            if final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif loss == 'bpr':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif loss == 'top1':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        network_args = {
            "epochs": epochs,
            "rnn_size": rnn_size,
            "layers" : layers,
            "dropout_p_hidden": dropout_p_hidden,
            "decay" : decay, 
            "decay_steps" : decay_steps,
            "sigma" : sigma,
            "init_as_normal": init_as_normal,
            "final_act" : final_act,
            "hidden_act" : hidden_act,
            "loss" : loss,
            "grad_cap" : grad_cap,
            # Opt arguments
            "learning_rate": learning_rate
        }

        self.create_model(batch_size=batch_size, **network_args)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)

        self.checkpoint_dir = temp_file_folder
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Checkpoint Dir not found")

        if self.is_training:
            return

        # use self.predict_state to hold hidden states during prediction. 
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, '{}/gru-model'.format(self.checkpoint_dir))

        self.error_during_train = False
        self.offset_sessions = self.init(self.train_df)
        print('{0}: fitting model...'.format(self.RECOMMENDER_NAME))

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

    def init(self, data):
        data.sort([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return offset_sessions

    def clear_session(self):
        del self.model

    def _run_epoch(self, num_epoch):

        with tqdm(total=self.session_number) as pbar:
            
            self.done_sessions_counter = 0
            epoch_cost = []
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
            session_idx_arr = np.arange(len(self.offset_sessions)-1)
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = self.offset_sessions[session_idx_arr[iters]]
            end = self.offset_sessions[session_idx_arr[iters]+1]
            mask = [] # indicator for the sessions to be terminated
            finished = False
            while not finished:
                minlen = (end-start).min()
                out_idx = self.train_df[self.item_key].values[start]
                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx = self.train_df[self.item_key].values[start+i+1]
                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: out_idx}
                    for j in range(self.layers): 
                        feed_dict[self.state[j]] = state[j]
                    
                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(num_epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)

                        pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(num_epoch, epoch_cost))
                        pbar.update(self.done_sessions_counter)
                start = start+minlen-1
                mask = np.arange(len(iters))[(end-start)<=1]
                self.done_sessions_counter = len(mask)
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(self.offset_sessions)-1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = self.offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = self.offset_sessions[session_idx_arr[maxiter]+1]
                if len(mask) and self.reset_after_session:
                    for i in range(self.layers):
                        state[i][mask] = 0
            
            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(num_epoch, avgc))
                self.error_during_train = True
                return
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=num_epoch)

    def predict_next(self, sessions_id_array, input_items_prediction, predict_for_item_ids, batch):

        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1 
            self.predict = True
        
        session_change = np.arange(batch)[sessions_id_array != self.current_session]
        if len(session_change) > 0: # change internal states with session changes
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session=sessions_id_array.copy()

        in_idxs = input_items_prediction
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: in_idxs}
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds)[:, predict_for_item_ids]
        return pd.DataFrame(data=preds, index=predict_for_item_ids)

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        self.model.save(folder_path + file_name + "_{0}_model.h5".format(self.RECOMMENDER_NAME))

        data_dict_to_save = {
            "n_sessions": self.session_number,
            "max_items": self.max_items,
            "emb_size_features": self.emb_size_features,
            "emb_size_items": self.emb_size_items,
            "batch_size": self.batch_size,
            "dropout_value": self.dropout_value,
            "learning_rate": self.learning_rate,
            "hidden_units": self.hidden_units,
            "epochs": self.epochs,
            "activation": self.activation,
            'activation_parameter': self.activation_parameter,
            'hidden_activation': self.hidden_activation
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):
        self.model = tf.keras.model.load_model(folder_path, file_name=file_name)