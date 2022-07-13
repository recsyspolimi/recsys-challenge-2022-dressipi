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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.utils import to_categorical

def BPR(y_true, y_pred):

    to_lookup = tf.argmax(y_true, axis = 1)   # = indices of the target items
    scores = tf.nn.embedding_lookup(tf.transpose(y_pred), to_lookup)  # embedding_lookup is the same as "extract_rows". In this way, the positive items end up on the diagonal
    return tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(tf.linalg.diag_part(scores) - scores) + 1e-12))

def softmax_neg(X: tf.Tensor):
    hm = 1.0 - tf.eye(X.shape[0], X.shape[1])
    X = X * hm
    e_x = tf.exp(X - tf.math.reduce_max(X, axis=1, keepdims=True)) * hm
    return e_x / tf.math.reduce_sum(e_x, axis=1, keepdims=True)

def BPR_MAX(y_true, y_pred):

    to_lookup = tf.argmax(y_true, axis = 1)   # = indices of the target items
    scores = tf.nn.embedding_lookup(tf.transpose(y_pred), to_lookup)  # embedding_lookup is the same as "extract_rows". In this way, the positive items end up on the diagonal
    softmax_score = softmax_neg(scores)

    return tf.reduce_mean(-tf.math.log(tf.reduce_sum(tf.nn.sigmoid(tf.linalg.diag_part(scores) - scores)*softmax_score, axis=1) + 1e-8) + 0.49*tf.reduce_sum((scores**2)*softmax_score, axis=1))

class GRU4RecRecommenderKeras(GRU4RecRecommenderNextBase, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "GRU4RecRecommenderKeras"

    def __init__(self, item_features=None, item_features_key=None, itemmap=None, *args, **kwargs):

        super(GRU4RecRecommenderKeras, self).__init__(*args, **kwargs)

        self.last_view_df = self.test_sessions_df.drop_duplicates(['session_id'], keep='last')
        self.item_features = item_features.copy() if item_features is not None else None
        self.item_features_key = item_features_key
        self.model = None
        self.validation_model = None
        self.itemmap = itemmap

        #self.loss = BPR

        self.max_items = 23691
        self.session_number = len(self.train_df.session_id.unique())

        if self.item_features is not None:
            print("{0}: version with embeddings.".format(self.RECOMMENDER_NAME))
            self.dataset = ge.SessionDatasetEmbeddings(self.train_df, 
                                                    self.item_features.copy(), 
                                                    session_key=self.session_key, 
                                                    item_key=self.item_key,
                                                    time_key=self.time_key,
                                                    item_features_key=self.item_features_key,
                                                    time_sort=True)
        else:
            print("{0}: version without embeddings.".format(self.RECOMMENDER_NAME))
            self.dataset = ge.SessionDataset(self.train_df, 
                                                    session_key=self.session_key, 
                                                    item_key=self.item_key,
                                                    time_key=self.time_key,
                                                    time_sort=True)

    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self.model.save(self.best_model_file)

        if self.save_weights:
            self.model.save_weights(self.weights_file)

    def create_model(self,
                     # Network Arguments
                    hidden_units=100,
                    emb_size_features=50,
                    emb_size_items=None,
                    dropout_value=0.25,
                    batch_size = 20,
                    activation = "linear",
                    hidden_activation = "relu",
                    activation_parameter = 0.3,
                    # Opt arguments
                    learning_rate=0.001, 
                    beta_1=0.9, 
                    beta_2=0.999, 
                    epsilon=None, 
                    decay=0.0,
                    amsgrad=False
                    ):

        # Input 1 - One-Hot Encoding of an Item
        input1 = Input(batch_shape=(batch_size, 1, self.max_items))

        # Optional Embedding for Items
        if emb_size_items is not None:
            embedding_layer_items = Dense(emb_size_items, name="embedding_items")(input1)
            # Embedding(input_dim=max_item_number, output_dim=emb_size_items, input_length=max_item_number)(input1)
        else:
            embedding_layer_items = input1

        # Input 2 - One-Hot like Encoding of its features
        if self.item_features is not None:
            feature_number = len(self.dataset.item_features.columns)
            input2 = Input(batch_shape=(batch_size, 1, feature_number))
            # Embedding layer for features
            embedding_layer_features = Dense(emb_size_features, name="embedding_features")(input2)
            #Embedding(input_dim=max_feature_combinations, output_dim=emb_size_features, input_length=feature_number)(input2)
            next_layer = Concatenate()([embedding_layer_items, embedding_layer_features])

            inputs = [input1, input2]
        else:
            next_layer = embedding_layer_items

            inputs = [input1]

        gru, gru_states = GRU(hidden_units, stateful=True, return_state=True, activation=hidden_activation, name="GRU")(next_layer)
        drop2 = Dropout(dropout_value)(gru)
        
        if activation == "leaky":
            drop2 = Dense(self.max_items)(drop2)
            predictions  = LeakyReLU(alpha=activation_parameter)(drop2)
        elif activation == "elu":
            drop2 = Dense(self.max_items)(drop2)
            predictions  = ELU(alpha=activation_parameter)(drop2)
        else:
            predictions = Dense(self.max_items, activation=activation)(drop2)

        model = Model(inputs=inputs, outputs=[predictions])
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, 
                                    beta_1=beta_1, 
                                        beta_2=beta_2, 
                                        epsilon=epsilon, 
                                        decay=decay,
                                        amsgrad=amsgrad)
        #opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        model.compile(loss=self.loss, optimizer=opt)
        
        print("{0}: model created.".format(self.RECOMMENDER_NAME))

        return model

    def fit(self,
            # Network Arguments
            epochs = 100,
            hidden_units = 100,
            emb_size_features = 50,
            emb_size_items = None,
            dropout_value = 0.25,
            batch_size = 20,
            activation = "linear",
            hidden_activation = "relu",
            activation_parameter = 0.3,
            # Opt arguments
            learning_rate=0.001,
            loss = "bpr", 
            # beta_1=0.9, 
            # beta_2=0.999, 
            # epsilon=None, 
            # decay=0.0,
            # amsgrad=False,
            # Recommender and Evaluatore args
            save_weights=False,
            temp_file_folder="./GRU4RecKeras/",
            **earlystopping_kwargs
            ):

        self.validation_model = None

        # Args to save later
        self.hidden_units=hidden_units
        self.emb_size_features=emb_size_features if self.item_features is not None else None
        self.emb_size_items=emb_size_items
        self.dropout_value=dropout_value
        self.learning_rate = learning_rate
        self.activation = activation
        self.hidden_activation = hidden_activation
        self.activation_parameter = activation_parameter

        # Shared args
        self.epochs=epochs
        self.save_weights = save_weights
        self.temp_file_folder = temp_file_folder
        self.URM_train = self.get_URM_train()

        self.session_number = self.train_df.session_id.nunique()

        if loss == "bpr":
            self.loss = BPR
        elif loss == "bpr_max":
            self.loss = BPR_MAX

        # Adapt batch_size -> sample random session_ids to fit, or truncate it

        if not hasattr(self, "batch_size"):
            should_change_net = True
        elif self.batch_size != batch_size or self.model is None:
            should_change_net = True
        else:
            should_change_net = False

        if self.session_number < batch_size:
            # Adapt
            self.batch_size = self.session_number
            should_change_net = True

            print("{0}: batch_size set to {1}, i.e. the number of train sessions".format(self.RECOMMENDER_NAME, self.session_number))
        elif self.session_number % batch_size != 0:
            # Sample random sessions
            print("{0}: session number is not divisible by batch_size. Sampling random sessions...".format(self.RECOMMENDER_NAME))
            to_sample = batch_size - (self.session_number % batch_size)
            sessions = self.train_df.session_id.unique()
            max_session = np.max(sessions)
            sampled = sessions[np.random.choice(len(sessions), size=to_sample, replace=False)]
            sampled_mapping = {k:v for k,v in zip(sampled, np.arange(max_session, max_session+to_sample))}

            # Integrate sessions
            sampled_df = self.train_df[self.train_df[self.session_key].isin(sampled)].copy()
            sampled_df[self.session_key] = sampled_df[self.session_key].map(sampled_mapping)

            # Concat to self.train_df
            self.has_changed_df = True
            new_df = pd.concat([self.train_df, sampled_df])

            self.session_number = len(new_df.session_id.unique())
            if self.item_features is not None:
                self.dataset = ge.SessionDatasetEmbeddings(new_df, 
                                                        self.item_features.copy(), 
                                                        session_key=self.session_key, 
                                                        item_key=self.item_key,
                                                        time_key=self.time_key,
                                                        item_features_key=self.item_features_key,
                                                        time_sort=True)
            else:
                self.dataset = ge.SessionDataset(new_df, 
                                                        session_key=self.session_key, 
                                                        item_key=self.item_key,
                                                        time_key=self.time_key,
                                                        time_sort=True)
            print("{0}: dataset has changed with additional sessions.".format(self.RECOMMENDER_NAME))
            self.batch_size = batch_size
        else:
            self.batch_size = batch_size

        # Check gpu
        
        gpus = tf.config.list_physical_devices('GPU')

        if len(gpus)>0:
            print("{0}: Cuda found.".format(self.RECOMMENDER_NAME))
        else:
            print("{0}: Cuda NOT found.".format(self.RECOMMENDER_NAME))

        if not os.path.exists(self.temp_file_folder):
            os.makedirs(self.temp_file_folder)

        self.log_dir = self.temp_file_folder + '{}-{}'.format(self.RECOMMENDER_NAME, time.time())
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_model_file = self.log_dir + "/{0}.h5".format(self.RECOMMENDER_NAME)
        self.weights_file = self.log_dir + "/{0}_weights.h5".format(self.RECOMMENDER_NAME)

        # Create Network

        if should_change_net:
            self.network_arguments = {
                'hidden_units' : hidden_units,
                'emb_size_features' : emb_size_features,
                'emb_size_items' : emb_size_items,
                'dropout_value' : dropout_value,
                'learning_rate' : learning_rate,
                # 'beta_1' : beta_1,
                # 'beta_2' : beta_2,
                # 'epsilon' : epsilon,
                # 'decay': decay,
                # 'amsgrad': amsgrad,
                'activation' : activation,
                'activation_parameter': activation_parameter,
                'hidden_activation': hidden_activation
            }

            model = self.create_model(batch_size = self.batch_size, **self.network_arguments)
            self.model = model

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        if self.has_changed_df:
            self.session_number = len(self.train_df.session_id.unique())
            self.has_changed_df = False

            if self.item_features is not None:
                self.dataset = ge.SessionDatasetEmbeddings(self.train_df, 
                                                        self.item_features.copy(), 
                                                        session_key=self.session_key, 
                                                        item_key=self.item_key,
                                                        time_key=self.time_key,
                                                        item_features_key=self.item_features_key,
                                                        time_sort=True)
            else:
                self.dataset = ge.SessionDataset(self.train_df, 
                                                        session_key=self.session_key, 
                                                        item_key=self.item_key,
                                                        time_key=self.time_key,
                                                        time_sort=True)
            print("{0}: restored original sessions".format(self.RECOMMENDER_NAME))

    def clear_session(self):
        del self.model

    def _run_epoch(self, num_epoch):

        with tqdm(total=self.session_number) as pbar:
            
            if self.item_features is not None:
                loader = ge.SessionDataLoaderEmbeddings(self.dataset, batch_size=self.batch_size)

                for feat, target, features, mask in loader:

                    gru_layer = self.model.get_layer(name="GRU")
                    hidden_states = gru_layer.states[0].numpy()
                    for elt in mask:
                        hidden_states[elt, :] = 0
                    gru_layer.reset_states(states=hidden_states)

                    # Categorical representation of input
                    input_oh = to_categorical(feat, num_classes=loader.n_items)
                    input_oh = np.expand_dims(input_oh, axis=1)

                    # Categorical representation of features
                    features = np.expand_dims(features, axis=1)

                    # Categorical representation of target
                    target_oh = to_categorical(target, num_classes=loader.n_items)

                    tr_loss = self.model.train_on_batch([input_oh, features], target_oh)
                    #write_log(callbacks, ['tr_loss'], [tr_loss], epoch)

                    pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(num_epoch, tr_loss))
                    pbar.update(loader.done_sessions_counter)

            else:
                loader = ge.SessionDataLoader(self.dataset, n_items=self.max_items, batch_size=self.batch_size)

                for feat, target, mask in loader:

                    gru_layer = self.model.get_layer(name="GRU")
                    hidden_states = gru_layer.states[0].numpy()
                    for elt in mask:
                        hidden_states[elt, :] = 0
                    gru_layer.reset_states(states=hidden_states)

                    # Categorical representation of input
                    input_oh = to_categorical(feat, num_classes=loader.n_items)
                    input_oh = np.expand_dims(input_oh, axis=1)

                    # Categorical representation of target
                    target_oh = to_categorical(target, num_classes=loader.n_items)

                    tr_loss = self.model.train_on_batch(input_oh, target_oh)
                    #write_log(callbacks, ['tr_loss'], [tr_loss], epoch)

                    if np.isnan(tr_loss):
                        self.dataset.crash = feat
                        raise ValueError

                    pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(num_epoch, tr_loss))
                    pbar.update(loader.done_sessions_counter)

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield np.array(lst[i:i + n])

    def adapt_to_batch(self, model, batch_size):
        # Get network weights
        print("{0}: adapting to batch_size {1}.".format(self.RECOMMENDER_NAME, batch_size))
        wg = model.get_weights()
        model = self.create_model(batch_size = batch_size, **self.network_arguments)
        model.set_weights(wg)
        return model

    def new_predict_next(self, sessions_id_array, input_items_prediction, predict_for_item_ids, batch):

        # We need to revisit this method...
        # Reconstruct session_dataframe, and turn it into a loader! 
        # Ignore items prediction and extract from test_df the needed items...

        relevant_df = self.test_sessions_df.loc[self.test_sessions_df[self.session_key].isin(sessions_id_array)]

        # Create validation network

        self.validation_model = self.adapt_to_batch(self.model, batch)

        # recommended items
        recommended_items = []

        if self.item_features is None:
            test_dataset = ge.SessionDataset(relevant_df,
                                             session_key=self.session_key,
                                             item_key=self.item_key,
                                             time_key=self.time_key,
                                             time_sort=True)
            loader = ge.SessionDataLoader(test_dataset, self.max_items, batch)

            for feat, _, mask in loader:

                gru_layer = self.model.get_layer(name="GRU")
                hidden_states = gru_layer.states[0].numpy()
                for elt in mask:
                    hidden_states[elt, :] = 0
                gru_layer.reset_states(states=hidden_states)

                # Categorical representation of input
                input_oh = to_categorical(feat, num_classes=loader.n_items)
                input_oh = np.expand_dims(input_oh, axis=1)

                pred = self.validation_model.predict(input_oh, batch_size=batch)
                relevant_pred = pred[:, predict_for_item_ids]
                relevant_pred_indices = np.indices(pred.shape)[1][:, predict_for_item_ids]
                recommended_items.append(relevant_pred)
            
        else:
            test_dataset = ge.SessionDatasetEmbeddings(relevant_df,
                                                       session_key=self.session_key,
                                                       item_features=self.item_features,
                                                       item_key=self.item_key,
                                                       time_key=self.time_key,
                                                       item_features_key=self.item_features_key,
                                                       time_sort = True)
            loader = ge.SessionDataLoaderEmbeddings(test_dataset, self.max_items, batch)

            for feat, _, features, mask in loader:

                gru_layer = self.model.get_layer(name="GRU")
                hidden_states = gru_layer.states[0].numpy()
                for elt in mask:
                    hidden_states[elt, :] = 0
                gru_layer.reset_states(states=hidden_states)

                # Categorical representation of input
                input_oh = to_categorical(feat, num_classes=loader.n_items)
                input_oh = np.expand_dims(input_oh, axis=1)

                # Categorical representation of features
                features = np.expand_dims(features, axis=1)

                pred = self.validation_model.predict(input_oh, batch_size=batch)
                relevant_pred = pred[:, predict_for_item_ids]
                relevant_pred_indices = np.indices(pred.shape)[1][:, predict_for_item_ids]
                recommended_items.append(relevant_pred)
            
        # Get matrix
        recommended_items = np.array(recommended_items)
        recommended_items = recommended_items.reshape((len(sessions_id_array), len(predict_for_item_ids)))

        # Turn to dataframe
        df = pd.DataFrame(recommended_items, columns={k for k in predict_for_item_ids})

        return df.T

    def predict_next(self, sessions_id_array, input_items_prediction, predict_for_item_ids, batch):

        # too_much_slow... we progressively adapt to batch!

        # Check if validation model exist
        if self.validation_model is None:
            self.validation_model = self.adapt_to_batch(self.model, batch_size = batch)
            self.validation_batch_size = batch

        # Start prediction
        self.last_used_batch = self.validation_batch_size

        # Results
        batched = self.chunks(input_items_prediction, batch)
        
        recommended_items = []

        for batch_to_process in batched:

            if batch_to_process.shape[0] < 2:
                # Single sample

                input_oh = to_categorical(batch_to_process, num_classes=self.max_items)
                input_oh = np.expand_dims(input_oh, axis=0)

                if self.item_features is not None:
                    features = self.dataset.item_features.loc[batch_to_process].values
                    features = np.expand_dims(features, axis=0)
                    inputs = [input_oh, features]
                else:
                    inputs = input_oh
            else:
                # Multi sample
                input_oh = to_categorical(batch_to_process, num_classes=self.max_items)
                input_oh = np.expand_dims(input_oh, axis=1)

                if self.item_features is not None:
                    features = self.dataset.item_features.loc[batch_to_process].values
                    features = np.expand_dims(features, axis=1)
                    inputs = [input_oh, features]
                else:
                    inputs = input_oh

            batch_shape = batch_to_process.shape[0]                    

            if batch_shape != self.last_used_batch:
                self.validation_model = self.adapt_to_batch(self.model, batch_shape)
                self.validation_batch_size = batch_shape
                self.last_used_batch = batch_shape

            gru_layer = self.validation_model.get_layer(name="GRU")
            hidden_states = gru_layer.states[0].numpy()
            hidden_states[np.arange(self.validation_batch_size), :]=0
            gru_layer.reset_states(states=hidden_states)

            pred = self.validation_model.predict(inputs, batch_size=batch_shape)
            relevant_pred = pred[:, predict_for_item_ids]
            relevant_pred_indices = np.indices(pred.shape)[1][:, predict_for_item_ids]
            recommended_items.append(relevant_pred)
        
        # Get matrix
        recommended_items = np.array(recommended_items)
        recommended_items = recommended_items.reshape((len(sessions_id_array), len(predict_for_item_ids)))

        # Turn to dataframe
        df = pd.DataFrame(recommended_items, columns={k for k in predict_for_item_ids})

        return df.T

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        self.model.save(folder_path + file_name + "_{0}_model.h5".format(self.RECOMMENDER_NAME))

        data_dict_to_save = {
            "n_sessions": self.session_number,
            "n_items": self.max_items,
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

    def get_embeddings_weights(self, with_layer=False):
        layer = self.model.get_layer("embedding_features")
        weights = layer.get_weights()
        return layer, weights if with_layer else weights