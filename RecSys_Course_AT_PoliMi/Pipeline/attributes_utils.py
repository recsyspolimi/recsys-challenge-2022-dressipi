import math
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

from RecSys_Course_AT_PoliMi.Pipeline.data_extraction import get_dataframes

tqdm.pandas()


def compute_mean(x, y):
    if y != 0:
        return x / y
    else:
        return -1


def compute_quartet_entropy(wi, sp, su, au):
    total = wi + sp + su + au
    if total == 0:
        return -1
    if wi == 0:
        wi_coeff = 0
    else:
        wi_coeff = - (wi / total) * math.log2(wi / total)
    if sp == 0:
        sp_coeff = 0
    else:
        sp_coeff = - (sp / total) * math.log2(sp / total)
    if su == 0:
        su_coeff = 0
    else:
        su_coeff = - (su / total) * math.log2(su / total)
    if au == 0:
        au_coeff = 0
    else:
        au_coeff = - (au / total) * math.log2(au / total)
    return 1 - ((wi_coeff + sp_coeff + su_coeff + au_coeff) / 2)


def compute_season(x, record):
    month = x.date.month
    elems = record[x.item_id]
    if month in [12, 1, 2]:
        elems[0] += 1
    elif month in [3, 4, 5]:
        elems[1] += 1
    elif month in [6, 7, 8]:
        elems[2] += 1
    else:
        elems[3] += 1


def compute_seasonality_tendency(df, attributes):
    print('Computing Seasonality Tendency...')
    assert len(attributes) == 4
    return df.progress_apply(lambda x:
                             compute_quartet_entropy(x[attributes[0]], x[attributes[1]],
                                                     x[attributes[2]], x[attributes[3]]),
                             axis=1)


def compute_season_tendency(df, target, columns):
    assert target in columns
    print('Computing {} Tendency...'.format(target))
    return df.progress_apply(lambda x:
                             compute_mean(x[target],
                                          x[columns[0]] + x[columns[1]] +
                                          x[columns[2]] + x[columns[3]]),
                             axis=1)


def extract_season(sessions, item_features, columns, attr_type='views'):
    assert type(attr_type) == str
    assert len(columns) == 4
    sessions.date = pd.to_datetime(sessions.date)
    record = {item: [0, 0, 0, 0] for item in item_features.item_id.unique()}

    print('Building Statistics...')
    sessions.progress_apply(lambda x: compute_season(x, record), axis=1)
    season_df = pd.DataFrame.from_dict(record, orient='index', columns=columns)
    season_df['seasonality_' + attr_type + '_tendency'] = compute_seasonality_tendency(season_df, columns)
    season_df[columns[0] + '_tendency'] = compute_season_tendency(season_df, columns[0], columns)
    season_df[columns[1] + '_tendency'] = compute_season_tendency(season_df, columns[1], columns)
    season_df[columns[2] + '_tendency'] = compute_season_tendency(season_df, columns[2], columns)
    season_df[columns[3] + '_tendency'] = compute_season_tendency(season_df, columns[3], columns)
    return season_df


def get_item_attributes(dataset_path, path, init_date='2020-01-01', end_date='2021-05-31', use_base_features=False):
    """
    init_date is inclusive
    end_date is exclusive
    dataset_path is the relative path to the Dataset directory to pass to get_dataframes
    path is the name of the file to load/create (if not present it will create a file
    with name 'path_init_date_end_date.csv')
    When use_base_features is True creates simil-one-hot-encoding based on original dataset features
    """
    path = path.split('.csv')[0] + '_' + init_date.replace('-', '_') + '_' + end_date.replace('-', '_') + '.csv'
    if os.path.exists(path):
        print('Attributes already computed, reloading...')
        return pd.read_csv(path)
    print('Attributes not computed, creating...')
    item_features_df, train_sessions_df, train_purchases_df, test_sessions_df, candidate_items_df = get_dataframes(
        dataset_path)
    train_sessions_df = train_sessions_df[(train_sessions_df.date >= init_date) & (train_sessions_df.date < end_date)]
    train_purchases_df = train_purchases_df[
        (train_purchases_df.date >= init_date) & (train_purchases_df.date <= end_date)]

    columns_views = ['winter_views', 'spring_views', 'summer_views', 'autumn_views']
    season_views_df = extract_season(train_sessions_df, item_features_df, columns_views, 'views')
    columns_purchases = ['winter_purchases', 'spring_purchases', 'summer_purchases', 'autumn_purchases']
    season_purchases_df = extract_season(train_purchases_df, item_features_df, columns_purchases, 'purchases')

    season_df = season_views_df.merge(right=season_purchases_df, left_index=True, right_index=True)
    season_df.insert(0, 'item_id', season_df.index)

    season_df['total_views'] = season_df.apply(lambda x: x[columns_views[0]] + x[columns_views[1]] +
                                                         x[columns_views[2]] + x[columns_views[3]], axis=1)
    season_df['total_purchases'] = season_df.apply(lambda x: x[columns_purchases[0]] + x[columns_purchases[1]] +
                                                             x[columns_purchases[2]] + x[columns_purchases[3]], axis=1)

    if use_base_features:
        item_features_unstack = simil_one_hot_mapping(item_features_df, 'feature_category_id',
                                                      'feature_value_id', 'item_id')
        season_df = item_features_unstack.merge(right=season_df, left_on='item_id', right_on='item_id')
        season_df.reset_index(inplace=True)
        season_df.drop(columns=['index'], inplace=True)

    season_df.to_csv(path, index=False)
    return season_df


class VariationalAutoEncoder(keras.Model):
    def __init__(self, inputShape, batchSize, latentSize):
        super(VariationalAutoEncoder, self).__init__()
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.latentSize = latentSize
        self.input_layer = keras.Input(shape=self.inputShape)
        self.e2 = keras.layers.Dense(units=self.latentSize * 4)(self.input_layer)
        self.b2 = tf.keras.layers.BatchNormalization()(self.e2)
        self.r2 = tf.keras.layers.LeakyReLU(alpha=0.3)(self.b2)
        self.d2 = tf.keras.layers.Dropout(0.2, seed=42)(self.r2)
        self.z_mean = keras.layers.Dense(self.latentSize)(self.d2)
        self.z_log_sigma = keras.layers.Dense(self.latentSize)(self.d2)
        self.z = keras.layers.Lambda(self.sampling)([self.z_mean, self.z_log_sigma])
        self.encoder = keras.Model(self.input_layer, [self.z_mean, self.z_log_sigma, self.z], name='encoder')
        self.latent_inputs = keras.Input(shape=(self.latentSize,), name='z_sampling')
        self.e5 = keras.layers.Dense(units=self.latentSize * 4)(self.latent_inputs)
        self.b5 = tf.keras.layers.BatchNormalization()(self.e5)
        self.r5 = tf.keras.layers.LeakyReLU(alpha=0.3)(self.b5)
        self.d5 = tf.keras.layers.Dropout(0.2, seed=42)(self.r5)
        self.output_layer = keras.layers.Dense(self.inputShape[0], activation='sigmoid')(self.d5)
        self.decoder = keras.Model(self.latent_inputs, self.output_layer, name='decoder')
        self.output_layer = self.decoder(self.encoder(self.input_layer)[2])
        self.vae = keras.Model(self.input_layer, self.output_layer, name='vae_mlp')

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latentSize),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def call(self, vector):
        z_mean, z_log_sigma, z = self.encoder(vector)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma) + 1
        )
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        return reconstructed

    def createEmbedding(self, vector):
        return self.encoder(vector)


def simil_one_hot_mapping(df, column_1, column_2, column_index):
    assert type(column_1) == str and type(column_2) == str and type(column_index) == str
    assert column_1 in df.columns and column_2 in df.columns and column_index in df.columns
    cp = df.copy(deep=True)
    cp['mapping'] = cp[column_1].astype(str) + '-' + cp[column_2].astype(str)
    keys = list(cp['mapping'].unique())
    values = [v for v in range(len(keys))]
    mapping_dict = dict(zip(keys, values))
    cp['mapping'] = cp['mapping'].map(mapping_dict)
    cp.drop([column_1, column_2], axis=1, inplace=True)
    cp['value'] = 1
    cp = cp.pivot(index=column_index, columns='mapping', values='value')
    cp.fillna(value=0, inplace=True)
    cp.reset_index(inplace=True)
    return cp


def get_embeddings(dataset_df, epochs, batch_size, learning_rate, validation_split, latent_dim,
                   patience_early, patience_reduce, path, one_hot=False):
    path = path.split('.csv')[0] + '_' + str(latent_dim) + '.csv'
    if os.path.exists(path):
        print('Embeddings already computed, reloading...')
        return pd.read_csv(path)
    print('Embeddings not computed, creating...')
    assert dataset_df.columns[0] == 'item_id'
    if not one_hot:
        dataset_df = simil_one_hot_mapping(dataset_df, 'feature_category_id', 'feature_value_id', 'item_id')
    dataset_tensor = tf.convert_to_tensor(dataset_df.copy(deep=True)[dataset_df.columns[1:]].values)
    auto_encoder = VariationalAutoEncoder(inputShape=(dataset_tensor.shape[1],),
                                          batchSize=batch_size,
                                          latentSize=latent_dim)
    auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                         loss=keras.losses.MeanSquaredError())
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce, min_lr=1e-7,
                                             verbose=1, cooldown=0)]
    auto_encoder.fit(dataset_tensor, dataset_tensor,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_split=validation_split,
                     callbacks=[callbacks],
                     verbose=1)
    encoded_df = pd.DataFrame(auto_encoder.createEmbedding(dataset_tensor)[2])
    encoded_df.insert(0, 'item_id', dataset_df.item_id)
    encoded_df.to_csv(path, index=False)
    return encoded_df


def compute_aggregation_score(item_list, aggregator, values):
    rows = values.loc[values.item_id.isin(item_list)]
    if aggregator == 'sum':
        return np.sum(np.array(rows[rows.columns[1:]]), axis=0)
    if aggregator == 'mean':
        return np.mean(np.array(rows[rows.columns[1:]]), axis=0)
    if aggregator == 'prod':
        return np.prod(np.array(rows[rows.columns[1:]]), axis=0)


def get_session_views_embeddings(sessions, dataset_path, latent_dim, aggregator, path):
    assert aggregator in ['sum', 'mean', 'prod']
    path = path.split('.csv')[0] + '_' + str(latent_dim) + '_' + str(aggregator) + '.csv'
    if os.path.exists(path):
        print('Embedding already computed, reloading...')
        return pd.read_csv(path)
    print('Embeddings not computed, creating...')
    item_features_df = pd.read_csv(dataset_path + 'Dataset/item_features.csv', sep=',')
    item_attributes = simil_one_hot_mapping(item_features_df, 'feature_category_id', 'feature_value_id', 'item_id')
    item_embeddings = get_embeddings(dataset_df=item_attributes[item_attributes.columns[:905]], epochs=200, batch_size=32,
                            learning_rate=1e-3, validation_split=0.2, latent_dim=latent_dim, patience_early=10,
                            patience_reduce=5, path='../../Dataset/item_embeddings.csv', one_hot=True)
    session_dict = sessions.groupby(['session_id'])['item_id'].progress_apply(lambda items:
                                                                           list(items.value_counts().index)).to_dict()
    session_scores_dict = {key: compute_aggregation_score(value, aggregator, item_embeddings)
                           for key, value in tqdm(session_dict.items())}

    session_scores_df = pd.DataFrame.from_dict(session_scores_dict, orient='index')
    session_scores_df.reset_index(inplace=True)
    session_scores_df.rename(columns={session_scores_df.columns[0]: 'session_id'}, inplace=True)
    session_scores_df.to_csv(path, index=False)
    return session_scores_df
