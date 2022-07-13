import numpy as np
import scipy.sparse as sp

from RecSys_Course_AT_PoliMi.Pipeline.data_splitting import split_dataframes_val, split_dataframes_test


def create_URM(df, session_mapping, item_mapping):
    """
    Weighted version: first call get_cross_validation_data_v0 with parameter weighted_URM=True;
                      otherwise add these two columns before the splitting:
                        train_sessions_df['interaction'] = 1
                        train_purchases_df['interaction'] = 2
    """

    if 'interaction' in df:
        interactions = df['interaction'].values
    else:
        interactions = np.ones(len(df), dtype=bool)

    session_ids = df['session_id'].values
    item_ids = df['item_id'].values

    mapped_session_ids = [session_mapping[elem] for elem in session_ids]
    mapped_item_ids = [item_mapping[elem] for elem in item_ids]

    URM = sp.csr_matrix((interactions,
                         (mapped_session_ids, mapped_item_ids)),
                        shape=(len(session_mapping), len(item_mapping)))

    return URM


def create_URM_GRU4Rec(df, session_mapping, item_mapping):  # mapping done before into DF for GRU4Rec
    df = df.drop(['date'], axis=1)

    df['interaction'] = bool(1)

    session_ids = df['session_id'].values
    item_ids = df['item_id'].values

    mapped_session_ids = [elem for elem in session_ids]
    mapped_item_ids = [elem for elem in item_ids]

    URM = sp.csr_matrix((df['interaction'].values,
                         (mapped_session_ids, mapped_item_ids)),
                        shape=(len(session_mapping), len(item_mapping)))

    return URM


def create_ICM(df, item_mapping, which="category_value"):
    feature_df = df.copy()
    feature_df = feature_df.sort_values(['item_id', 'feature_category_id', 'feature_value_id'])

    if which == "category_value":
        feature_df['category_value'] = feature_df['feature_category_id'].astype(str) + '-' + feature_df['feature_value_id'].astype(str)
        feature_df = feature_df.drop(['feature_value_id', 'feature_category_id'], axis=1).reset_index(drop=True)
        category_ids = feature_df.category_value.unique()
        mapping_feature = dict(zip(category_ids, np.arange(len(category_ids))))

        item_ids = [item_mapping[idx] for idx in feature_df.item_id.values]
        cat_ids = [mapping_feature[idx] for idx in feature_df.category_value.values]

        ICM = sp.csr_matrix((np.ones(shape=(len(feature_df,))), (item_ids, cat_ids)))

    elif which == "feature_value":
        feature_df = feature_df.drop(columns=['feature_category_id'])
        feature_df = feature_df.drop_duplicates()
        unique_ids = feature_df.feature_value_id.unique()
        mapping_feature = {k:v for k,v in zip(unique_ids, np.arange(len(unique_ids)))}
        
        item_ids = [item_mapping[idx] for idx in feature_df.item_id.values]
        cat_ids = [mapping_feature[idx] for idx in feature_df.feature_value_id.values]

        ICM = sp.csr_matrix((np.ones(shape=(len(feature_df,))), (item_ids, cat_ids)))
    elif which == "category_id":
        feature_df = feature_df.drop(columns=['feature_value_id'])
        feature_df = feature_df.drop_duplicates()
        unique_ids = feature_df.feature_category_id.unique()
        mapping_feature = {k:v for k,v in zip(unique_ids, np.arange(len(unique_ids)))}
        
        item_ids = [item_mapping[idx] for idx in feature_df.item_id.values]
        cat_ids = [mapping_feature[idx] for idx in feature_df.feature_category_id.values]

        ICM = sp.csr_matrix((np.ones(shape=(len(feature_df,))), (item_ids, cat_ids)))
    else:
        raise NotImplementedError

    return ICM, mapping_feature


def weight_ICM(ICM, train_df):
    itc_df = train_df.groupby('item_id').count().rename(columns={'session_id': 'count'})
    itc_df.reset_index(inplace=True)
    print(itc_df)
    csc_icm = ICM.tocsc()
    icm_weights = np.array([sum(itc_df[itc_df['item_id'].isin(
        csc_icm.indices[csc_icm.indptr[col]:csc_icm.indptr[col+1]])]['count'].values)
                            for col in range(csc_icm.shape[-1])])
    # icm_weights += 1  # some weights could be 0 because of cold items
    return ICM.multiply(icm_weights)


def create_csr_matrix(df, M, N):
    return sp.csr_matrix((df['score'].values,
                          (df['session_id'].values, df['item_id'].values)),
                         shape=(M, N))


def get_URM_split_val(
        train_sessions_df, train_purchases_df, item_features_df,
        val_start_ts='2021-05-01', val_end_ts='2021-06-01',
        unique_interactions=True, view_weight=1, purch_weight=1, abs_half_life=None, cyclic_decay=False, score_graph=False,
):
    weighted_train_set_df, val_purch_df, val_views_df, train_session_mapping, \
    val_session_mapping, item_mapping, mapped_items_to_ignore, mapped_val_sessions_arr = split_dataframes_val(
        train_sessions_df, train_purchases_df, item_features_df,
        val_start_ts=val_start_ts, val_end_ts=val_end_ts,
        unique_interactions=unique_interactions, view_weight=view_weight, purch_weight=purch_weight, abs_half_life=abs_half_life,
        cyclic_decay=cyclic_decay, score_graph=score_graph
    )
    URM_train = create_csr_matrix(weighted_train_set_df, len(train_session_mapping), len(item_mapping))
    URM_val_views = create_csr_matrix(val_views_df, len(val_session_mapping), len(item_mapping))
    URM_val_purch = create_csr_matrix(val_purch_df, len(val_session_mapping), len(item_mapping))

    return URM_train, URM_val_views, URM_val_purch, mapped_items_to_ignore, mapped_val_sessions_arr, val_session_mapping, item_mapping


def get_URM_split_test(
        train_sessions_df, train_purchases_df, item_features_df, candidate_items_df, test_sessions_df,
        unique_interactions=True, view_weight=1, purch_weight=1, abs_half_life=None, cyclic_decay=False, score_graph=False,
):
    weighted_train_set_df, test_sessions_df, train_session_mapping, test_session_mapping, \
    item_mapping, mapped_items_to_ignore, mapped_test_sessions_arr = \
        split_dataframes_test(
            train_sessions_df, train_purchases_df, item_features_df, candidate_items_df, test_sessions_df,
            unique_interactions=unique_interactions, view_weight=view_weight, purch_weight=purch_weight, abs_half_life=abs_half_life,
            cyclic_decay=cyclic_decay, score_graph=score_graph
        )

    URM_train = create_csr_matrix(weighted_train_set_df, len(train_session_mapping), len(item_mapping))
    URM_test_views = create_csr_matrix(test_sessions_df, len(test_session_mapping), len(item_mapping))

    return URM_train, URM_test_views, mapped_items_to_ignore, mapped_test_sessions_arr, test_session_mapping, item_mapping