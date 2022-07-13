from DressipiChallenge.Evaluation.Evaluator_CrossValidation import EvaluatorHoldout_CrossValidation
from DressipiChallenge.Pipeline.data_splitting import keep_sessions_in_intervals, align_purchases_to_views, \
    train_val_split_shuffle_v0, train_val_split_sliding_window_v1, train_val_split_sliding_window_v2
from DressipiChallenge.Pipeline.matrices_creation import create_URM
from DressipiChallenge.Pipeline.utils import create_mapping, create_candidate_set, get_items_to_exclude


def get_cross_validation_data_v0(train_sessions_df,
                                 train_purchases_df,
                                 item_features_df,
                                 train_intervals_list=None,
                                 weighted_URM=False,
                                 **split_kwargs):
    if train_intervals_list is not None:
        train_sessions_df = keep_sessions_in_intervals(train_sessions_df, train_intervals_list)
        train_purchases_df = align_purchases_to_views(train_sessions_df, train_purchases_df)

    if weighted_URM:
        train_sessions_df['interaction'] = 1
        train_purchases_df['interaction'] = 2

    train_set_df, val_views_dfs, val_purchases_dfs = train_val_split_shuffle_v0(train_sessions_df,
                                                                                train_purchases_df,
                                                                                **split_kwargs)

    print('Train-Val split done')

    evaluator_validation_list = []

    item_mapping = create_mapping(item_features_df['item_id'])
    train_session_mapping = create_mapping(train_set_df['session_id'])

    URM_train = create_URM(train_set_df, train_session_mapping, item_mapping)

    #  Create a different URM_train, URM_val, candidate_items set and Evaluator for each fold
    for fold in range(len(val_purchases_dfs)):
        #  URM_val
        val_session_mapping = create_mapping(val_views_dfs[fold]['session_id'])
        URM_val_views = create_URM(val_views_dfs[fold], val_session_mapping, item_mapping)
        URM_val_purchases = create_URM(val_purchases_dfs[fold], val_session_mapping, item_mapping)
        #  candidate items & items to ignore
        candidate_items = create_candidate_set(val_purchases_dfs[fold])
        items_to_ignore = get_items_to_exclude(train_set_df, candidate_items)
        mapped_items_to_ignore = [item_mapping[elem] for elem in items_to_ignore]
        #  Evaluator
        evaluator_validation_list.append(
            EvaluatorHoldout_CrossValidation(
                URM_val_views, URM_val_purchases, cutoff_list=[100], ignore_items=mapped_items_to_ignore))

    print('Mapping done, URMs created and Evaluators set')

    return URM_train, evaluator_validation_list


def get_cross_validation_data_v1(train_sessions_df,
                                 train_purchases_df,
                                 item_features_df,
                                 train_intervals_list=None,
                                 weighted_URM=False,
                                 **split_kwargs):
    if train_intervals_list is not None:
        train_sessions_df = keep_sessions_in_intervals(train_sessions_df, train_intervals_list)
        train_purchases_df = align_purchases_to_views(train_sessions_df, train_purchases_df)

    if weighted_URM:
        train_sessions_df['interaction'] = 1
        train_purchases_df['interaction'] = 2

    train_set_dfs, val_views_dfs, val_purchases_dfs = train_val_split_sliding_window_v1(train_sessions_df,
                                                                                        train_purchases_df,
                                                                                        **split_kwargs)

    print('Train-Val split done')

    URM_train_list = []
    evaluator_validation_list = []

    item_mapping = create_mapping(item_features_df['item_id'])

    #  Create a different URM_train, URM_val, candidate_items set and Evaluator for each fold
    for fold in range(len(val_purchases_dfs)):
        #  URM_train
        train_session_mapping = create_mapping(train_set_dfs[fold]['session_id'])
        URM_train_list.append(create_URM(train_set_dfs[fold], train_session_mapping, item_mapping))
        #  URM_val
        val_session_mapping = create_mapping(val_views_dfs[fold]['session_id'])
        URM_val_views = create_URM(val_views_dfs[fold], val_session_mapping, item_mapping)
        URM_val_purchases = create_URM(val_purchases_dfs[fold], val_session_mapping, item_mapping)
        #  candidate items & items to ignore
        candidate_items = create_candidate_set(val_purchases_dfs[fold])
        items_to_ignore = get_items_to_exclude(train_set_dfs[fold], candidate_items)
        mapped_items_to_ignore = [item_mapping[elem] for elem in items_to_ignore]
        #  Evaluator
        evaluator_validation_list.append(
            EvaluatorHoldout_CrossValidation(
                URM_val_views, URM_val_purchases, cutoff_list=[100], ignore_items=mapped_items_to_ignore))

    print('Mapping done, URMs created and Evaluators set')

    return URM_train_list, evaluator_validation_list


def get_cross_validation_data_v2(train_sessions_df,
                                 train_purchases_df,
                                 item_features_df,
                                 weighted_URM=False,
                                 **split_kwargs):
    if weighted_URM:
        train_sessions_df['interaction'] = 1
        train_purchases_df['interaction'] = 2

    train_set_dfs, val_views_dfs, val_purchases_dfs = train_val_split_sliding_window_v2(train_sessions_df,
                                                                                        train_purchases_df,
                                                                                        **split_kwargs)
    print('Train-Val split done')

    URM_train_list = []
    evaluator_validation_list = []

    item_mapping = create_mapping(item_features_df['item_id'])

    #  Create a different URM_train, URM_val, candidate_items set and Evaluator for each fold
    for fold in range(len(val_purchases_dfs)):
        #  URM_train
        train_session_mapping = create_mapping(train_set_dfs[fold]['session_id'])
        URM_train_list.append(create_URM(train_set_dfs[fold], train_session_mapping, item_mapping))
        #  URM_val
        val_session_mapping = create_mapping(val_views_dfs[fold]['session_id'])
        URM_val_views = create_URM(val_views_dfs[fold], val_session_mapping, item_mapping)
        URM_val_purchases = create_URM(val_purchases_dfs[fold], val_session_mapping, item_mapping)
        #  candidate items & items to ignore
        candidate_items = create_candidate_set(val_purchases_dfs[fold])
        items_to_ignore = get_items_to_exclude(train_set_dfs[fold], candidate_items)
        mapped_items_to_ignore = [item_mapping[elem] for elem in items_to_ignore]
        #  Evaluator
        evaluator_validation_list.append(
            EvaluatorHoldout_CrossValidation(
                URM_val_views, URM_val_purchases, cutoff_list=[100], ignore_items=mapped_items_to_ignore))

    print('Mapping done, URMs created and Evaluators set')

    return URM_train_list, evaluator_validation_list

