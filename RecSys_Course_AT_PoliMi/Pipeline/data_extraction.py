import pandas as pd


def get_dataframes(path_to_dataset='./'):
    path_to_dataset += 'Dataset'
    item_features_df = pd.read_csv(path_to_dataset + '/item_features.csv', sep=',')
    train_sessions_df = pd.read_csv(path_to_dataset + '/train_sessions.csv', sep=',')
    train_purchases_df = pd.read_csv(path_to_dataset + '/train_purchases.csv', sep=',')
    test_sessions_df = pd.read_csv(path_to_dataset + '/test_leaderboard_sessions.csv', sep=',')
    candidate_items_df = pd.read_csv(path_to_dataset + '/candidate_items.csv', sep=',')
    print('CSVs read')

    return item_features_df, train_sessions_df, train_purchases_df, test_sessions_df, candidate_items_df


def load_attributes(path_to_dataset='./'):
    path_to_dataset += 'Dataset'
    session_attributes_train_df = pd.read_csv(
        path_to_dataset + '/xgb_attributes/session_attributes_train.csv', sep=',')
    session_attributes_test_df = pd.read_csv(
        path_to_dataset + '/xgb_attributes/session_attributes_leaderboard.csv', sep=',')
    item_attributes_df = pd.read_csv(
        path_to_dataset + '/xgb_attributes/item_attributes.csv', sep=',', dtype='float32')

    print('Attributes read')

    return session_attributes_train_df, session_attributes_test_df, item_attributes_df


def remove_consecutive_duplicate_itemviews(sessions_df):

    """
    La funzione deve ricevere in input un df con colonne ('session_id', 'item_id', 'date')
    e.g. train_sessions, test_leaderboard_sessions, test_final_sessions
    è importante assicurarsi che vengano rimossi item_id consecutivi SOLO se facenti parte della stessa sessione
    (potrebbe capitare che ci siano due itemview uguali consecutive ma in due sessioni diverse!)

    """

    cleaned_df = sessions_df.copy()
    cleaned_df = cleaned_df.sort_values(['session_id', 'date'], ascending=[True, True]).reset_index(drop=True)

    cleaned_df = cleaned_df[(cleaned_df['session_id'] != cleaned_df['session_id'].shift()) |
                            (cleaned_df['item_id'] != cleaned_df['item_id'].shift())].reset_index(drop=True)
    return cleaned_df
