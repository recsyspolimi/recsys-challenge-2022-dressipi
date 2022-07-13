import numpy as np
import pandas as pd
import random

from matplotlib import pyplot as plt

from DressipiChallenge.Pipeline.utils import create_mapping, get_mapped_sessions_to_recommend, \
    get_items_to_exclude, merge_views_purchases


def train_val_split(views_df, purchases_df,
                    n_sets: int = 1,
                    ts_start='2021-05-01',
                    ts_end='2021-06-01',
                    return_discarded=False):
    """
    Argomenti:

    * views_df: dataset composto dalle colonne ['session_id', 'item_id', 'date'] delle VIEWS di ogni sessione
    * purchases_df: dataset composto dalle colonne ['session_id', 'item_id', 'date'] dei PURCHASES di ogni sessione
    * n_sets: numero di validation sets (randomicamente splittati) da creare all'interno del periodo temporale specificato
    * ts_start: timestamp di inizio periodo validation INCLUSO (stringa in formato YYYY-MM-DD hh:mm:ss oppure parte di esso)
    * ts_end: timestamp di fine periodo validation ESCLUSO (stringa in formato YYYY-MM-DD hh:mm:ss oppure parte di esso)
    * return_discarded: valore booleano che se settato a True aggiunge ai valori di ritorno anche un df con dentro
      le sessioni 'scartate', cioè quelle successive a ts_end (se presenti)

    Specifiche:

    La funzione restituisce: train_set_df, val_set_dfs, discarded_sessions

    - train_set_df contiene:
        - VIEWS di tutte le sessioni la cui *prima view* è stata effettuata in un istante temporale precedente
          a *ts_end*
        - PURCHASES di tutte le sessioni la cui *prima view* è stata effettuata in un istante temporale precedente
          a *ts_start*
    - val_set_df(s) contiene:
        - PURCHASES di tutte la sessioni la cui *prima view* è stata effettuata in un istante temporale compreso
          negli estremi specificati (ts_start compreso, ts_end escluso).
    - discarded_sessions contiene:
        - tutte le SESSION_ID delle sessioni la cui *prima view* è stata effettuata in un istante temporale
          successivo a *ts_end* (compreso)

    Se return_discarded è settato a False il ritorno è una doppia (train_set_df, val_set_df(s))
    Se return_discarded è settato a True il ritorno è una tripla (train_set_df, val_set_df(s), discarded_sessions).

    Se n_sets > 1 i dataframe contenenti i validation sets sono ritornati all'interno di una lista di dataframes
    chiamata val_set_dfs.
    """

    if n_sets < 1:
        raise Exception('n_sets must be strictly greater than 0')

    if ts_start >= max(views_df['date'])[:10]:
        raise Exception('ts_start must be strictly lower than the latest date of training set (i.e. {}).'.format(
            max(views_df['date'])[:10]))
    if ts_end <= min(views_df['date'])[:10]:
        raise Exception('ts_end must be strictly greater than the earlier date of training set (i.e. {}).'.format(
            min(views_df['date'])[:10]))
    if ts_start >= ts_end:
        raise Exception('ts_end must be strictly greater than ts_start')

    sorted_sessions_df = views_df.sort_values(['date'], ascending=True)
    sorted_sessions_df = sorted_sessions_df.drop_duplicates(
        ['session_id'], keep='first')

    train_sessions = sorted_sessions_df[sorted_sessions_df['date']
                                        < ts_start]['session_id'].values
    val_sessions = sorted_sessions_df[(sorted_sessions_df['date'] >= ts_start) &
                                      (sorted_sessions_df['date'] < ts_end)]['session_id'].values

    train_set_df = pd.concat([views_df[views_df['session_id'].isin(train_sessions)],
                              purchases_df[purchases_df['session_id'].isin(train_sessions)]]
                             ).sort_values(['session_id', 'date'], ascending=[True, True])

    discarded_sessions = None
    if return_discarded:
        discarded_sessions = sorted_sessions_df[sorted_sessions_df['date']
                                                >= ts_end]['session_id'].values

    if n_sets == 1:
        val_set_df = purchases_df[purchases_df['session_id'].isin(
            val_sessions)]
        if return_discarded:
            return train_set_df, val_set_df, discarded_sessions
        return train_set_df, val_set_df

    val_set_dfs = []

    np.random.shuffle(val_sessions)

    remainder = len(val_sessions) % n_sets
    val_len = len(val_sessions) // n_sets + 1

    for i in range(remainder):
        val_set_dfs.append(purchases_df[purchases_df['session_id'].isin(
            val_sessions[i * val_len: (i + 1) * val_len])])

    base = remainder * val_len
    val_len -= 1

    for i in range(n_sets - remainder):
        val_set_dfs.append(
            purchases_df[purchases_df['session_id'].isin(val_sessions[base + i * val_len: base + (i + 1) * val_len])])

    if return_discarded:
        return train_set_df, val_set_dfs, discarded_sessions

    return train_set_df, val_set_dfs


def train_val_split_shuffle_v0(views_df, purchases_df,
                               n_folds: int = 1,
                               val_len: int = 50000,
                               ts_start='2021-05-01',
                               ts_end='2021-06-01',
                               random_cuts=False):
    """
    Argomenti:

    - views_df: dataset composto dalle colonne ['session_id', 'item_id', 'date'] delle VIEWS di ogni sessione
    - purchases_df: dataset composto dalle colonne ['session_id', 'item_id', 'date'] dei PURCHASES di ogni sessione
    - n_folds: numero di folds (randomicamente splittati) da creare all'interno del periodo temporale
      specificato
    - val_len: lunghezza di ogni fold in numero di sessioni
    - ts_start: timestamp di inizio periodo (stringa in formato YYYY-MM-DD hh:mm:ss oppure parte di esso)
    - ts_end: timestamp di fine periodo (stringa in formato YYYY-MM-DD hh:mm:ss oppure parte di esso)
    - random_cuts: valore booleano che se settato a True taglia via da ogni sessione un numero di view compreso tra
      lo 0% e il 50% della lunghezza di tale sessione (la sessione viene tagliata a partire dal fondo)

    Specifiche:

    La funzione restituisce: train_set_df, val_views_dfs, val_purchases_dfs
    I fold del validation set sono costruiti pescando val_len sessioni a random dal validation set totale


    - train_set_df contiene:
        - VIEWS ePURCHASES di tutte le sessioni la cui *prima view* è stata effettuata in un istante temporale precedente
          a *ts_start*
    - val_views_dfs è una lista di dataframes contenenti:
        - VIEWS di tutte la sessioni la cui *prima view* è stata effettuata in un istante temporale compreso
          negli estremi specificati (ts_start compreso, ts_end escluso)
    - val_purchases_dfs è una lista di dataframes contenenti:
        - PURCHASES di tutte la sessioni la cui *prima view* è stata effettuata in un istante temporale compreso
          negli estremi specificati (ts_start compreso, ts_end escluso)
    """

    assert n_folds > 0, 'n_folds must be strictly greater than 0'
    assert ts_start < max(views_df['date'])[:10], \
        'ts_start must be strictly lower than the latest date of training set (i.e. {}).'.format(
            max(views_df['date'])[:10])
    assert ts_end > min(views_df['date'])[:10], \
        'ts_end must be strictly greater than the earlier date of training set (i.e. {}).'.format(
            min(views_df['date'])[:10])
    assert ts_start < ts_end, 'ts_end must be strictly greater than ts_start'

    sorted_views_df = views_df.sort_values(
        ['session_id', 'date'], ascending=True)
    sorted_dropped_views_df = sorted_views_df.drop_duplicates(['session_id'], keep='first')[
        ['session_id', 'date']]

    train_sessions = sorted_dropped_views_df[sorted_dropped_views_df['date']
                                             < ts_start]['session_id'].values
    val_sessions = sorted_dropped_views_df[(sorted_dropped_views_df['date'] >= ts_start) &
                                           (sorted_dropped_views_df['date'] < ts_end)]['session_id'].values

    assert val_len <= len(val_sessions), \
        'val_len must be less or equal than the number of session in the specified time interval (i.e. {})'.format(
            len(val_sessions))

    train_set_df = pd.concat([views_df[views_df['session_id'].isin(train_sessions)],
                              purchases_df[purchases_df['session_id'].isin(train_sessions)]]
                             ).sort_values(['session_id', 'date'], ascending=[True, True])

    val_views_df = sorted_views_df[sorted_views_df['session_id'].isin(
        val_sessions)]

    if random_cuts:
        remove_mask = np.zeros(len(val_views_df), dtype=bool)
        count = 0
        n_views_arr = val_views_df.groupby('session_id').count()['item_id'].values
        for session in range(len(n_views_arr)):
            n_views = n_views_arr[session]
            n_remove = random.randint(0, n_views // 2)
            count += n_views
            remove_mask[count - n_remove: count] = True
        val_views_df = val_views_df.drop(val_views_df[remove_mask].index)

    val_views_dfs = []
    val_purchases_dfs = []

    for i in range(n_folds):
        np.random.shuffle(val_sessions)
        val_views_dfs.append(
            val_views_df[val_views_df['session_id'].isin(val_sessions[: val_len])])
        val_purchases_dfs.append(
            purchases_df[purchases_df['session_id'].isin(val_sessions[: val_len])])

    return train_set_df, val_views_dfs, val_purchases_dfs


def train_val_split_sliding_window_v1(views_df, purchases_df,
                                      window_size: int = 50000,
                                      step_size: int = 5000,
                                      n_folds: int = 5,
                                      ts_end='2021-06-01'):
    """
    Argomenti:

    - views_df: dataset composto dalle colonne ['session_id', 'item_id', 'date'] delle VIEWS di ogni sessione
    - purchases_df: dataset composto dalle colonne ['session_id', 'item_id', 'date'] dei PURCHASES di ogni sessione
    - window_size: dimensione (in numero di sessioni) della sliding window
    - step_size: dimensione (in numero di sessioni) del passo di avanzamento della sliding window
    - n_folds: numero di validation subset da creare
    - ts_end: timestamp di fine periodo (stringa in formato YYYY-MM-DD hh:mm:ss oppure parte di esso)
    - return_discarded: valore booleano che se settato a True aggiunge ai valori di ritorno anche un df con dentro
      le sessioni 'scartate', cioè quelle successive a ts_end (se presenti)

    Specifiche:

    La funzione restituisce: train_set_dfs, val_views_dfs, val_purchases_dfs

    - train_sets_dfs è una lista di dataframes contenenti:
        - VIEWS e PURCHASES di tutte le sessioni la cui *prima view* è stata effettuata in un istante temporale
          precedente alla prima sessione del validation set associatogli
    - val_views_dfs è una lista di dataframes contenenti:
        - VIEWS di tutte la sessioni la cui *prima view* è stata effettuata in un istante temporale compreso
          negli estremi specificati dalla corrispondente window
    - val_purchases_dfs è una lista di dataframes contenenti:
        - PURCHASES di tutte la sessioni la cui *prima view* è stata effettuata in un istante temporale compreso
          negli estremi specificati dalla corrispondente window
    - discarded_sessions contiene:
        - tutte le SESSION_ID delle sessioni la cui *prima view* è stata effettuata in un istante temporale
          successivo a *ts_end* (compreso)
    """
    assert ts_end > min(views_df['date']), \
        'ts_end must be strictly greater than the earlier date of training set (i.e. {}).'.format(
            min(views_df['date']))
    assert window_size > 0, 'window_size must be strictly greater than 0.'
    assert step_size > 0, 'step_size must be strictly greater than 0.'
    assert n_folds > 0, 'n_folds must be strictly greater than 0.'

    sorted_sessions_df = \
        views_df.sort_values(['date'], ascending=True).drop_duplicates(
            ['session_id'])[['session_id', 'date']]

    train_set_dfs = []
    val_views_dfs = []
    val_purchases_dfs = []
    train_val_sessions = sorted_sessions_df[sorted_sessions_df['date']
                                            < ts_end]['session_id'].values

    for i in range(n_folds - 1, 0, -1):
        train_set_dfs.append(pd.concat([
            views_df[views_df['session_id'].isin(
                train_val_sessions[: -(i * step_size + window_size)])],
            purchases_df[purchases_df['session_id'].isin(train_val_sessions[: -(i * step_size + window_size)])]]))
        val_views_dfs.append(
            views_df[views_df['session_id'].isin(
                train_val_sessions[-(i * step_size + window_size): -(i * step_size)])]
        )
        val_purchases_dfs.append(
            purchases_df[purchases_df['session_id'].isin(
                train_val_sessions[-(i * step_size + window_size): -(i * step_size)])]
        )
    train_set_dfs.append(pd.concat([views_df[views_df['session_id'].isin(train_val_sessions[:-window_size])],
                                    purchases_df[purchases_df['session_id'].isin(train_val_sessions[:-window_size])]]))
    val_views_dfs.append(
        views_df[views_df['session_id'].isin(train_val_sessions[-window_size:])])
    val_purchases_dfs.append(
        purchases_df[purchases_df['session_id'].isin(train_val_sessions[-window_size:])])

    return train_set_dfs, val_views_dfs, val_purchases_dfs


def train_val_split_sliding_window_v2(views_df, purchases_df,
                                      val_window_size: int = 50000,
                                      train_window_size: int = 100000,
                                      step_size: int = 5000,
                                      n_folds: int = 5,
                                      ts_end='2021-06-01'):
    """
    Questa è una versione leggermente modificata di train_val_split_sliding_window_v1 in cui è possibile definire una
    dimensione fissata dei train sets associati ai diversi fold del validation set. Questi avranno una finestra
    temporale che si sposterà in avanti contiguamente a quella del validation set.
    Per maggiori info sulle specifiche vedere la descrizione di train_val_split_sliding_window_v1.
    """
    assert ts_end > min(views_df['date']), \
        'ts_end must be strictly greater than the earlier date of training set (i.e. {}).'.format(
            min(views_df['date']))
    assert val_window_size > 0, 'val_window_size must be strictly greater than 0.'
    assert train_window_size > 0, 'train_window_size must be strictly greater than 0.'
    assert step_size > 0, 'step_size must be strictly greater than 0.'
    assert n_folds > 0, 'n_folds must be strictly greater than 0.'

    sorted_sessions_df = \
        views_df.sort_values(['date'], ascending=True).drop_duplicates(
            ['session_id'])[['session_id', 'date']]

    train_set_dfs = []
    val_views_dfs = []
    val_purchases_dfs = []
    train_val_sessions = sorted_sessions_df[sorted_sessions_df['date']
                                            < ts_end]['session_id'].values

    for i in range(n_folds - 1, 0, -1):
        train_set_dfs.append(pd.concat([
            views_df[views_df['session_id'].isin(
                train_val_sessions[
                    -(i * step_size + val_window_size + train_window_size): -(i * step_size + val_window_size)])],
            purchases_df[purchases_df['session_id'].isin(
                train_val_sessions[
                    -(i * step_size + val_window_size + train_window_size): -(i * step_size + val_window_size)])]]))
        val_views_dfs.append(
            views_df[views_df['session_id'].isin(
                train_val_sessions[-(i * step_size + val_window_size): -(i * step_size)])]
        )
        val_purchases_dfs.append(
            purchases_df[purchases_df['session_id'].isin(
                train_val_sessions[-(i * step_size + val_window_size): -(i * step_size)])]
        )
    train_set_dfs.append(pd.concat([
        views_df[views_df['session_id'].isin(
            train_val_sessions[-(val_window_size + train_window_size): -val_window_size])],
        purchases_df[purchases_df['session_id'].isin(
            train_val_sessions[-(val_window_size + train_window_size): -val_window_size])]]))
    val_views_dfs.append(views_df[views_df['session_id'].isin(
        train_val_sessions[-val_window_size:])])
    val_purchases_dfs.append(purchases_df[purchases_df['session_id'].isin(
        train_val_sessions[-val_window_size:])])

    return train_set_dfs, val_views_dfs, val_purchases_dfs


def keep_sessions_in_intervals(train_set_df, train_intervals_list):
    """
    Questa funzione riceve in input train_set_df e rimuove da questo tutte le sessioni la cui *prima view* è
    all'infuori degli intervalli temporali specificate dalle doppie contenuti in train_intervals_list
    """
    assert isinstance(train_intervals_list,
                      list), 'intervals_list must be a list'
    for t in train_intervals_list:
        assert isinstance(
            t, tuple), 'intervals_list elements must be tuples of strings'

    train_set_dfs = []
    no_duplicates_train_sessions = train_set_df.drop_duplicates(['session_id'])

    for t in train_intervals_list:
        assert len(t) == 2, 'all tuples must be of 2 elements'
        assert t[0] < t[1], 'first element of all tuples must be strictly lower then the second one'
        keep_sessions = no_duplicates_train_sessions[(no_duplicates_train_sessions['date'] >= t[0]) &
                                                     (no_duplicates_train_sessions['date'] < t[1])]['session_id'].values
        train_set_dfs.append(
            train_set_df[train_set_df['session_id'].isin(keep_sessions)])

    return pd.concat(train_set_dfs)


def keep_last_n_sessions(train_set_df, n):
    """
    Questa funzione riceve in input train_set_df e rimuove da esso tutte le sessioni tranne le ultime n
    (ordinate in base alla data)
    """
    keep_sessions = train_set_df.sort_values(['date'], ascending=True).drop_duplicates(
        ['session_id'])['session_id'].values[-n:]
    return train_set_df[train_set_df['session_id'].isin(keep_sessions)]


def align_purchases_to_views(views_df, purchases_df):
    return purchases_df[purchases_df['session_id'].isin(views_df['session_id'].values)]


def time_weighting(df, ref_date, time_mode, abs_time_weight, f):
    if 'score' not in df.columns.to_list():
        df['score'] = 1

    if time_mode == 1:
        df['abs_delta'] = [(ref_date - date).days for date in df.date]
        df['score'] *= round(np.exp(-abs_time_weight * df.abs_delta), 4)
    elif time_mode == 2:
        df['abs_delta'] = [(ref_date - date).days for date in df.date]
        df['score'] *= round((1 + np.cos(2 * np.pi * f * df.abs_delta)) / 2, 4)
    elif time_mode == 3:
        df['abs_delta'] = [(ref_date - date).days for date in df.date]
        df['score_1'] = round(2 + np.cos(2 * np.pi * f * df.abs_delta), 4)
        df['score_2'] = round(np.exp(-abs_time_weight * df.abs_delta), 4)
        df['score'] *= round(df['score_1'] * df['score_2'] / 3, 4)
        df.drop(columns=['score_1', 'score_2'], inplace=True)

    return df


def split_dataframes_val(
        train_sessions_df, train_purchases_df, item_features_df,
        val_start_ts='2021-05-01', val_end_ts='2021-06-01',
        unique_interactions=True, view_weight=1, purch_weight=1, abs_half_life=None, cyclic_decay=False, score_graph=False,
):
    # pd.options.mode.chained_assignment = None

    # sorted_sessions_df = train_sessions_df.sort_values(['date'], ascending=True)
    # sorted_sessions_df.drop_duplicates(['session_id'], keep='first', inplace=True)

    # train_sessions = sorted_sessions_df[sorted_sessions_df['date'] < val_start_ts]['session_id'].values
    # val_sessions = sorted_sessions_df[(sorted_sessions_df['date'] >= val_start_ts) & (sorted_sessions_df['date'] < val_end_ts)]['session_id'].values

    selected_train_views_df = train_sessions_df[train_sessions_df['date'] < val_start_ts].sort_values(by=['session_id', 'date'], ascending=[True, True]).reset_index(drop=True)
    selected_train_purchases_df = train_purchases_df[train_purchases_df['date'] < val_start_ts].sort_values(by=['session_id', 'date'], ascending=[True, True]).reset_index(drop=True)

    selected_train_views_df['score'] = view_weight
    selected_train_purchases_df['score'] = purch_weight

    train_set_df = merge_views_purchases(selected_train_views_df, selected_train_purchases_df)

    val_views_df = train_sessions_df[(train_sessions_df['date'] >= val_start_ts) & (train_sessions_df['date'] < val_end_ts)]\
        .sort_values(by=['session_id', 'date'], ascending=[True, True]).reset_index(drop=True)
    val_purch_df = train_purchases_df[(train_purchases_df['date'] >= val_start_ts) & (train_purchases_df['date'] < val_end_ts)]\
        .sort_values(by=['session_id', 'date'], ascending=[True, True]).reset_index(drop=True)

    val_views_df['score'] = view_weight
    val_purch_df['score'] = purch_weight

    recommendable_items = np.unique(val_purch_df['item_id'].values)
    items_to_ignore = get_items_to_exclude(item_features_df, recommendable_items)

    train_session_mapping = create_mapping(train_set_df['session_id'])
    val_session_mapping = create_mapping(val_purch_df['session_id'])
    item_mapping = create_mapping(item_features_df['item_id'])

    mapped_val_sessions_arr = get_mapped_sessions_to_recommend(val_purch_df, val_session_mapping)

    mapped_items_to_ignore = [item_mapping[item] for item in items_to_ignore]

    train_set_df['session_id'] = train_set_df['session_id'].map(train_session_mapping)
    train_set_df['item_id'] = train_set_df['item_id'].map(item_mapping)

    val_purch_df['session_id'] = val_purch_df['session_id'].map(val_session_mapping)
    val_purch_df['item_id'] = val_purch_df['item_id'].map(item_mapping)

    val_views_df['session_id'] = val_views_df['session_id'].map(val_session_mapping)
    val_views_df['item_id'] = val_views_df['item_id'].map(item_mapping)

    frequency = 1 / 365

    time_mode = 0
    abs_time_weight = 0
    if abs_half_life is not None:
        time_mode += 1
        abs_time_weight = 1 / abs_half_life
    if cyclic_decay:
        time_mode += 2

    train_set_df['date'] = pd.to_datetime(train_set_df['date'])
    weighted_train_set_df = time_weighting(train_set_df, pd.to_datetime(val_start_ts), time_mode, abs_time_weight, frequency)

    if score_graph:
        weighted_train_set_df.plot(x='date', y='score')
        plt.show()

    if unique_interactions:
        weighted_train_set_df.drop_duplicates(subset=['session_id', 'item_id'], inplace=True, keep='last')
        val_views_df.drop_duplicates(subset=['session_id', 'item_id'], inplace=True, keep='last')

    return weighted_train_set_df, val_purch_df, val_views_df, train_session_mapping, val_session_mapping, item_mapping,\
           mapped_items_to_ignore, mapped_val_sessions_arr


def split_dataframes_test(
        train_sessions_df, train_purchases_df, item_features_df, candidate_items_df, test_sessions_df,
        unique_interactions=True, view_weight=1, purch_weight=1, abs_half_life=None, cyclic_decay=False, score_graph=False,
):
    # pd.options.mode.chained_assignment = None

    train_sessions_df['score'] = view_weight
    train_purchases_df['score'] = purch_weight
    test_sessions_df['score'] = view_weight

    train_set_df = merge_views_purchases(train_sessions_df, train_purchases_df)
    test_sessions_df = test_sessions_df.sort_values(by=['session_id', 'date'], ascending=[True, True]).reset_index(drop=True)

    train_session_mapping = create_mapping(train_set_df['session_id'])
    item_mapping = create_mapping(item_features_df['item_id'])
    test_session_mapping = create_mapping(test_sessions_df['session_id'])

    mapped_test_sessions_arr = get_mapped_sessions_to_recommend(test_sessions_df, test_session_mapping)

    recommendable_items = candidate_items_df['item_id'].values
    items_to_ignore = get_items_to_exclude(item_features_df, recommendable_items)

    mapped_items_to_ignore = [item_mapping[item] for item in items_to_ignore]

    train_set_df['session_id'] = train_set_df['session_id'].map(train_session_mapping)
    train_set_df['item_id'] = train_set_df['item_id'].map(item_mapping)

    test_sessions_df['session_id'] = test_sessions_df['session_id'].map(test_session_mapping)
    test_sessions_df['item_id'] = test_sessions_df['item_id'].map(item_mapping)

    frequency = 1 / 365

    time_mode = 0
    abs_time_weight = 0
    if abs_half_life is not None:
        time_mode += 1
        abs_time_weight = 1 / abs_half_life
    if cyclic_decay:
        time_mode += 2

    train_set_df['date'] = pd.to_datetime(train_set_df['date'])
    weighted_train_set_df = time_weighting(train_set_df, pd.to_datetime('2021-06-01'), time_mode, abs_time_weight,
                                           frequency)

    if score_graph:
        weighted_train_set_df.plot(x='date', y='score')
        plt.show()

    if unique_interactions:
        weighted_train_set_df.drop_duplicates(subset=['session_id', 'item_id'], inplace=True, keep='last')
        test_sessions_df.drop_duplicates(subset=['session_id', 'item_id'], inplace=True, keep='last')

    return weighted_train_set_df, test_sessions_df, train_session_mapping, test_session_mapping, item_mapping, \
           mapped_items_to_ignore, mapped_test_sessions_arr