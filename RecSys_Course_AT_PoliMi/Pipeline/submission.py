import os
from datetime import datetime

import numpy as np
import pandas as pd


def build_submission_df(ranking_lists, session_ids, inverse_item_mapping, cutoff=100):

    """
      La funzione riceve come input una ranking list per ciascuna sessione della quale deve essere fornita una raccomandazione
      (quindi una list of lists), l'array dei session_id corrispondenti e quello relativo al mapping delle item

      In pratica riceve come primo parametro l'output di ritorno del metodo "recommend" della classe "BaseRecommender"
      presente nella repository "RecSys_Course_AT_PoliMi"

      Si assume che la funzione "recommend" riceva come parametri:

        - l'array dei session_id di cui si vuole ottenere un ranking di raccomandazione ("user_id_array"),
        - "cutoff" settato a 100,
        - "remove_seen_flag" settato a True,
        - "remove_custom_items_flag" settato a True (*)

      La funzione ritorna il dataframe contenente i ranking di raccomandazione nel formato corretto.

      (*) precedentemente alla chiamata della funzione "recommend" è necessario definire le items da escludere dalla raccomandazione
      chiamando il metodo "set_items_to_ignore" della classe "BaseRecommender" assegnando come parametro il risultato della
      chiamata alla funzione "get_items_to_exclude" disponibile nel file "utils.py".

    """

    assert np.shape(ranking_lists)[0] == len(session_ids) and np.shape(ranking_lists)[-1] == cutoff

    recommendation_lists = list([[], [], []])

    for recommended_items, session_id in zip(ranking_lists, session_ids):
        recommendation_lists[0].extend([session_id for _ in range(cutoff)])  # contenente il session_id ripetuto 100 volte
        recommendation_lists[1].extend([inverse_item_mapping[i] for i in recommended_items])  # contenente le item nell'ordine di ranking
        recommendation_lists[2].extend([r for r in range(1, cutoff+1)])  # contenente le posizioni delle item nel ranking, da 1 a 100

    recommendation_arr = np.array(recommendation_lists).T  # trasposizione per creare la struttura finale da trasformare in dataframe
    recommendation_df = pd.DataFrame(recommendation_arr)
    recommendation_df.columns = ['session_id', 'item_id', 'rank']

    return recommendation_df


def create_submission(prediction_df, item_mapping, session_mapping, save_path="./subs/", cutoff=100):
    # pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

    prediction_df = prediction_df[['session_id', 'item_id']]
    num_sessions = len(np.unique(prediction_df.session_id))
    rank_col = list(range(1, cutoff+1)) * num_sessions

    prediction_df['rank'] = rank_col

    # create inverse mapping
    inv_item_map = {v: k for k, v in item_mapping.items()}
    inv_session_map = {v: k for k, v in session_mapping.items()}

    # map to original ids
    prediction_df['item_id'] = prediction_df['item_id'].map(inv_item_map)
    prediction_df['session_id'] = prediction_df['session_id'].map(inv_session_map)

    if not os.path.exists(save_path):
        # Create a new directory because it does not exist
        os.makedirs(save_path)

    now = datetime.now()  # current date and time
    final_path = os.path.join(save_path, 'submission_{:%Y_%m_%d_at_%H_%M_%S}.csv'.format(now))
    prediction_df.to_csv(final_path, index=False)
    print("SUBMISSION COMPLETED")

    return prediction_df
