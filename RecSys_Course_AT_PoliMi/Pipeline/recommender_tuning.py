import optuna
import joblib as jl
import pathlib
import itertools as it
import numpy as np
import scipy.sparse as sps
from datetime import datetime 
import os
from DressipiChallenge.Pipeline.telegram_utils import telegram_bot_sendfile, telegram_bot_sendtext
from DressipiChallenge.Pipeline.optuna_utils import Range, Categorical, Integer, Real, Space, suggest
from DressipiChallenge.Pipeline.utils import single_train_test_model
from DressipiChallenge.Pipeline.matrices_creation import get_URM_split_val
from DressipiChallenge.Pipeline.data_extraction import get_dataframes

def hypertune (
    URM_train, URM_val_views, URM_val_purch, item_mapping, mapped_items_to_ignore,
    recommender_class,
    hyperparameters_dict,
    additional_recommender_args = {},
    num_trials = 500, save_folder = "./save", study_name = 'study', resume=False, with_datetime = True, telegram_notifications = True,
    ignore_user_to_optimize = None
    ):
    
        class Hypertuner:

            def __init__(self):
                return

            def __call__(self, trial):

                chosen_data = suggest(trial, hyperparameters_dict)

                print(chosen_data)

                results, _ = single_train_test_model(
                    URM_train = URM_train,
                    URM_val_views = URM_val_views,
                    URM_val_purch = URM_val_purch,
                    item_mapping = item_mapping,
                    recommender_class = recommender_class,
                    additional_recommender_args = additional_recommender_args,
                    hyperparameters_dict = chosen_data,
                    mapped_items_to_ignore = mapped_items_to_ignore,
                    ignore_user_to_optimize = ignore_user_to_optimize
                )

                if "epochs" in chosen_data:
                    print("[HYPERTUNING] Previous model stopped in {}".format(chosen_data['epochs']))
                    trial.set_user_attr("best_epoch", chosen_data['epochs'])

                return results.MRR.to_list()[0]

        class SaveCallback:

            def __init__(self, std_name, param_name):
                self.std_name=std_name
                self.param_name = param_name

            def __call__(self, study: optuna.Study, trial):
                jl.dump(study, self.std_name)

                if 'best_epoch' in trial.user_attrs:
                    best_epoch = study.best_trial.user_attrs['best_epoch']
                    study.best_trial.params['epochs'] = best_epoch
                
                jl.dump(study.best_trial.params, self.param_name)

        class TelegramCallback:

            def __init__(self, std_name):
                self.best = 0
                self.std_name=std_name

            def __call__(self, study: optuna.Study, trial):
                if study.best_value > self.best:
                    self.best = study.best_value
                    best_params = study.best_params
                    if 'best_epoch' in trial.user_attrs:
                        best_epoch = study.best_trial.user_attrs['best_epoch']
                        best_params['epochs'] = best_epoch
                    telegram_bot_sendtext("[" + recommender_class.RECOMMENDER_NAME + "] " + "HYPERPARAMETERS: " + str(best_params) + ' MRR: ' + str(self.best))
                    telegram_bot_sendfile(self.std_name, "study_" + recommender_class.RECOMMENDER_NAME + ".pkl")

        dt = ""
        if with_datetime:
            dt = datetime.now().strftime('%d-%m-%y_%H_%M_%S')
            dt = dt + '_'
        
        save_folder = os.path.join(save_folder, recommender_class.RECOMMENDER_NAME)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            study = optuna.create_study(direction='maximize')
            study_path = os.path.join(save_folder, dt + study_name + '.pkl')
            param_path = os.path.join(save_folder, dt + study_name + "_best_parameters" + '.pkl')
        else:
            matches = list(pathlib.Path(os.path.join(save_folder)).glob('*' + study_name + '.pkl'))
            param_path = os.path.join(save_folder, dt + study_name + "_best_parameters" + '.pkl')
            if resume and len(matches) > 0:
                filename = max([str(m) for m in matches], key=os.path.getctime)
                print('[HYPERTUNE] Loading from file: ' + filename)
                with open(filename, 'rb') as f:
                    study = jl.load(f)
                study_path = filename
            else:
                study = optuna.create_study(direction='maximize')
                study_path = os.path.join(save_folder, dt + study_name + '.pkl')
            
        callbacks = []
        callbacks.append(SaveCallback(study_path, param_path))

        if telegram_notifications:
            callbacks.append(TelegramCallback(study_path)) 

        study.optimize(Hypertuner(),n_trials=num_trials, callbacks=callbacks)
    
        if telegram_notifications:
            telegram_bot_sendtext("[" + recommender_class.RECOMMENDER_NAME + "] " + "Hypertuning finished.")

        best_params = study.best_params
        print("[HYPERTUNE] Best params: ", best_params)

        return best_params

def URM_tuning(recommender_class_list, best_hyperparameters_list, additional_arguments_list,
               view_weight_values = [0.8, 0.5, 0.2],
               abs_hal_life_values = [365, int(365/2)],
               decay_values = [True, False],
               unique_interactions_values = [True, False],
               custom_URM_handler = None,
               ignore_users = None):
    '''
    For each recommender in the list, it computes the best URM hyperparameters
    '''

    assert custom_URM_handler is None or isinstance(custom_URM_handler, list) or isinstance(custom_URM_handler, tuple)

    def produce_URM(URM_train, URM_val_views, URM_val_purch, additional_recommender_args):
        return URM_train, URM_val_views, URM_val_purch, additional_recommender_args

    best_params = {k.RECOMMENDER_NAME:{'MRR': -np.inf} for k in recommender_class_list}

    handlers = [produce_URM for _ in range(len(recommender_class_list))]
    if custom_URM_handler is not None:
        for custom_handler in custom_URM_handler:
            handlers[custom_handler[0]] = custom_handler[1]

    # Produce all combinations
    combinations = it.product(view_weight_values, abs_hal_life_values, decay_values, unique_interactions_values)
    vwidx = 0
    abidx = 1
    deidx = 2
    uiidx = 3

    if ignore_users is None:
        ignore_users = [None] * len(recommender_class_list)

    for comb in combinations:
        
        comb_print = {k:v for k,v in zip(['view_weight', 'abs_halflife', 'decay', 'unique_inter'], comb)}

        # Produce URM with given combination
        item_features_df, train_sessions_df, train_purchases_df, test_sessions_df, candidate_items_df = get_dataframes()

        URM_train, URM_val_views, URM_val_purch, mapped_items_to_ignore, mapped_val_sessions_arr, val_session_mapping, item_mapping = get_URM_split_val(
            item_features_df=item_features_df,
            train_purchases_df=train_purchases_df,
            train_sessions_df=train_sessions_df,
            val_start_ts='2021-05-01',
            val_end_ts='2021-06-01',
            unique_interactions=comb[uiidx],
            abs_half_life=comb[abidx],
            cyclic_decay=comb[deidx],
            purch_weight=1,
            view_weight=comb[vwidx],
            score_graph=False,
        )

        for i, model_class in enumerate(recommender_class_list):
            # Execute evaluation

            URM_tr, URM_va, URM_te, additional_recommender_args_model = handlers[i](URM_train, 
                                                                                    URM_val_views, 
                                                                                    URM_val_purch, 
                                                                                    additional_arguments_list[i])

            results, _ = single_train_test_model(model_class, 
                                                 URM_tr, 
                                                 URM_va, 
                                                 URM_te, 
                                                 mapped_items_to_ignore, 
                                                 item_mapping, 
                                                 best_hyperparameters_list[i],
                                                 additional_recommender_args=additional_recommender_args_model,
                                                 ignore_user_to_optimize=ignore_users[i])
            obtained_MRR = results.loc[100]['MRR']

            print("{0}: MRR {1} for attributes: {2}".format(model_class.RECOMMENDER_NAME,
                                                            obtained_MRR,
                                                            comb_print))

            if obtained_MRR > best_params[model_class.RECOMMENDER_NAME]['MRR']:
                best_params[model_class.RECOMMENDER_NAME] = comb_print.copy()
                best_params[model_class.RECOMMENDER_NAME]['MRR'] = obtained_MRR

    print("RESULT\n", best_params)
    return best_params


    