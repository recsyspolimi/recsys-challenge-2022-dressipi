import optuna
import numpy as np
import pandas as pd
import os
import pathlib
import xgboost as xgb
import joblib as jl
from datetime import datetime

from RecSys_Course_AT_PoliMi.Pipeline.optuna_utils import suggest, Categorical, Integer, Real, Range, Space
from RecSys_Course_AT_PoliMi.Pipeline.telegram_utils import telegram_bot_sendfile, telegram_bot_sendtext


def XGB_hypertune(
        candidates_df, target_df, xgb_hyp_params_dict=None,
        num_trials=500, num_folds=3, early_stopping_rounds=100, num_boost_round=1000,
        save_folder="./save", study_name='study', resume=False, with_datetime=True,
        telegram_notifications=True,
):
    class Hypertuner:

        def __init__(self, dcandidates):
            self.dcandidates = dcandidates

        def __call__(self, trial):
            chosen_data = suggest(trial, xgb_hyp_params_dict)

            print('[HYPERTUNE] Chosen parameters: ' + str(chosen_data))

            xgb_model = xgb.cv(
                chosen_data,
                dcandidates,
                # folds = group_kfold,
                num_boost_round=num_boost_round,
                nfold=num_folds,
                # metrics = ['ndcg@100' 'map@100'],
                early_stopping_rounds=early_stopping_rounds,
                as_pandas=True,
            )

            best_score = xgb_model.iloc[-1, -2]

            print('[HYPERTUNE] Best mean validation score: ' + str(best_score))
            print('[HYPERTUNE] Std of best validation score: ' + str(xgb_model.iloc[-1, -1]))

            trial.set_user_attr("best_iteration", xgb_model.index[-1])
            # trial.set_user_attr("feature_importance", xgb_model.get_score())

            del xgb_model

            return best_score

    class SaveCallback:

        def __init__(self, std_name, param_name):
            self.std_name = std_name
            self.param_name = param_name

        def __call__(self, study: optuna.Study, trial):
            jl.dump(study, self.std_name)
            jl.dump(study.best_trial.params, self.param_name)

    class TelegramCallback:

        def __init__(self, std_name):
            self.best = 0
            self.std_name = std_name

        def __call__(self, study: optuna.Study, trial):
            if study.best_value > self.best:
                self.best = study.best_value
                telegram_bot_sendtext(
                    "[XGBOOST] " + "HYPERPARAMETERS: " + str(study.best_params) + ' MAP: ' + str(self.best))
                telegram_bot_sendtext(
                    "[XGBOOST] " + "NUMBER OF ITERATIONS: " + str(study.best_trial.user_attrs['best_iteration']))
                telegram_bot_sendfile(self.std_name, "study_xgboost.pkl")

    # pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

    if xgb_hyp_params_dict is None:
        xgb_hyp_params_dict = {
            "booster": Categorical(['gbtree']),  # , 'dart', 'gblinear'
            "sampling_method": Categorical(['uniform']),  # , 'gradient_based' -> needs gpu
            "max_depth": Integer(1, 15),
            "eta": Real(1e-2, 1, prior='log-uniform'),
            "gamma": Real(1e-6, 1e-1, prior='log-uniform'),
            "min_child_weight": Real(0.01, 10, prior='log-uniform'),
            "subsample": Real(1e-3, 1, prior='log-uniform'),
            "colsample_bytree": Real(1e-5, 1e-1, prior='log-uniform'),  # [0,1]
            "colsample_bylevel": Real(1e-5, 1e-1, prior='log-uniform'),  # [0,1]
            "colsample_bynode": Real(1e-5, 1e-1, prior='log-uniform'),  # [0,1]
            "alpha": Real(1e-7, 5, prior='log-uniform'),  # [0, inf]
            "lambda": Real(1e-7, 5, prior='log-uniform'),
            # "base_score": 0, # [0, inf]
            # "num_parallel_tree": 1, # [1, inf]
        }

        '''if boosting_type == "dart":
            xgb_hyp_params_dict['sample_type'] = Categorical(['uniform', 'weighted'])
            xgb_hyp_params_dict["rate_drop"] = Real(1e-2, 0.5, prior='log-uniform')
            xgb_hyp_params_dict["skip_drop"] = Real(0.2, 0.8)'''

        fixed_params = {
            "verbosity": 2,
            "validate_parameters": True,
            "objective": 'rank:map',
            "eval_metric": 'map@100',
            "tree_method": 'hist',  # 'gpu_hist'
            "seed": 10,
            # "gpu_id": 0,
            # "n_jobs": 4,
        }

        xgb_hyp_params_dict.update(fixed_params)

    else:
        fixed_params = {k: v for k, v in xgb_hyp_params_dict.items() if
                        not isinstance(v, (Real, Integer, Categorical, Range, Space))}

    dt = ""
    if with_datetime:
        dt = datetime.now().strftime('%d-%m-%y_%H_%M_%S')
        dt = dt + '_'

    save_folder = os.path.join(save_folder, 'xgboost')
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

    dcandidates = xgb.DMatrix(
        candidates_df.drop(columns=['session_id', 'item_id']),
        label=target_df,
        qid=candidates_df['session_id'],
        nthread=-1,
        missing=np.NaN,
    )

    study.optimize(Hypertuner(dcandidates), n_trials=num_trials, callbacks=callbacks)

    best_iteration = study.best_trial.user_attrs['best_iteration']
    # feature_importance = study.best_trial.user_attrs['feature_importance']

    if telegram_notifications:
        telegram_bot_sendtext("[XGBOOST] " + "Best iteration : " + str(best_iteration))
        # telegram_bot_sendtext("[XGBOOST] " + "Feature importance : " + str(feature_importance))
        telegram_bot_sendtext("[XGBOOST] " + "Hypertuning finished.")

    best_params = {**fixed_params, **study.best_params}
    # best_params['booster'] = 'gbtree'

    print("[HYPERTUNE] Best params: ", str(best_params))
    print("[HYPERTUNE] Best iteration: ", str(best_iteration))
    # print("[HYPERTUNE] Feature importance: ", str(feature_importance))

    return best_params, best_iteration  # , feature_importance