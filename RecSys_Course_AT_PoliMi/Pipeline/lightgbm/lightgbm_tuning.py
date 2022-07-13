import os
import pathlib
import lightgbm as lgb
import optuna
import joblib as jl
import pandas as pd
from datetime import datetime

from DressipiChallenge.Pipeline.optuna_utils import suggest, Categorical, Integer, Real, Range, Space
from DressipiChallenge.Pipeline.telegram_utils import telegram_bot_sendtext, telegram_bot_sendfile


def LGB_hypertune(
        candidates_df, target_df, lgb_hyp_params_dict=None,
        num_trials=500, num_folds=3, early_stopping_rounds=100, num_boost_round=1000,
        save_folder="./save", study_name='study', resume=False, with_datetime=True,
        telegram_notifications=True,
):
    class Hypertuner:

        def __init__(self, dcandidates):
            self.dcandidates = dcandidates

        def __call__(self, trial):
            chosen_data = suggest(trial, lgb_hyp_params_dict)

            print('[HYPERTUNE] Chosen parameters: ' + str(chosen_data))

            lgb_model = lgb.cv(
                chosen_data,
                dcandidates,
                num_boost_round=num_boost_round,
                nfold=num_folds,
                return_cvbooster=False,
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
                categorical_feature=[i for i in range(73, 141)]
            )

            scores = list(lgb_model.values())[0]
            best_score = scores[-1]

            print('[HYPERTUNE] Best mean validation score: ' + str(best_score))
            print('[HYPERTUNE] Std of best validation score: ' + str(list(lgb_model.values())[1][-1]))

            trial.set_user_attr("best_iteration", scores.index(scores[-1]))
            # trial.set_user_attr("feature_importance", xgb_model.get_score())

            del lgb_model

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
                    "[LIGHTGBM] " + "HYPERPARAMETERS: " + str(study.best_params) + ' MAP: ' + str(self.best))
                telegram_bot_sendtext(
                    "[LIGHTGBM] " + "NUMBER OF ITERATIONS: " + str(study.best_trial.user_attrs['best_iteration']))
                telegram_bot_sendfile(self.std_name, "study_lightgbm.pkl")

    # pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

    if lgb_hyp_params_dict is None:
        lgb_hyp_params_dict = {
            'boosting': Categorical(['goss']),  # , 'dart', 'gbdt',
            # 'num_iterations': 1000, same as num_boost_round
            'objective': Categorical(['lambdarank']),  # , 'rank_xendcg'
            'learning_rate': Real(1e-3, 1e-1, prior='log-uniform'),
            'max_bin': Categorical([63, 127, 255, 511, 1023]),  # with gpu use smaller max_bin e.g. 63
            "max_depth": Integer(4, 15),
            'num_leaves': Integer(31, 4095),  # max 131072
            # 'subsample_for_bin': Integer(1000, 1000000),
            'min_gain_to_split': Real(1e-8, 1e-4, prior='log-uniform'),
            'min_sum_hessian_in_leaf': Real(1e-3, 1e-1, prior='log-uniform'),
            'min_data_in_leaf': Integer(2, 40),  # >= 0
            'feature_fraction': Real(0.6, 1.0, prior='log-uniform'),  # 1 is disabled,
            # 'feature_fraction_bynode': Real(0.8, 1),  # 1 is disabled
            'extra_trees': Categorical([True, False]),
            "lambda_l1": Real(1e-4, 1e-1, prior='log-uniform'),  # [0, inf]
            "lambda_l2": Real(1e-7, 0.01, prior='log-uniform'),  # [0, inf]

            # for categorical features
            'min_data_per_group': Integer(20, 200),  # minimal number of data per categorical group
            'max_cat_threshold': 32,
            'cat_l2': Real(1e-2, 20),  # L2 regularization in categorical split
            'cat_smooth': Real(1e-2, 20),
            'max_cat_to_onehot': Integer(4, 12),

            # only for goss
            "top_rate": Real(0.1, 0.3),
            "other_rate": Real(0.02, 0.2)
        }

        '''if boosting_type != "goss":
            lgb_hyp_params_dict["bagging_freq"] = Integer(0, 5)
            if lgb_hyp_params_dict["bagging_freq"] > 0:
                lgb_hyp_params_dict["bagging_fraction"] = Real(1e-2, 1.0, prior='log-uniform')
            else:
                lgb_hyp_params_dict["bagging_fraction"] = 1.

        if boosting_type == "dart":
            lgb_hyp_params_dict["n_estimators"] = Integer(400, 1000, prior='log-uniform')
            lgb_hyp_params_dict["drop_rate"] = Real(1e-2, 0.5, prior='log-uniform')
            lgb_hyp_params_dict["skip_drop"] = Real(0.2, 0.8)
            lgb_hyp_params_dict["max_drop"] = Real(5, 100, prior='log-uniform')
        elif boosting_type == "goss":
            lgb_hyp_params_dict["top_rate"] = Real(0.1, 0.3)
            lgb_hyp_params_dict["other_rate"] = Real(0.02, 0.2)'''

        fixed_params = {
            'feature_pre_filter': False,  # True for prediction
            'device_type': 'cpu',  # gpu
            'verbosity': 0,
            'metric': 'map',
            'eval_at': 100,
        }

        lgb_hyp_params_dict.update(fixed_params)

    else:
        fixed_params = {k: v for k, v in lgb_hyp_params_dict.items() if
                        not isinstance(v, (Real, Integer, Categorical, Range, Space))}

    dt = ""
    if with_datetime:
        dt = datetime.now().strftime('%d-%m-%y_%H_%M_%S')
        dt = dt + '_'

    save_folder = os.path.join(save_folder, 'lightgbm')
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

    input_df = candidates_df.drop(columns=['session_id', 'item_id'])
    dcandidates = lgb.Dataset(
        input_df,
        label=target_df,
        group=candidates_df.groupby('session_id').size().to_list(),
        feature_name=input_df.columns.to_list(),
        categorical_feature=None  # TODO declare categorical features
        # use number for index, categorical_feature=0,1,2 means column_0, column_1 and column_2 are categorical features
    )

    study.optimize(Hypertuner(dcandidates), n_trials=num_trials, callbacks=callbacks)

    best_iteration = study.best_trial.user_attrs['best_iteration']
    # feature_importance = study.best_trial.user_attrs['feature_importance']

    if telegram_notifications:
        telegram_bot_sendtext("[LIGHTGBM] " + "Best iteration : " + str(best_iteration))
        # telegram_bot_sendtext("[XGBOOST] " + "Feature importance : " + str(feature_importance))
        telegram_bot_sendtext("[LIGHTGBM] " + "Hypertuning finished.")

    best_params = {**fixed_params, **study.best_params}
    best_params['feature_pre_filter'] = True

    print("[HYPERTUNE] Best params: ", str(best_params))
    print("[HYPERTUNE] Best iteration: ", str(best_iteration))
    # print("[HYPERTUNE] Feature importance: ", str(feature_importance))

    return best_params, best_iteration  # , feature_importance
