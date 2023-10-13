import os
import gc
import json

import pickle
import pandas as pd
import polars as pl
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

from functools import partial
from typing import Tuple

from src.metric.custom_metric import competition_metric_lgb, MetricUtils

def run_lgb_experiment(
        experiment_name: str,
        config: dict, params_model: dict,
        feature_list: list, log_evaluation: int, skip_save: bool,
        dev: bool, sampling_: int
    ) -> None:
    init_metric = MetricUtils(sampling=sampling_)
    
    metric_lgb = partial(competition_metric_lgb, init_metric)
    
    init_metric.update_tolerances(config['TOLERANCES'])

    save_path = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        experiment_name
    )
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_list = []
    progress_list = []

    for fold_ in range(config['N_FOLD']):
        
        train = scan_train_parquet(
            path_file=os.path.join(
                config['DATA_FOLDER'], config['PREPROCESS_FOLDER'], 
                'train.parquet'
            ), dev=dev
        )
        print(f'\n\nStarting fold {fold_}\n\n\n')
        
        progress = {}

        callbacks_list = [
            lgb.record_evaluation(progress),
            lgb.log_evaluation(
                period=log_evaluation, 
                show_stdv=False
            )
        ]
        print('Collecting dataset')
        train_filtered = train.filter(
            pl.col('fold') != fold_
        )
        test_filtered = train.filter(
            pl.col('fold') == fold_
        )
        train_matrix = lgb.Dataset(
            train_filtered.select(feature_list).collect().to_numpy().astype('float32'),
            train_filtered.select(config['TARGET_COL']).collect().to_numpy().reshape((-1)).astype('uint8')
        )
        
        test_matrix = lgb.Dataset(
            test_filtered.select(feature_list).collect().to_numpy().astype('float32'),
            test_filtered.select(config['TARGET_COL']).collect().to_numpy().reshape((-1)).astype('uint8')
        )

        init_metric.update_df(
            test_filtered.select(
                ['series_id', 'step', 'number_event']
            ).collect().to_pandas()
        )

        print('Start training')
        model = lgb.train(
            params=params_model,
            train_set=train_matrix, 
            num_boost_round=params_model['n_round'],
            valid_sets=[test_matrix],
            valid_names=['valid'],
            callbacks=callbacks_list,
            feval=metric_lgb,
        )

        if ~skip_save:
            model.save_model(
                os.path.join(
                    save_path,
                    f'lgb_{fold_}.txt'
                )
            )

        model_list.append(model)
        progress_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()
        if ~skip_save:
            save_model(
                model_list=model_list, progress_list=progress_list,
                save_path=save_path
            )

def scan_train_parquet(path_file: str, dev: bool) -> pl.LazyFrame:
    train = pl.scan_parquet(path_file)
    
    if dev:
        selected_series_id = (
            train.select('series_id')
            .sort(by='series_id').unique()
            .head(3).collect().to_numpy().reshape((-1)).tolist()
        )
        
        train = train.filter(
            pl.col('series_id').is_in(selected_series_id)
        )
        train = (
            train.sort(['series_id', 'step']).group_by('series_id')
            .agg(pl.all().head(14400))
            .explode(pl.all().exclude("series_id"))
        )
        print(f'Using only: {train.select(pl.count()).collect().item()} rows')

    return train

def save_model(
        model_list: list, progress_list: list, save_path: str
    )->None:
        with open(
            os.path.join(
                save_path,
                'model_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)

        with open(
            os.path.join(
                save_path,
                'progress_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)

def evaluate_lgb_score(
        config: dict, experiment_name: str,
        params_model: dict, feature_list: list,
    ) -> None:

    save_path = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        experiment_name
    )
    
    # Find best epoch
    with open(
        os.path.join(
            save_path,
            'progress_list_lgb.pkl'
        ), 'rb'
    ) as file:
        progress_list = pickle.load(file)

    with open(
        os.path.join(
            save_path,
            'model_list_lgb.pkl'
        ), 'rb'
    ) as file:
        model_list = pickle.load(file)

        
    progress_dict = {
        'time': range(params_model['n_round']),
    }

    progress_dict.update(
            {
                f"event_detection_ap_fold_{i}": progress_list[i]['valid']["event_detection_ap"]
                for i in range(config['N_FOLD'])
            }
        )

    progress_df = pd.DataFrame(progress_dict)

    progress_df[f"average_event_detection_ap"] = progress_df.loc[
        :, ["event_detection_ap" in x for x in progress_df.columns]
    ].mean(axis =1)
    
    progress_df[f"std_{params_model['metric']}"] = progress_df.loc[
        :, ["event_detection_ap" in x for x in progress_df.columns]
    ].std(axis =1)

    best_epoch = int(progress_df[f"average_event_detection_ap"].argmax())
    
    best_score = progress_df.loc[
        best_epoch,
        f"average_event_detection_ap"
    ]
    std_score = progress_df.loc[
        best_epoch, f"std_event_detection_ap"
    ]

    print(f'Best epoch: {best_epoch}, CV-Event Detection: {best_score:.5f} ± {std_score:.5f}')

    best_result = {
        'best_epoch': best_epoch+1,
        'best_score': best_score
    }

    with open(
        os.path.join(
            save_path,
            'best_result_lgb.txt'
        ), 'w'
    ) as file:
        json.dump(best_result, file)

    explain_model(
        config=config, best_result=best_result, experiment_name=experiment_name, 
        model_list=model_list, feature_list=feature_list
    )


def explain_model(
        config: dict, best_result: dict, experiment_name: str,
        model_list: Tuple[lgb.Booster], feature_list: list,
    ) -> None:
    
    save_path = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        experiment_name
    )
    
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = feature_list

    for fold_, model in enumerate(model_list):
        feature_importances[f'fold_{fold_}'] = model.feature_importance(
            importance_type='gain', iteration=best_result['best_epoch']
        )

    feature_importances['average'] = feature_importances[
        [f'fold_{fold_}' for fold_ in range(config['N_FOLD'])]
    ].mean(axis=1)

    fig = plt.figure(figsize=(12,8))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
    plt.title(f"50 TOP feature importance over {config['N_FOLD']} average")

    fig.savefig(
        os.path.join(save_path, 'importance_plot.png')
    )