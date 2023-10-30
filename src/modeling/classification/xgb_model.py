import os
import gc
import json

import pickle
import pandas as pd
import polars as pl
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Tuple
from functools import partial

from src.metric.custom_metric import competition_metric_xgb, MetricUtils
from src.modeling.utils import scan_train_parquet
from src.modeling.explanation.explanation import get_shap_insight
from src.modeling.postprocess.postprocess import oof_post_process_score

def run_xgb_experiment(
        experiment_name: str,
        config: dict, params_model: dict,
        feature_list: list, log_evaluation: int, skip_save: bool,
        dev: bool
    ) -> None:
    
    if isinstance(config['TARGET_COL'], list):
        assert params_model['multi_strategy'] == 'multi_output_tree'
        
    init_metric = MetricUtils()
    
    metric_xgb = partial(competition_metric_xgb, init_metric)
    
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
        
        init_metric.reset_iteration()

        train = scan_train_parquet(
            path_file=os.path.join(
                config['DATA_FOLDER'], config['PREPROCESS_FOLDER'], 
                'train.parquet'
            ), dev=dev
        )
        train_events = pl.scan_parquet(
            source=os.path.join(
                config['DATA_FOLDER'], config['PREPROCESS_FOLDER'], 
                'train_events.parquet'
            )
        )
        print(f'\n\nStarting fold {fold_}\n\n\n')
        
        progress = {}

        print('Collecting dataset')
        train_filtered = train.filter(
            pl.col('fold') != fold_
        )
        test_filtered = train.filter(
            pl.col('fold') == fold_
        )
        
        test_events_filtered = train_events.filter(
            pl.col('fold') == fold_
        )
        
        train_matrix = xgb.DMatrix(
            train_filtered.select(feature_list).collect().to_numpy().astype('float32'),
            train_filtered.select(config['TARGET_COL']).collect().to_numpy()
        )
        
        test_matrix = xgb.DMatrix(
            test_filtered.select(feature_list).collect().to_numpy().astype('float32'),
            test_filtered.select(config['TARGET_COL']).collect().to_numpy()
        )

        init_metric.update_df(
            test_filtered.select(
                ['series_id', 'step', 'year', 'month', 'day', 'hour']
            ).sort(['series_id', 'step']).collect().to_pandas()
        )
        init_metric.update_events(
            test_events_filtered.select(
                ['series_id', 'event', 'step']
            ).sort(['series_id', 'step']).collect().to_pandas()
        )

        print('Start training')
        params_train_xgb = params_model.copy()
        num_boost_round = params_train_xgb.pop('num_boost_round')
        
        model = xgb.train(
            params=params_train_xgb,
            dtrain=train_matrix, 
            num_boost_round=num_boost_round,
            evals=[(test_matrix, 'valid')],
            verbose_eval=log_evaluation,
            evals_result=progress,
            custom_metric=metric_xgb,
        )

        if ~skip_save:
            model.save_model(
                os.path.join(
                    save_path,
                    f'xgb_{fold_}.json'
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

def save_model(
        model_list: list, progress_list: list, save_path: str
    )->None:
        with open(
            os.path.join(
                save_path,
                'model_list_xgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)

        with open(
            os.path.join(
                save_path,
                'progress_list_xgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)

def oof_xgb_prediction(
        config: dict, best_result: dict, experiment_name: str, 
        model_list: Tuple[xgb.Booster], feature_list: Tuple[str]
    ) -> None:
    multi_output = isinstance(config['TARGET_COL'], list)
    
    print('Getting OOF predictions')
    
    save_path = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        experiment_name
    )
    path_train_file = os.path.join(
        config['DATA_FOLDER'], config['PREPROCESS_FOLDER'], 
        'train.parquet'
    )
    
    number_rows = scan_train_parquet(
        path_file=path_train_file, dev=False
    ).select(pl.count().alias('number_rows')).collect().item()
    
    prediction_df = pd.DataFrame(
        {
            'y_true': pd.Series([-1] * number_rows, dtype='int8'),
            'y_pred': pd.Series([-1] * number_rows, dtype='float'),
            'series_id': pd.Series([-1] * number_rows, dtype='int32'),
            'step': pd.Series([-1] * number_rows, dtype='int64')
        }
    )
    prediction_df[['series_id', 'step', 'hour', 'fold']] = scan_train_parquet(
        path_file=path_train_file, dev=False
    ).select(['series_id', 'step', 'hour', 'fold']).collect().to_numpy().tolist()
    
    for fold_ in range(config['N_FOLD']):
        model_ = model_list[fold_]
        
        dataset_test = scan_train_parquet(
            path_file=path_train_file, dev=False
        ).filter(
            pl.col('fold') == fold_
        ).select(
            feature_list + ['event', 'series_id', 'step']
        ).collect()
    
        test_x = dataset_test.select(feature_list).to_numpy().astype('float32')
        
        pred_y = model_.predict(
            xgb.DMatrix(test_x), 
            iteration_range=(0, best_result['best_epoch'])
        )

        prediction_df.loc[prediction_df['fold']==fold_, 'y_true'] = dataset_test.select('event').to_numpy()
        if multi_output:
            for dim_pred in range(len(config['TARGET_COL'])):
                prediction_df.loc[prediction_df['fold']==fold_, f'y_pred_{dim_pred}'] = pred_y[:, dim_pred]
        else:
            prediction_df.loc[prediction_df['fold']==fold_, 'y_pred'] = pred_y

        prediction_df.loc[prediction_df['fold']==fold_, 'series_id'] = dataset_test.select('series_id').to_numpy()
        prediction_df.loc[prediction_df['fold']==fold_, 'step'] = dataset_test.select('step').to_numpy()
    
    assert (prediction_df[['y_true']] == -1).mean().sum() == 0.
    
    prediction_df['y_true'] = prediction_df['y_true'].astype('uint8')
    prediction_df['series_id'] = prediction_df['series_id'].astype('uint16')
    prediction_df['step'] = prediction_df['step'].astype('uint32')

    prediction_df.to_parquet(
        os.path.join(save_path, 'oof_pred.parquet'), 
        index=False
    )
    
    
def evaluate_xgb_score(
        config: dict, experiment_name: str,
        params_model: dict, feature_list: list,
        add_comp_metric: bool, metric_to_max: str
    ) -> None:

    save_path = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        experiment_name
    )
    
    # Find best epoch
    with open(
        os.path.join(
            save_path,
            'progress_list_xgb.pkl'
        ), 'rb'
    ) as file:
        progress_list = pickle.load(file)

    with open(
        os.path.join(
            save_path,
            'model_list_xgb.pkl'
        ), 'rb'
    ) as file:
        model_list = pickle.load(file)

        
    progress_dict = {
        'time': range(params_model['num_boost_round']),
    }
    metric_to_eval = params_model['eval_metric'].copy()

    if add_comp_metric:
        metric_to_eval += ['event_detection_ap']

    for metric_ in metric_to_eval:
        progress_dict.update(
                {
                    f"{metric_}_fold_{i}": progress_list[i]['valid'][metric_]
                    for i in range(config['N_FOLD'])
                }
            )

    progress_df = pd.DataFrame(progress_dict)
    metric_line_plot = []
    
    for metric_ in metric_to_eval:
        metric_line_plot.append(f"average_{metric_}")
        
        progress_df[f"average_{metric_}"] = progress_df.loc[
            :, [metric_ in x for x in progress_df.columns]
        ].mean(axis =1)
        
        progress_df[f"std_{metric_}"] = progress_df.loc[
            :, [metric_ in x for x in progress_df.columns]
        ].std(axis =1)
        
    progress_df[['time'] + metric_line_plot].to_csv(
        os.path.join(save_path, 'metric_df.csv'), index=False
    )

    plot_df = pd.melt(progress_df[['time'] + metric_line_plot], ['time'])

    fig = plt.figure(figsize=(12,8))
    sns.lineplot(data=plot_df, x='time', y='value', hue='variable')
    plt.title(f"Metric line plot over {config['N_FOLD']} average")

    fig.savefig(
        os.path.join(save_path, 'performance_plot.png')
    )
    plt.close(fig)

    best_epoch = int(progress_df[f"average_{metric_to_max}"].argmax())
    
    best_score = progress_df.loc[
        best_epoch,
        f"average_{metric_to_max}"
    ]
    std_score = progress_df.loc[
        best_epoch, f"std_{metric_to_max}"
    ]

    print(f'Best epoch: {best_epoch}, CV-{metric_to_max}: {best_score:.5f} ± {std_score:.5f}')

    best_result = {
        'best_epoch': best_epoch+1,
        'best_score': best_score
    }

    with open(
        os.path.join(
            save_path,
            'best_result_xgb.txt'
        ), 'w'
    ) as file:
        json.dump(best_result, file)

    oof_xgb_prediction(
        config=config, best_result=best_result, experiment_name=experiment_name, 
        model_list=model_list, feature_list=feature_list
    )
    oof_post_process_score(
        config=config, experiment_name=experiment_name
    )
    
    if isinstance(config['TARGET_COL'], str):
        explain_model(
            config=config, best_result=best_result, experiment_name=experiment_name, 
            model_list=model_list, feature_list=feature_list
        )
    else:
        print('Explainability not supported for multi target')

def explain_model(
        config: dict, best_result: dict, experiment_name: str,
        model_list: Tuple[xgb.Booster], feature_list: list,
    ) -> None:
    
    save_path = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        experiment_name
    )
    
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = feature_list

    for fold_, model in enumerate(model_list):
        feature_importances[f'fold_{fold_}'] = model.get_score(
            importance_type='total_gain'#, iteration_range =(0, best_result['best_epoch'])
        )

    feature_importances['average'] = feature_importances[
        [f'fold_{fold_}' for fold_ in range(config['N_FOLD'])]
    ].mean(axis=1)

    fig = plt.figure(figsize=(12,8))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature')
    plt.title(f"50 TOP feature importance over {config['N_FOLD']} average")

    fig.savefig(
        os.path.join(save_path, 'importance_plot.png')
    )
    plt.close(fig)
    
    get_shap_insight(save_path=save_path, config=config, model='xgb')