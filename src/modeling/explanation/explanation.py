import os
import json
import shap
import pickle
import warnings

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Union
from src.modeling.utils import scan_train_parquet

def get_shap_insight(
        save_path: str, config: dict, model: str,
        fold_: int = 0, sample_series_id: int =3, 
        batch_size: int=1024, top_n: int=5
    ):

    with open(
            os.path.join(
                save_path,
                f'model_list_{model}.pkl'
            ), 'rb'
        ) as file:
            model_list = pickle.load(file)
            
    with open(
            os.path.join(
                save_path,
                f'best_result_{model}.txt'
            ), 'r'
        ) as file:
            best_result = json.load(file)

    model_ = model_list[fold_]
    test = scan_train_parquet(
        path_file=os.path.join(
            config['DATA_FOLDER'], config['PREPROCESS_FOLDER'], 
            'train.parquet'
        ), dev=False
    ).filter(
        pl.col('fold') == fold_
    )

    series_id = (
        test
        .group_by('series_id').agg(pl.count().alias('count'))
        .sort('count', descending=True)
        .select('series_id').head(sample_series_id).unique()
        .collect().to_numpy()
        .reshape((-1)).tolist()
    )

    test = (
        test.filter(pl.col('series_id').is_in(series_id))
        .select(config['FEATURE_LIST'])
        .collect().to_pandas()
    )

    shaps = shap_by_chunk(
        model=model_, 
        test=test,batch_size=batch_size, best_result=best_result
    )
    explanation_ = shap.Explanation(
        shaps[:, :-1],
        base_values=shaps[:, -1],
        data=test.iloc[:shaps.shape[0], :],
        feature_names=config['FEATURE_LIST'],
        compute_time=10,
    )
    #summary
    print('Getting shap plot')
    save_path_shap = os.path.join(save_path, 'shap')
    
    if not os.path.exists(save_path_shap):
        os.makedirs(save_path_shap)

    shap.summary_plot(
        explanation_, features=test.iloc[:shaps.shape[0], :], 
        feature_names=config['FEATURE_LIST'], show=False
    )
    plt.savefig(
        os.path.join(save_path_shap, "shap_summary.png")
    )
    plt.close()

    col_importance = np.argsort(
        np.mean(
            np.abs(shaps[:, :-1]), axis=0
        ), 
    )[::-1]

    #top-n
    for k in range(top_n):
        col_num = col_importance[k]
        feature_name = config['FEATURE_LIST'][col_num]
        
        shap.dependence_plot(
            feature_name, 
            explanation_.values, features=test.iloc[:shaps.shape[0]], 
            feature_names=config['FEATURE_LIST'], show=False
        )
        plt.savefig(
            os.path.join(save_path_shap, f"dependance_{feature_name}.png")
        )
        plt.close()


def shap_by_chunk(
        model: Union[lgb.Booster, xgb.Booster], test: pd.DataFrame, 
        batch_size: int, best_result: dict
    )-> np.array:
    if isinstance(model, xgb.Booster):
        dataset_wrapper = lambda x: xgb.DMatrix(x)
        param_predict = {
            'pred_contribs': True, 'iteration_range': (0, best_result['best_epoch'])
        }
    elif isinstance(model, lgb.Booster):
        dataset_wrapper = lambda x: x
        param_predict = {
            'pred_contrib': True, 'num_iteration': best_result['best_epoch']
        }

    else:
        raise ValueError
    
    shap_list = []

    idxs = list(
        map(
            int, 
            np.linspace(0, test.shape[0] - 1, int(test.shape[0]/batch_size))
        )
    )[:-1]

    print('Starting calculate shap by chunk')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i1, i2 in tqdm(zip(idxs[:-1], idxs[1:]), total=len(idxs)):
            chunk_feature = dataset_wrapper(test.iloc[i1:i2, :])

            shaps = model.predict(chunk_feature, **param_predict)

            shap_list.append(shaps)
        
    shaps = np.concatenate(shap_list, axis=0)
    return shaps
