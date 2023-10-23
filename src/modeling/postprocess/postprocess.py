import os
import numpy as np
import pandas as pd

import polars as pl

from tqdm import tqdm

from src.metric import official_metric

def calculate_onset_wakeup(df: pd.DataFrame, series_id: int, window:int=30) -> pd.DataFrame:
    skip_logic = [np.inf]
    current_ = None
    
    onset_list = []
    wakeup_list = []
    onset_pred_list, wakeup_pred_list = [], []

    series_df = df.loc[df['series_id'] == series_id].reset_index(drop=True)

    for iter_, row in series_df.iterrows():
        
        #reset
        if iter_ > max(skip_logic):
            skip_logic = [np.inf]
        
        #skip evaluation
        if iter_ in skip_logic:
            continue
        
        #onset start
        if (current_ == 'wakeup') | (current_ is None):
            if row['y_pred'] >=0.5:
                future_window = series_df.loc[iter_:]
                future_30_min = future_window.loc[:(iter_+window)]
                
                #check if at least 30 min 
                if (
                    (future_30_min['y_pred'].mean() >= 0.5) & 
                    (future_30_min['y_pred'].iloc[-1] >= 0.5)
                ):

                    #find best future window to skip
                    next_skip = list(future_window.loc[future_window['y_pred']<0.5].index)
                    if len(next_skip) > 5:
                        skip_window = next_skip[0]
                        skip_logic = list(range(iter_, skip_window))

                    onset_list.append(row['onset_step'])
                    onset_pred_list.append(future_30_min['y_pred'].mean())
                    current_ = 'onset'
                
        #wakeup start
        if (current_ == 'onset')| (current_ is None):
            if row['y_pred'] < 0.5:
                future_window = series_df.loc[iter_:]
                future_30_min = future_window.loc[:(iter_+window)]
                
                if (
                    (future_30_min['y_pred'].mean() < 0.5) & 
                    (future_30_min['y_pred'].iloc[-1] < 0.5)
                ):
                    
                    #find best future window to skip
                    next_skip = list(future_window.loc[future_window['y_pred']>0.5].index)
                    if len(next_skip) > 5:
                        skip_window = next_skip[0]
                        skip_logic = list(range(iter_, skip_window))

                    
                    wakeup_list.append(int(row['wakeup_step']))
                    wakeup_pred_list.append(future_30_min['y_pred'].mean())
                    current_ = 'wakeup'
                    
    onset_df = pd.DataFrame(
        {
            'event': 1, #'onset',
            'series_id': [int(series_id)]*len(onset_list),
            'step': onset_list,
            'score': onset_pred_list
        }
    )
    wakeup_df = pd.DataFrame(
        {
            'event': 0, #'wakeup',
            'series_id': [int(series_id)]*len(wakeup_list),
            'step': wakeup_list,
            'score': wakeup_pred_list
        }
    )
    res = pd.concat([onset_df, wakeup_df], axis=0)
    res = res.sort_values(['step']).reset_index(drop=True)
    return res

def format_preprocess_dataset(oof_pred: pl.LazyFrame, num_minutes: int=30) -> pd.DataFrame:
    oof_pred = oof_pred.with_row_count().with_columns(
        (pl.col('row_nr')//12).over('series_id').alias('block_nr'),
        pl.col('y_pred').rolling_mean(
            window_size=f'{12*num_minutes}i', center=False,
            closed='left', min_periods=12*num_minutes
        ).over('series_id').shift(-12*num_minutes)
    )


    oof_pred = oof_pred.group_by(
        ['series_id', 'block_nr']
    ).agg(
        pl.col('y_pred').mean(),
        pl.col('row_nr').filter(pl.col('y_pred')==pl.max('y_pred'))
    ).with_columns(
        pl.col('row_nr').list.last().alias('wakeup_step'),
        pl.col('row_nr').list.first().alias('onset_step'),
    ).drop_nulls().drop('row_nr').sort(['series_id', 'block_nr'])

    oof_pred_df = oof_pred.collect().to_pandas()
    return oof_pred_df

def oof_post_process_score(config: dict, experiment_name: str) -> pd.DataFrame:
    save_path = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        experiment_name
    )
    train_events = pl.scan_parquet(
        source=os.path.join(
            config['DATA_FOLDER'], config['PREPROCESS_FOLDER'], 
            'train_events.parquet'
        )
    ).select(
        ['series_id', 'event', 'step']
    ).sort(['series_id', 'step']).collect().to_pandas()
    
    oof_pred = pl.scan_parquet(
        os.path.join(save_path, 'oof_pred.parquet')
    )
    oof_pred_df = predict_post_process(oof_pred=oof_pred)
    
    score_oof = official_metric.score(
        solution=train_events,
        submission=oof_pred_df,
        tolerances=config['TOLERANCES'],
        series_id_column_name='series_id',
        time_column_name='step',
        event_column_name='event',
        score_column_name='score',
    )
    print(f'Official metric oof score: {score_oof}')    

def predict_post_process(oof_pred: pl.LazyFrame) -> pd.DataFrame:
    oof_pred_df = format_preprocess_dataset(oof_pred=oof_pred)
    
    result = [
        calculate_onset_wakeup(oof_pred_df, series_id)
        for series_id in tqdm(oof_pred_df['series_id'].unique())
    ]
        
    
    return pd.concat(result, ignore_index=True).sort_values(['series_id', 'step'])