import numpy as np
import pandas as pd

from typing import List

pd.options.mode.chained_assignment = None  # default='warn'

def detection_event_only_prediction(
        submission: pd.DataFrame, y_pred: np.array,
        prediction_rolling: int=360, score_rolling: int=360,
        col_select: List[str] = ['series_id','step','event','score']
    ) -> pd.DataFrame:

    submission['event'] = np.argmax(y_pred, axis=1).astype(int)
    submission.loc[submission['event']==2, 'event'] = np.nan
    
    submission['event'] = (
        submission.groupby('series_id')['event']
        .transform(lambda group: group.rolling(prediction_rolling+1, center=True).median())
    )
    mask_event = submission['event'].notna()
    
    pred_ = pd.DataFrame(y_pred)
    pred_['series_id'] = submission['series_id']

    pred_ = pred_.groupby('series_id')[[0, 1, 2]].transform(
        lambda group: 
            group.rolling(score_rolling+1, center=True, min_periods=10)
            .mean().bfill().ffill()
    ).values
    submission = submission.loc[mask_event].reset_index(drop=True)
    y_pred = (1-y_pred[mask_event, 0])

    submission['score'] = y_pred

    sub_onset = submission[submission['event']==1].groupby(['series_id', 'year', 'month', 'day']).agg('first').reset_index()
    sub_wakeup = submission[submission['event']==0].groupby(['series_id', 'year', 'month', 'day']).agg('last').reset_index()

    result = pd.concat([sub_wakeup, sub_onset], ignore_index=True)[col_select].sort_values(['series_id', 'step'])

    return result

def detection_prediction(
        submission: pd.DataFrame, y_pred: np.array,
        prediction_rolling: int=720, score_rolling: int=12*60*5,
        col_select: List[str] = ['series_id','step','event','score']
    ) -> pd.DataFrame:
    
    submission['probability'] = y_pred
    submission['prediction'] = (y_pred > 0.5).astype(int)
    
    submission['score'] = (
        submission.groupby('series_id')['probability']
        .transform(
            lambda group: 
                group.rolling(score_rolling+1, center=True, min_periods=10)
                .mean().bfill().ffill()
        )
    )
    
    submission['prediction'] = (
        submission.groupby('series_id')['prediction']
        .transform(lambda group: group.rolling(prediction_rolling+1, center=True).median())
    )

    submission['pred_diff'] = submission['prediction'].diff()
    submission['event'] = submission['pred_diff'].replace({1: 1, -1: 0, 0: np.nan})
    
    sub_onset = submission[submission['event']==1].groupby(['series_id', 'year', 'month', 'day']).agg('first').reset_index()
    sub_wakeup = submission[submission['event']==0].groupby(['series_id', 'year', 'month', 'day']).agg('last').reset_index()

    result = pd.concat([sub_wakeup, sub_onset], ignore_index=True)[col_select].sort_values(['series_id', 'step'])
    
    return result

def detection_multi_prediction(
        submission: pd.DataFrame, y_pred: np.ndarray,
        prediction_rolling: int=720, score_rolling: int=12*60*5,
        col_select: List[str] = ['series_id','step','event','score']
    ) -> pd.DataFrame:
    
    submission['score'] = pd.Series([-1.] * submission.shape[0], dtype='float')
    submission['event'] = pd.Series([-1.] * submission.shape[0], dtype='int8')

    mask_0 = y_pred[:, 0] > 0.5
    mask_1 = y_pred[:, 1] > 0.5

    submission.loc[
        mask_0, 'event'
    ] = 0
    submission.loc[
        mask_1, 'event'
    ] = 1

    submission.loc[
        mask_0, 'score'
    ] = y_pred[mask_0, 0]
    
    submission.loc[
        mask_1, 'score'
    ] = y_pred[mask_1, 1]
    
    sub_onset = submission[submission['event']==1].groupby(['series_id', 'year', 'month', 'day']).agg('first').reset_index()
    sub_wakeup = submission[submission['event']==0].groupby(['series_id', 'year', 'month', 'day']).agg('last').reset_index()

    result = pd.concat([sub_wakeup, sub_onset], ignore_index=True)[col_select].sort_values(['series_id', 'step'])

    return result

# def detection_multi_prediction(
#         submission: pd.DataFrame, y_pred: np.array,
#         min_interval: int=720,
#         col_select: List[str] = ['series_id','step','event','score']
#     ) -> pd.DataFrame:
    
#     wakeup_onset_submission = []

#     submission['pred_0'] = y_pred[:, 0]
#     submission['pred_1'] = y_pred[:, 1]
    
#     submission['pred_max_0'] = (
#         submission.groupby('series_id')['pred_0']
#         .transform(
#             lambda group: 
#                 group.rolling(min_interval+1, center=True, min_periods=min_interval).max()
#         )
#     )
#     submission.loc[
#         (submission['pred_0'] != submission['pred_max_0']),
#         'pred_0'
#     ] = 0

#     submission['pred_max_1'] = (
#         submission.groupby('series_id')['pred_1']
#         .transform(
#             lambda group: 
#                 group.rolling(min_interval+1, center=True, min_periods=min_interval).max()
#         )
#     )
#     submission.loc[
#         (submission['pred_1'] != submission['pred_max_1']),
#         'pred_1'
#     ] = 0

#     for series_id in submission['series_id'].unique():
        
#         mask_series = submission['series_id'] == series_id
#         submission_series = submission.loc[mask_series].reset_index(drop=True)
        
#         pred_size = submission_series.shape[0]

#         days = int(pred_size/17280)
#         # scores0, scores1 = np.zeros(pred_size, dtype=np.float32), np.zeros(pred_size, dtype=np.float32)

#         # for index in range(pred_size):
#         #     low_interval = max(0,index-min_interval)
#         #     up_interval = index+min_interval
            
#         #     max_score_window_0 = max(y_pred_series[low_interval:up_interval, 0])
#         #     max_score_window_1 = max(y_pred_series[low_interval:up_interval, 1])

#         #     if y_pred_series[index, 0]==max_score_window_0:
#         #         scores0[index] = max_score_window_0
                
#         #     if y_pred_series[index, 1]==max_score_window_1:
#         #         scores1[index] = max_score_window_1
        
#         candidates_onset = np.argsort(submission_series['pred_1'])[-max(1,round(days)):]
#         candidates_wakeup = np.argsort(submission_series['pred_0'])[-max(1,round(days)):]
    
#         max_candidates = min(len(candidates_onset), len(candidates_wakeup))
#         sub_onset = submission_series.loc[candidates_onset[:max_candidates], ['series_id', 'step', 'pred_1']].copy()
#         sub_onset['event'] = 1
#         sub_onset = sub_onset.rename(columns={'pred_1': 'score'})

#         sub_wakeup = submission_series.loc[candidates_wakeup[:max_candidates], ['series_id', 'step', 'pred_0']].copy()
#         sub_wakeup['event'] = 0
#         sub_wakeup = sub_wakeup.rename(columns={'pred_0': 'score'})

#         wakeup_onset_submission.append(sub_onset)
#         wakeup_onset_submission.append(sub_wakeup)

#     result = pd.concat(wakeup_onset_submission, ignore_index=True)
#     result = result[col_select].sort_values(['series_id','step']).reset_index(drop=True)
    
#     result['score'] = result['score'].fillna(result['score'].mean())
    
#     return result