import numpy as np
import pandas as pd

from typing import List

def detection_prediction(
        submission: pd.DataFrame, y_pred: np.array,
        prediction_rolling: int=720, score_rolling: int=12*60*5,
        col_select: List[str] = ['series_id','step','event','score']
    ) -> pd.DataFrame:
    
    if y_pred.shape[1] > 1:
        submission['probability'] = np.max(y_pred, axis=1)
        submission['prediction'] = np.argmax(y_pred, axis=1).astype(int)
    
    else:
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