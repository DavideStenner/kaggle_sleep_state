import os

from typing import Tuple, Dict, List

import numpy as np
import lightgbm as lgb
import polars as pl
import xgboost as xgb

from src.metric import official_metric
from src.modeling.classification.interval_pred import detection_prediction

class MetricUtils():
    def __init__(self) -> None:
        self.df_ = None
        self.tolerances_ = None
        self.events_ = None
        
    def update_df(self, df_):
        self.df_=df_
        
    def update_events(self, events_):
        self.events_=events_

    def update_tolerances(self, tolerances):
        self.tolerances_=tolerances
    
    def return_events(self):
        return self.events_
    
    def return_df(self):
        return self.df_.copy()
        
    def return_tolerances(self):
        return self.tolerances_

def competition_metric_lgb(
    init_metric: MetricUtils,
    y_pred: np.ndarray, eval_data: lgb.Dataset,
) -> Tuple[str, float, bool]:
    """
    Pearson correlation coefficient metric
    """
    # y_true = eval_data.get_label()

    solution_ = init_metric.return_events()
    submission_ = init_metric.return_df()
    tolerances_ = init_metric.return_tolerances()

    submission_ = detection_prediction(submission=submission_, y_pred=y_pred)

    score_ = official_metric.score(
        solution=solution_,
        submission=submission_,
        tolerances=tolerances_,
        series_id_column_name='series_id',
        time_column_name='step',
        event_column_name='event',
        score_column_name='score',
    )
    
    return 'event_detection_ap', score_, True

def competition_metric_xgb(
    init_metric: MetricUtils,
    y_pred: np.ndarray, eval_data: xgb.DMatrix,
) -> Tuple[str, float]:
    """
    Pearson correlation coefficient metric
    """
    solution_ = init_metric.return_events()
    submission_ = init_metric.return_df()
    tolerances_ = init_metric.return_tolerances()

    submission_ = detection_prediction(submission=submission_, y_pred=y_pred)

    score_ = official_metric.score(
        solution=solution_,
        submission=submission_,
        tolerances=tolerances_,
        series_id_column_name='series_id',
        time_column_name='step',
        event_column_name='event',
        score_column_name='score',
    )
    
    return 'event_detection_ap', score_

def pl_average_precision(group: pl.DataFrame, class_counts_: dict) -> pl.DataFrame:
    match_ = group['matched'].to_numpy()
    score_ = group['score'].to_numpy()
    
    event_ = group['event'].to_numpy()[0]
    tolerances_ = group['tolerances'].to_numpy()[0]
    
    count_ = class_counts_[event_]

    res = official_metric.average_precision_score(
            match_, score_, count_
        )

    return pl.DataFrame(
        {'event': [event_], 'tolerance': [tolerances_], 'score': [res]}
    )

def polars_new_score(
    solution_: pl.LazyFrame,
    submission_: pl.LazyFrame,
    tolerances_: Dict[str, List[float]]
) -> float:

    submission_ = submission_.with_columns(
        pl.col('event').cast(pl.Int64),
        pl.col('series_id').cast(pl.Int64)
    ).sort(['series_id', 'step'])
    
    solution_ = solution_.with_columns(
        pl.col('event').cast(pl.Int64),
        pl.col('series_id').cast(pl.Int64)
    ).sort(['series_id', 'step'])

    class_counts_ = {
        info_['event']: info_['count']
        for info_ in (
            solution_.group_by('event')
            .agg(pl.count())
            .collect().to_dicts()
        )
    }
    event_classes = solution_.select('event').unique().collect()
    
    #combination of event, tolerances and series_id
    aggregation_keys = pl.LazyFrame(
            data=[
                (ev, tol, vid)
                for ev in tolerances_.keys()
                for tol in tolerances_[ev]
                for vid in solution_.select(
                    pl.col('series_id')
                ).unique().collect().to_numpy().reshape((-1))
            ],
            schema=['event', 'tolerances', 'series_id']
    ).with_columns(
        pl.col('event').cast(pl.Int64),
        pl.col('series_id').cast(pl.Int64)
    )
    #each group is based on tolerance
    #add information on submission by event, series -> varies on tolerance
    aggregation_keys_submission = aggregation_keys.join(
        submission_, on=['event', 'series_id'], how='left'
    ).with_columns(pl.col('step').cast(pl.Int32)).filter(pl.col('step').is_not_null()).sort('step')

    aggregation_keys_solution = (
        aggregation_keys.join(
            solution_, on=['event', 'series_id'], how='left'
        )
        .with_columns(
            pl.lit(True).cast(pl.Boolean).alias('matched'),
            pl.col('step').alias('step_sol'),
            (pl.col('step')-pl.col('tolerances')-1).cast(pl.Int32).alias('step_low'),
        ).filter(pl.col('step').is_not_null())
        .select(['event', 'series_id', 'tolerances', 'step_sol', 'step_low', 'matched'])
        .sort('step_low')
    )

    detection_matched = aggregation_keys_submission.join_asof(
        aggregation_keys_solution,
        by=['event', 'series_id', 'tolerances'],
        strategy='backward',
        left_on='step',
        right_on='step_low',
    ).with_columns(
        #everything after second step is not matched
        pl.when(
            pl.col('step') < (pl.col('step_sol')+pl.col('tolerances'))
        ).then(pl.col('matched')).otherwise(False).alias('matched')
    ).collect()
    
    ap_table = (
        detection_matched
        .group_by(['event', 'tolerances'])
        .map_groups(
            lambda group: pl_average_precision(group, class_counts_)
        )
    )
    mean_ap = ap_table.group_by('event').agg(pl.col('score').mean()).select('score').sum().item() / len(event_classes)
    return mean_ap
