import os

from typing import Tuple, Dict, List

import numpy as np
import lightgbm as lgb
import polars as pl

from src.metric import official_metric

class MetricUtils():
    def __init__(self, sampling: int) -> None:
        self.df_ = None
        self.tolerances_ = None
        self.sampling_ = sampling
    
    def update_df(self, df_):
        self.df_=df_
        
    def update_tolerances(self, tolerances):
        self.tolerances_=tolerances
    
    def return_sampling(self):
        return self.sampling_
    
    def return_df(self):
        return self.df_.copy()
        
    def return_tolerances(self):
        return self.tolerances_

def competition_metric_lgb(
    init_metric: MetricUtils,
    y_pred: np.array, eval_data: lgb.Dataset,
    new_score: bool = False
) -> Tuple[str, float, bool]:
    """
    Pearson correlation coefficient metric
    """
    y_true = eval_data.get_label()

    solution_ = init_metric.return_df()
    submission_ = init_metric.return_df()
    tolerances_ = init_metric.return_tolerances()
    sampling_ = init_metric.return_sampling()
    
    assert (submission_['series_id']==solution_['series_id']).mean() == 1.
    assert (submission_['step']==solution_['step']).mean() == 1.

    submission_['event'] = (y_pred>0.5).astype(int)
    submission_['score'] = y_pred

    solution_['event'] = (y_true).astype(int)
    sampled_index = solution_.groupby(['series_id', 'number_event']).sample(sampling_).index

    solution_ = solution_.loc[sampled_index, ['series_id', 'step', 'event']]
    submission_ = submission_.loc[sampled_index, ['series_id', 'step', 'event', 'score']]

    if new_score:
        score_ = polars_new_score(
            solution_=pl.LazyFrame(solution_),
            submission_=pl.LazyFrame(submission_),
            tolerances_=tolerances_
        )
    else:
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
    #NEW WORKING
    # aggregation_keys_submission = aggregation_keys.join(
    #     submission_, on=['event', 'series_id'], how='left'
    # ).filter(pl.col('step').is_not_null())

    # aggregation_keys_solution = (
    #     aggregation_keys.join(
    #         solution_, on=['event', 'series_id'], how='left'
    #     ).filter(pl.col('step').is_not_null())
    #     .select(['event', 'series_id', 'tolerances', 'step'])
    # )

    # matched_first_logic = aggregation_keys_submission.join(
    #     aggregation_keys_solution, 
    #     left_on=['event', 'series_id', 'tolerances'],
    #     right_on=['event', 'series_id', 'tolerances'],
    #     how='cross', suffix='_sol'
    # ).select(
    #     [
    #         'event', 'series_id', 'tolerances', 
    #         'score', 'step', 'step_sol'
    #     ]
    # ).filter(
    #     (pl.col('step')-pl.col('step_sol')).abs()<pl.col('tolerances')
    # ).sort('score', descending=True).unique(
    #     [
    #         'event', 'series_id', 'tolerances', 'step'
    #     ], keep='first'
    # ).with_columns(
    #     pl.lit(True).cast(pl.Boolean).alias('matched')
    # )


    # detection_matched = aggregation_keys_submission.join(
    #     matched_first_logic, 
    #     on = ['event', 'series_id', 'tolerances', 'step'],
    #     how='left'
    # ).select(
    #     ['event', 'series_id', 'tolerances', 'step', 'matched']
    # ).with_columns(pl.col('matched').fill_null(False)).collect()

    #LAST WORKING
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
