import os
import gc
import json
import time
import itertools

import numpy as np
import pandas as pd
import polars as pl

from typing import Tuple, List, Dict


from src.utils import import_config_dict

def downcast_timestamp(
        train_series: pl.LazyFrame,
        mapped_tz: dict={
            '-0400': 0,
            '-0500': 1
        }
    ) -> Tuple[pl.LazyFrame]:
    #https://www.kaggle.com/code/rimbax/lightgbm-feature-implementation-from-paper/notebook#Feature-Engineering
    signal_awake = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.208 * np.pi) ** 24))
    signal_onset = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.555 * np.pi) ** 24))

    #keep local time zone
    train_series = train_series.with_columns(
        pl.col('timestamp').str.replace(r".{5}$", "").alias('timestamp'),
        pl.col('timestamp').str.slice(-5).map_dict(mapped_tz).cast(pl.UInt8).alias('tz'),
    )
    #get feature
    transform_list = [
        (pl.col('timestamp').str.to_datetime().dt.year()-2000).cast(pl.UInt8).alias('year'), 
        pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month'),
        pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'), 
        pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour'),
        pl.col('timestamp').str.to_datetime().dt.minute().cast(pl.UInt8).alias('minute')
    ]
    transform_signal = [
        (pl.col('hour')*60 + pl.col('minute')).cast(pl.Int16).map_dict(signal_awake).cast(pl.Float32).alias('signal_awake'),
        (pl.col('hour')*60 + pl.col('minute')).cast(pl.Int16).map_dict(signal_onset).cast(pl.Float32).alias('signal_onset')
    ]
    train_series = train_series.with_columns(
        transform_list
    ).drop('timestamp').with_columns(
        transform_signal
    )
    
    return train_series

def correct_train_events(train_events: pl.LazyFrame) -> pl.LazyFrame:
    set_null_on_this_series_event = train_events.group_by(
        ['series_id', 'night']
    ).agg(
        pl.col('step').is_null().sum().alias('number_null')
    ).filter(
        (pl.col('number_null')!=0) &
        (pl.col('number_null')!=2)
    ).select(['series_id', 'night']).unique().collect().to_dicts()
    
    if len(set_null_on_this_series_event) > 0:
        number_rows_to_correct = train_events.filter(
            (pl.struct(['series_id', 'night']).is_in(set_null_on_this_series_event)) &
            (pl.col('step').is_not_null())
        ).select(pl.count()).collect().item()
        print(f'Correcting {number_rows_to_correct} train events rows')

        train_events = train_events.with_columns(
            (
                pl.when(
                    pl.struct(["series_id", "night"]).is_in(set_null_on_this_series_event)
                ).then(None).otherwise(pl.col(col)).alias(col)
                for col in ['step', 'timestamp']
            )
        )
    return train_events

def import_dataset(config: dict, dev: bool) -> Tuple[pl.LazyFrame]:
    
    path_original_data = os.path.join(
        config['DATA_FOLDER'],
        config['ORIGINAL_FOLDER'],
        config['ORIGINAL_DATA_PATH']
    )
    print('Starting lazy evaluation')    
    train_series = pl.scan_parquet(
        os.path.join(
            path_original_data,
            'train_series.parquet'
        )
    )
    train_events = pl.scan_csv(
        os.path.join(
            path_original_data,
            'train_events.csv'
        )
    )

    if dev:
        print('Sampling for dev purpose')
        sample_id = train_series.select('series_id').unique().head(10).collect()['series_id']
    
        train_series = train_series.filter(pl.col('series_id').is_in(sample_id))
        train_events = train_events.filter(pl.col('series_id').is_in(sample_id))
    
    #correct train_events -> make null an event if inside the same night the other event is not null
    train_events = correct_train_events(train_events=train_events)
    
    return train_series, train_events

def downcast_series(
        config: dict, 
        train_series: pl.LazyFrame, train_events: pl.LazyFrame,
    ) -> Tuple[pl.LazyFrame]:
    
    path_mapping = os.path.join(
        config['DATA_FOLDER'],
        config['MAPPING_PATH']
    )
    unique_train_series = train_series.select(
        'series_id'
    ).unique().collect().sort(by='series_id')['series_id'].to_numpy()
    
    id_mapping_train = {
        id_str: id_int
        for id_int, id_str in enumerate(unique_train_series) 
    }
    with open(
        os.path.join(path_mapping, 'mapping_series_id_train.json'), 'w'
    ) as file:
        json.dump(id_mapping_train, file)
            
    train_series = train_series.with_columns(
        pl.col('series_id').map_dict(id_mapping_train).cast(pl.UInt16)
    )
    
    train_events = train_events.with_columns(
        pl.col('series_id').map_dict(id_mapping_train).cast(pl.UInt16)
    )
    
    return train_series, train_events

def filter_series(
        train_series: pl.LazyFrame,
        train_events: pl.LazyFrame  
    ) -> Tuple[pl.LazyFrame]:
    
    train_series = train_series.filter(~pl.col('series_id').is_in(['31011ade7c0a', 'a596ad0b82aa']))
    train_events = train_events.filter(~pl.col('series_id').is_in(['31011ade7c0a', 'a596ad0b82aa']))

    return train_series, train_events

def filter_target(
        train: pl.LazyFrame
    ) -> pl.LazyFrame:
    #filter missing values on event
    train = train.filter(pl.col('event').is_not_null())
    return train

def downcast_all(
    config: dict, 
    train_series: pl.LazyFrame, train_events: pl.LazyFrame,
) -> Tuple[pl.LazyFrame]:
    
    train_series, train_events = downcast_series(
        config=config, 
        train_series=train_series, train_events=train_events,
    )    
    train_series = downcast_timestamp(train_series=train_series)

    print('Running check on train_events')
    assert train_events.group_by(['series_id', 'night']).agg(
        pl.count()
    ).select('count').mean().collect().item() == 2.
    
    assert train_events.group_by(['series_id', 'night']).agg(
        pl.all().sort_by('step').first()
    ).filter(pl.col('step').is_not_null()).select('event').unique().collect().item() == 'onset'

    assert train_events.group_by(['series_id', 'night']).agg(
        pl.all().sort_by('step').last()
    ).filter(pl.col('step').is_not_null()).select('event').unique().collect().item() == 'wakeup'

    train_events = train_events.with_columns(
        pl.col('event').map_dict(
            {
                'wakeup': 0,
                'onset': 1
            }
        ).cast(pl.UInt8),
        pl.col('step').cast(pl.UInt32),
    ).select(
        ['series_id', 'event', 'step']
    )

    return train_series, train_events

def gaussian_window_coefficient(window: int, std: float, normalize: bool=True) -> Dict[int, float]:
    # guassian distribution function with 0 mean and std deviation
    normalize_coef = (
        np.exp(-float(0)**2/(2*std**2)) / (std * np.sqrt(2*np.pi)) if normalize
        else 1
    )
    return {
        x: (np.exp(-float(x)**2/(2*std**2)) / (std * np.sqrt(2*np.pi))) / normalize_coef
        for x in range(-int(window/2), int(window/2)+1 )
    }

def add_gaussian_target(
        train_series: pl.LazyFrame, train_events: pl.LazyFrame, 
        gaussian_window: int=720, gaussian_coef: float=0.15
    ) -> pl.LazyFrame:
    
    gaussian_dict = gaussian_window_coefficient(window=gaussian_window, std=gaussian_window*gaussian_coef)
    
    print('Creating gaussian and multi target')
    start_rows = train_series.select(pl.count()).collect().item()

    train = train_series.join(
        train_events,
        on=['series_id', 'step'],
        how='left'
    ).with_columns(
        pl.col('prev_step').fill_null(strategy='backward').fill_null(value=-1).alias('back_prev_step'),
        pl.col('prev_step').fill_null(strategy='forward').fill_null(value=-1).alias('forw_prev_step'),
        
        pl.col('prev_event').fill_null(strategy='backward').fill_null(value=-1).alias('back_prev_event'),
        pl.col('prev_event').fill_null(strategy='forward').fill_null(value=-1).alias('forw_prev_event'),
    ).with_columns(
        (
            pl.when(
                (pl.col('step')-pl.col('back_prev_step')).abs() < (pl.col('step')-pl.col('forw_prev_step')).abs()
            ).then(pl.col('back_prev_step'))
            .otherwise(pl.col('forw_prev_step'))
         ).alias('nearest_step'),
        (
            pl.when(
                (pl.col('step')-pl.col('back_prev_step')).abs() < (pl.col('step')-pl.col('forw_prev_step')).abs()
            ).then(pl.col('back_prev_event'))
            .otherwise(pl.col('forw_prev_event'))
         ).alias('nearest_event')
    ).with_columns(
         (pl.col('step')-pl.col('nearest_step')).cast(pl.Int64).alias('nearest_distance')
    ).with_columns(
        #wakeup
        (
            (
                pl.when(
                    (pl.col('nearest_event')==0) &
                    (pl.col('nearest_distance')<=gaussian_window)
                ).then(0)
                .when(
                    (pl.col('nearest_event')==1) &
                    (pl.col('nearest_distance')<=gaussian_window)
                ).then(1).otherwise(2)
            )
            .fill_null(value=2).cast(pl.UInt8).alias('event_window')
        ),
        #gaussian wakeup
        (
            (
                pl.when(pl.col('nearest_event')==0).then(
                    pl.col('nearest_distance').map_dict(gaussian_dict, return_dtype=pl.Float32)
                ).otherwise(0.)
            )
            .fill_null(value=0).alias('gaussian_wakeup_event')
        ),
        #onset
        (
            (
                pl.when(pl.col('nearest_event')==1).then(
                    pl.col('nearest_distance').map_dict(gaussian_dict, return_dtype=pl.Float32)
                ).otherwise(0.)
            )
            .fill_null(value=0).alias('gaussian_onset_event')
        )        
    )
    #ensure no duplication
    end_rows = train.select(pl.count()).collect().item()
    assert start_rows == end_rows

    train = train.drop(
        [
            'back_prev_step', 'forw_prev_step', 'nearest_step', 'prev_step',
            'back_prev_event', 'forw_prev_event', 'nearest_event', 'nearest_distance', 'prev_event'
        ]
    )
    
    return train

def correct_gaussian_events(train_events: pl.LazyFrame) -> pl.LazyFrame:
    #get range of usable step from train events
    train_events = (
        train_events.with_columns(
            pl.col('step'),
            pl.col('step').shift(-1).over('series_id').alias('second_step')
        ).filter(pl.all_horizontal('step', 'second_step').is_not_null())
        .with_columns(
            pl.col('step').alias('prev_step'),
            pl.col('event').alias('prev_event')
        )
        .select(
            ['series_id', 'step', 'prev_step', 'prev_event']
        )
    )
    return train_events

def sleep_gaussian_target(train_series: pl.LazyFrame, train_events: pl.LazyFrame) -> Tuple[pl.LazyFrame]:
    train_events = correct_gaussian_events(train_events=train_events)
    train = add_gaussian_target(train_series=train_series, train_events=train_events)
    return train, train_events

def sleep_interval_target(train_series: pl.LazyFrame, train_events: pl.LazyFrame) -> Tuple[pl.LazyFrame]:
    train_events = correct_events(train_events=train_events)
    train = add_target(train_series=train_series, train_events=train_events)
    return train, train_events

def correct_events(train_events: pl.LazyFrame) -> pl.LazyFrame:
    #get range of usable step from train events
    train_events = (
        train_events.with_columns(
            pl.col('step').alias('first_step'),
            pl.col('step').shift(-1).over('series_id').alias('second_step')
        ).filter(pl.all_horizontal('first_step', 'second_step').is_not_null())
        .with_columns(
            pl.arange(0, pl.count()).over('series_id').alias('number_event')
        )
        .select(
            ['series_id', 'event', 'first_step', 'second_step', 'number_event']
        )
    )
    return train_events

def add_target(train_series: pl.LazyFrame, train_events: pl.LazyFrame) -> pl.LazyFrame:
    print('Creating interval target')
    start_rows = train_series.select(pl.count()).collect().item()

    #add first step, second step and do a asof_join to filter step>=first_step
    train = train_series.sort('step').join_asof(
        train_events.sort('first_step'),
        left_on='step',
        right_on='first_step',
        by='series_id',
        strategy='backward',
    ).with_columns(
        #everything after second step is not correct -> set to None
        (
            pl.when(
                pl.col('step') <= pl.col('second_step')
            ).then(pl.col(col_name)).otherwise(None).alias(col_name)
            for col_name in ['event', 'first_step', 'second_step']
        )
    ).sort(['series_id', 'step'])
    
    #ensure no duplication
    end_rows = train.select(pl.count()).collect().item()
    assert start_rows == end_rows

    #ensure correcteness of not null rows
    correcteness_not_null = train.drop_nulls().with_columns(
        ((pl.col('step') <= pl.col('second_step')) &
        (pl.col('step')>=pl.col('first_step'))
        ).alias('check')
    ).collect()['check'].mean()

    assert correcteness_not_null == 1.
    train = train.drop(['first_step', 'second_step'])
    
    return train

def add_shift(train: pl.LazyFrame) -> pl.LazyFrame:
    train = train.with_columns(
        (pl.col('anglez') - pl.col('anglez').shift(1).over('series_id')).alias('diff_anglez').cast(pl.Float32),
        (pl.col('enmo') - pl.col('enmo').shift(1).over('series_id')).alias('diff_enmo').cast(pl.Float32)
    )
    
    return train

def lift(train: pl.LazyFrame, center: bool, suffix: str) -> pl.LazyFrame:
    #TODO
    #ADD ROLLING AGG BOTTOM TO CALCULATE RIGHT ROLLING
        
    train = train.with_columns(
        pl.col('step').cast(pl.Int32),
        (
            pl.max_horizontal(
                pl.col('enmo') -0.2, pl.lit(0)
            ).rolling_sum(
                window_size='120i', center=center,
                closed='left', min_periods=120
            ).over('series_id').alias('activity_count' + suffix).cast(pl.Float32)
        )
    ).with_columns(
        (100/(1+pl.col('activity_count' + suffix))).rolling_mean(
            window_size='360i', center=center,
            closed='left', min_periods=360
        ).over('series_id').alias('lids' + suffix).cast(pl.Float32)
    )
        
    return train

def add_lift(train: pl.LazyFrame) -> pl.LazyFrame:
    # https://www.nature.com/articles/s41598-020-79217-x.pdf

    train = lift(train, center=False, suffix='_left')
    train = lift(train, center=True, suffix='_center')
        
    return train

def add_rolling_feature(
        train: pl.LazyFrame, 
        period_list: List[int] = [15, 30, 60, 180], 
        col_list: List[str]=['enmo', 'anglez']
    ) -> pl.LazyFrame:
    
    rolling_param = {
        'center': True, 'closed': 'left'
    }
    list_rolling_operation = []
    num_rolling_operation = 0
    
    col_period_product = list(itertools.product(col_list, period_list))

    for col, period in col_period_product:
        period_step = period * 12
        
        rolling_param.update(
            {
                'window_size': f'{period_step}i', 
                'min_periods': period_step
            }
        )
        current_operation = [
            pl.col(col).rolling_mean(**rolling_param)
                .over('series_id').alias(f'{col}_{period_step}_mean').cast(pl.Float32),
            
            pl.col(col).rolling_std(**rolling_param)
                .over('series_id').alias(f'{col}_{period_step}_std').cast(pl.Float32),
            
            pl.col(col).rolling_min(**rolling_param)
                .over('series_id').alias(f'{col}_{period_step}_min').cast(pl.Float32),
            
            pl.col(col).rolling_max(**rolling_param)
                .over('series_id').alias(f'{col}_{period_step}_max').cast(pl.Float32),
            
            pl.col(col).rolling_median(**rolling_param)
                .over('series_id').alias(f'{col}_{period_step}_median').cast(pl.Float32),
        ]
        list_rolling_operation += current_operation
        num_rolling_operation += len(current_operation)
        
    train = train.with_columns(list_rolling_operation)
    
    #adding mad feature
    for col, period in col_period_product:
        period_step = period * 12
        
        num_rolling_operation += 2
        rolling_param.update(
            {
                'window_size': f'{period_step}i', 
                'min_periods': period_step
            }
        )

        train = train.with_columns(
                (pl.col(col)-pl.col(f'{col}_{period_step}_median')).abs()
                    .rolling_median(**rolling_param)
                    .over('series_id').alias(f'{col}_{period_step}_mad').cast(pl.Float32),
        )
        train = train.with_columns(
                (pl.col(f'{col}_{period_step}_max')-pl.col(f'{col}_{period_step}_min'))
                .alias(f'{col}_amplit_{period_step}').cast(pl.Float32)
        )
    print(f'Using {num_rolling_operation} rolling feature')

    return train
    
    
def add_feature(train: pl.LazyFrame) -> pl.LazyFrame:
    train = add_shift(train)
    
    train = add_lift(train)
    
    train = add_rolling_feature(train)
    
    return train

def add_cv_folder(train: pl.LazyFrame, train_events: pl.LazyFrame) -> Tuple[pl.LazyFrame]:
    """
    Standard Cross Validation on series id

    Args:
        train (pl.LazyFrame): 
        train_events (pl.LazyFrame)

    Returns:
        Tuple[pl.LazyFrame]: 
    """
    series_id_fold = train.group_by(
        ['series_id']
    ).agg(pl.count().alias('count')).sort(['series_id']).collect()

    shape_ = series_id_fold['count'].sum()
    
    series_id_fold = series_id_fold.sample(fraction=1., shuffle=True).select(
        pl.col('series_id'),
        ((pl.cumsum("count")/shape_)).alias('pct_rate')
    ).with_columns(
        pl.when(
            pl.col('pct_rate')<.2
        ).then(0).when(
            pl.col('pct_rate')<.4
        ).then(1).when(
            pl.col('pct_rate')<.6
        ).then(2).when(
            pl.col('pct_rate')<.8
        ).then(3).otherwise(4).alias('fold').cast(pl.UInt8)
    ).select(['series_id', 'fold'])
    
    train = train.join(
        series_id_fold.lazy(),
        on='series_id', how='left'
    )
    train_events = train_events.join(
        series_id_fold.lazy(),
        on='series_id', how='left'
    )
    return train, train_events

def train_pipeline(
        file_name: str, 
        dev: bool=False, 
        dash_data: bool=False, save_event: bool=False
    ) -> None:
    
    #import dataset
    config=import_config_dict()
    
    train_series, train_events = import_dataset(config=config, dev=dev)
    train_series, train_events = filter_series(train_series=train_series, train_events=train_events)
    
    train_series, train_events = downcast_all(
        config=config, train_series=train_series, train_events=train_events,
    )
    
    train_series, _ = sleep_gaussian_target(train_series=train_series, train_events=train_events)
    train, train_events = sleep_interval_target(train_series=train_series, train_events=train_events)
    
    if dash_data:
        print('Saving csv for dashboard')
        train.collect().write_csv(
            os.path.join(
                config['DATA_FOLDER'],
                config['DASHBOARD_FOLDER'],
                'train.csv'
            )  
        )
        _ = gc.collect()
    
    train = add_feature(train)
    
    train = filter_target(train)
    
    train, train_events = add_cv_folder(train=train, train_events=train_events)
    
    #save in correct original format, without na
    if save_event:
        print('Saving train_events')
        (
            train_events.select(
                ['series_id', 'event', 'first_step', 'fold']
            )
            .rename({'first_step': 'step'}).collect()
        ).write_parquet(
            os.path.join(
                config['DATA_FOLDER'],
                config['PREPROCESS_FOLDER'],
                'train_events.parquet'
            )
        )

    print('Starting to collect data')
    train = train.collect()

    print('Saving parquet')
    train.write_parquet(
        os.path.join(
            config['DATA_FOLDER'],
            config['PREPROCESS_FOLDER'],
            file_name
        )
    )