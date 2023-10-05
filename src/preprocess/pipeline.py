import os
import gc
import json
import time

import numpy as np
import pandas as pd
import polars as pl

from typing import Tuple

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
        pl.col('timestamp').str.slice(-5).map_dict(mapped_tz).alias('tz'),
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

def filter_train_set(
        train_series: pl.LazyFrame, train_events: pl.LazyFrame, 
    ) -> pl.LazyFrame:
    # print('Filtering intersection series_id from train')
    
    #ADD FILTER LOGIC
    return train_series, train_events

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

def add_target(train_series: pl.LazyFrame, train_events: pl.LazyFrame) -> pl.LazyFrame:
    #get range of usable step from train events
    train_events = train_events.with_columns(
        pl.col('step').alias('first_step'),
        pl.col('step').shift(-1).over('series_id').alias('second_step')
    ).filter(pl.all_horizontal('first_step', 'second_step').is_not_null()).select(
        ['series_id', 'event', 'first_step', 'second_step']
    )
    print('Running join assert')
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

def add_lift(train: pl.LazyFrame) -> pl.LazyFrame:
    # https://www.nature.com/articles/s41598-020-79217-x.pdf
    train = train.with_columns(
        pl.col('step').cast(pl.Int32),
        (
            pl.max_horizontal(
                pl.col('enmo') -0.2, pl.lit(0)
            ).rolling_sum(
                window_size='120i', center=False,
                closed='left', min_periods=120
            ).over('series_id').alias('activity_count').cast(pl.Float32)
        )
    ).with_columns(
        (100/(1+pl.col('activity_count'))).rolling_mean(
            window_size='360i', center=False,
            closed='left', min_periods=360
        ).over('series_id').alias('lids').cast(pl.Float32)
    )
    return train

def add_feature(train: pl.LazyFrame) -> pl.LazyFrame:
    train = add_shift(train)
    
    train = add_lift(train)
    return train

def add_cv_folder(train: pl.LazyFrame) -> pl.LazyFrame:
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
        ).then(3).otherwise(4).alias('fold')
    ).select(['series_id', 'fold'])
    
    train = train.join(
        series_id_fold.lazy(),
        on='series_id', how='left'
    )
    return train

def train_pipeline(filter_target_na: bool=True, dev: bool=False, dash_data: bool=False) -> None:
    
    #import dataset
    config=import_config_dict()
    
    train_series, train_events = import_dataset(config=config, dev=dev)
    
    train_series, train_events = downcast_all(
        config=config, train_series=train_series, train_events=train_events,
    )
    train = add_target(train_series=train_series, train_events=train_events)
        
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
    
    if filter_target_na:
        train = filter_train(train)

    train = add_cv_folder(train)

    print('Starting to collect data')
    train = train.collect()

    print('Saving parquet')
    train.write_parquet(
        os.path.join(
            config['DATA_FOLDER'],
            config['PREPROCESS_FOLDER'],
            'train.parquet'
        )
    )