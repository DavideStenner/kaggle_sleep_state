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

    transform_list = [
        pl.col('timestamp').str.slice(-5).map_dict(mapped_tz).alias('tz'),
        (pl.col('timestamp').str.to_datetime().dt.year()-2000).cast(pl.UInt8).alias('year'), 
        pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month'),
        pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'), 
        pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour'),
        pl.col('timestamp').str.to_datetime().dt.second().cast(pl.UInt8).alias('second')
    ]
    
    train_series = train_series.with_columns(transform_list).drop('timestamp')
    
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
    train_series = train_series.with_columns(
        pl.col('timestamp').str.to_datetime()
    )

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

    # train_series = downcast_timestamp(train_series=train_series)
    
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

def train_pipeline(filter_intersection_id: bool=True, dev: bool=False, dash_data: bool=False) -> None:
    
    #import dataset
    config=import_config_dict()
    
    train_series, train_events = import_dataset(config=config, dev=dev)
    
    if filter_intersection_id:
        #filter from intersection id
        train_series, train_events = filter_train_set(
            train_series=train_series, train_events=train_events,
        )

    train_series, train_events = downcast_all(
        config=config, train_series=train_series, train_events=train_events,
    )
    train = add_target(train_series=train_series, train_events=train_events)
    
    train = add_shift(train)
    
    print('Starting to collect data')
    train_series = train_series.collect()

    print('Saving parquet')
    train_series.write_parquet(
        os.path.join(
            config['DATA_FOLDER'],
            config['PREPROCESS_FOLDER'],
            'train_series.parquet'
        )
    )
    if dash_data:
        print('Saving csv for dashboard')
        train_series.write_csv(
        os.path.join(
                config['DATA_FOLDER'],
                config['PREPROCESS_FOLDER'],
                config['DASHBOARD_FOLDER'],
                'train_series.csv'
            )  
        )