if __name__ == '__main__':
    import os
    import argparse

    from src.utils import import_config_dict
    from src.preprocess.pipeline import train_pipeline
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()
    
    config=import_config_dict()
    
    train_series = train_pipeline(dev=args.dev)
    
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
    print('Saving csv for dashboard')
    train_series.write_csv(
      os.path.join(
            config['DATA_FOLDER'],
            config['PREPROCESS_FOLDER'],
            config['DASHBOARD_FOLDER'],
            'train_series.csv'
        )  
    )