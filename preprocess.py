if __name__ == '__main__':
    import argparse

    from src.preprocess.pipeline import train_pipeline
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--dash_data', action='store_true')

    args = parser.parse_args()
    
    #normal train file
    train_pipeline(file_name='train.parquet', filter_data=True, dev=args.dev, dash_data=args.dash_data)
    
    #missing values file
    train_pipeline(file_name='train_missing.parquet', filter_data=False, dev=args.dev, dash_data=False)