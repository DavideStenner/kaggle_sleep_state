if __name__ == '__main__':
    import argparse

    from src.preprocess.pipeline import train_pipeline
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--dash_data', action='store_true')
    parser.add_argument('--filter_data', action='store_false')

    args = parser.parse_args()
        
    train_pipeline(filter_data=args.filter_data, dev=args.dev, dash_data=args.dash_data)