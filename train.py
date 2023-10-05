if __name__ == '__main__':
    import argparse

    from src.utils import import_config_dict, import_params
    from src.modeling.classification.lgb_model import run_lgb_experiment
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default='lgbm')
    parser.add_argument('--log', default=10)
    parser.add_argument('--skip_save', action='store_true')
    parser.add_argument('--dev', action='store_true')

    args = parser.parse_args()
        
    config=import_config_dict()
    params_model = import_params(model_name='params_lgb')
    
    if args.dev:
        print('Starting to debug')
        args.log=1
        params_model['n_round'] = 10

    run_lgb_experiment(
        experiment_name=args.name,config=config,
        params_model=params_model, feature_list=config['FEATURE_LIST'],
        log_evaluation=args.log, skip_save=args.skip_save
    )