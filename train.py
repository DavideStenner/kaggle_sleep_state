if __name__ == '__main__':
    import argparse
    import warnings

    from src.utils import import_config_dict, import_params
    warnings.filterwarnings(action='error')

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='lgb', type=str)
    parser.add_argument('--name', default='main', type=str)
    parser.add_argument('--log', default=10, type=int)
    parser.add_argument('--skip_save', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--train', action='store_true')
    
    args = parser.parse_args()

    config=import_config_dict()
    config_model = import_params(model_name=f'params_{args.model}')

    params_model, metric_eval = config_model['params_model'], config_model['metric_eval']

    if args.model == 'lgb':
        from src.modeling.classification.lgb_model import run_lgb_experiment as run_experiment
        from src.modeling.classification.lgb_model import evaluate_lgb_score as evaluate_score

        params_model['metric'] = metric_eval
        
    elif args.model == 'xgb':
        from src.modeling.classification.xgb_model import run_xgb_experiment as run_experiment
        from src.modeling.classification.xgb_model import evaluate_xgb_score as evaluate_score

        params_model['eval_metric'] = metric_eval
        
    else:
        raise NotImplementedError
    
    experiment_name = args.model + '_' + args.name
    
    
    if args.dev:
        print('Starting to debug')
        args.log=1
        params_model['n_round'] = 10

    if args.train:
        print('Starting training normal')
        run_experiment(
            experiment_name=experiment_name,config=config,
            params_model=params_model, feature_list=config['FEATURE_LIST'],
            log_evaluation=args.log, skip_save=args.skip_save,
            dev=args.dev
        )
        
    evaluate_score(
        config=config, experiment_name=experiment_name, 
        params_model=params_model, feature_list=config['FEATURE_LIST'],
        add_comp_metric=True, metric_to_max = 'event_detection_ap'
    )