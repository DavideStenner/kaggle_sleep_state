if __name__ == '__main__':
    import argparse

    from src.utils import import_config_dict, import_params
    from src.modeling.classification.lgb_model import run_lgb_experiment, evaluate_lgb_score, run_missing_lgb_experiment
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default='lgbm')
    parser.add_argument('--log', default=10)
    parser.add_argument('--skip_save', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--sampling', default=30)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--missing', action='store_true')
    
    args = parser.parse_args()
        
    config=import_config_dict()
    config_model = import_params(model_name='params_lgb')
    
    params_model, metric_eval = config_model['params_model'], config_model['metric_eval']
    params_model['metric'] = metric_eval
    
    if args.dev:
        print('Starting to debug')
        args.log=1
        params_model['n_round'] = 10

    if args.train:
        print('Starting training normal')
        run_lgb_experiment(
            experiment_name=args.name,config=config,
            params_model=params_model, feature_list=config['FEATURE_LIST'],
            log_evaluation=args.log, skip_save=args.skip_save,
            dev=args.dev, sampling_=args.sampling
        )
        
    evaluate_lgb_score(
        config=config, experiment_name=args.name, 
        params_model=params_model, feature_list=config['FEATURE_LIST'],
        add_comp_metric=True, metric_to_max = 'event_detection_ap'
    )
    
    if args.missing:
        missing_experiment_name = args.name + '_na'
        print('Starting training missing')
        
        if args.train:

            run_missing_lgb_experiment(
                experiment_name=missing_experiment_name,config=config,
                params_model=params_model, feature_list=config['FEATURE_LIST'],
                log_evaluation=args.log, dev=args.dev, skip_save=args.skip_save
            )
        
        evaluate_lgb_score(
            config=config, experiment_name=missing_experiment_name, 
            params_model=params_model, feature_list=config['FEATURE_LIST'],
            add_comp_metric=False, metric_to_max = 'auc'
        )