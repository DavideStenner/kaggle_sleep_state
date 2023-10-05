import os
import gc

import pickle
import polars as pl
import lightgbm as lgb


def run_lgb_experiment(
        experiment_name: str,
        config: dict, params_model: dict,
        feature_list: list, log_evaluation: int, skip_save: bool
    ) -> None:

    save_path = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        experiment_name
    )
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_list = []
    progress_list = []

    train = pl.scan_parquet(
        os.path.join(
            config['DATA_FOLDER'], config['PREPROCESS_FOLDER'], 
            'train.parquet'
        )
    )

    for fold_ in range(config['N_FOLD']):
        print(f'\n\nStarting fold {fold_}\n\n\n')
        
        progress = {}

        callbacks_list = [
            lgb.record_evaluation(progress),
            lgb.log_evaluation(
                period=log_evaluation, 
                show_stdv=False
            )
        ]
        print('Collecting dataset')
        train_filtered = train.filter(
            pl.col('fold') != fold_
        )
        test_filtered = train.filter(
            pl.col('fold') == fold_
        )
        train_matrix = lgb.Dataset(
            train_filtered.select(feature_list).collect().to_numpy().astype('float32'),
            train_filtered.select(config['TARGET_COL']).collect().to_numpy().reshape((-1)).astype('uint8')
        )
        
        test_matrix = lgb.Dataset(
            test_filtered.select(feature_list).collect().to_numpy().astype('float32'),
            test_filtered.select(config['TARGET_COL']).collect().to_numpy().reshape((-1)).astype('uint8')
        )
        
        print('Start training')
        model = lgb.train(
            params=params_model,
            train_set=train_matrix, 
            num_boost_round=params_model['n_round'],
            valid_sets=[test_matrix],
            valid_names=['valid'],
            callbacks=callbacks_list,
        )

        if ~skip_save:
            model.save_model(
                os.path.join(
                    save_path,
                    f'lgb_{fold_}.txt'
                )
            )

        model_list.append(model)
        progress_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()
        if ~skip_save:
            save_model(
                model_list=model_list, progress_list=progress_list,
                save_path=save_path
            )

def save_model(
        model_list: list, progress_list: list, save_path: str
    )->None:
        with open(
            os.path.join(
                save_path,
                'model_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)

        with open(
            os.path.join(
                save_path,
                'progress_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)
