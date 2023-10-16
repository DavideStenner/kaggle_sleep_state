import polars as pl

def scan_train_parquet(path_file: str, dev: bool) -> pl.LazyFrame:
    train = pl.scan_parquet(path_file)
    
    if dev:
        selected_series_id = (
            train.select('series_id')
            .sort(by='series_id').unique()
            .head(3).collect().to_numpy().reshape((-1)).tolist()
        )
        
        train = train.filter(
            pl.col('series_id').is_in(selected_series_id)
        )
        train = (
            train.sort(['series_id', 'step']).group_by('series_id')
            .agg(pl.all().head(14400))
            .explode(pl.all().exclude("series_id"))
        )
        print(f'Using only: {train.select(pl.count()).collect().item()} rows')

    return train
