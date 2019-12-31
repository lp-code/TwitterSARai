# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd

from data_utils import split_into_tags_and_doc


_project_dir = Path(__file__).resolve().parents[2]
_data_dir = _project_dir / "data"


@click.command()
@click.argument(
    'input_filepath',
    default=_data_dir / "raw" / "vicinitas_user_tweets_vest_scoring_layout.xlsx",
    type=click.Path(exists=True)
)
@click.argument(
    'output_filepath',
    default=_data_dir.joinpath("processed", "tweets_processed.csv"),
    type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    df_xlsx = pd.read_excel(input_filepath,
                            sheet_name="tweets",
                            header=0,
                            usecols=["i_man", "i_alg", "Text"],
                            dtype={"Text": str, "i_man": np.float32})  # float to accomodate nan
    df_xlsx.dropna(inplace=True)  # in practice: drop rows without an i_man label
    # In the ML filter, we consider only tweets with nonzero score from the C# part!!!
    df_xlsx.drop(df_xlsx[df_xlsx["i_alg"] == 0].index, inplace=True)
    df_xlsx = df_xlsx.astype({"i_man": np.int8})
    df_xlsx["tags"], df_xlsx["doc"] = zip(*df_xlsx["Text"].map(split_into_tags_and_doc))
    df_xlsx.drop(columns="Text", inplace=True)

    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)
    df_xlsx.to_csv(output_filepath, index=False)

    logger.info('Finished making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
