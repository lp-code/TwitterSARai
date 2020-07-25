Getting new tweets
------------------

All tweets retrieved by AzTwitterSAR are logged to a table in an Azure
storage account. The easiest way to get them from there into the xls-file
in the `data/raw` directory is to:
1. In MS Storage Explorer, export from the table to a csv-file and save the
   file locally.
1. Activate the venv.
1. Adapt the input file path in `data/external/convert_az_table_to_xls.py`
   and run the script.
1. Open the resulting xls file and copy the new tweets over into
   `data/raw/vicinitas_user_tweets_vest_scoring_layout.xlsx`.
