"""
Simple conversion from the csv format exported from table storage (through
storage explorer) to an Excel file with the right columns to copy into the
main data file.

Enter the working directory and file names below.
"""

import pandas as pd
from pathlib import Path

working_dir = Path("c:\\Users\\lpesch\\Downloads")
input_file = "TweetTable2.csv"
output_file = "tweets2x.xlsx"

df = pd.read_csv(working_dir / input_file)

df["RowKey"] = df["RowKey"].astype(str)
df["empty"] = ""

df.to_excel(
    working_dir / output_file,
    index=False,
    columns=["empty", "Score", "RowKey", "FullText", "ScoreML", "empty", "CreatedAt"]
)

