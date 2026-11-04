import pandas as pd

df = pd.read_parquet("colet.parquet")

print("SHAPE (rows, columns):", df.shape)
print("\nCOLUMN NAMES:")
print(df.columns.tolist())
print("\nFIRST 5 ROWS:")
print(df.head())
print("\nDATA TYPES:")
print(df.dtypes)