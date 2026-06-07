import pandas as pd

df = pd.read_csv("data/sensor_data.csv")

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA INFO ===")
print(df.info())

print("\n=== STATISTICS ===")
print(df.describe())

print("\n=== CLASS DISTRIBUTION ===")
print(df["failure"].value_counts())