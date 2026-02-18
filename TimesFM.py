import pandas as pd

Case_stdy_data = pd.read_csv("./Case_study.csv")
series = Case_stdy_data["V82"].values

print(len(series))
