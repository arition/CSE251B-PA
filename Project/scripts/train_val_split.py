import pandas

csv = pandas.read_csv('data/train.csv')

val_csv_index = list(range(0, len(csv), 10))
train_csv_index = [i for i in range(0, len(csv)) if i not in val_csv_index]

train_csv = csv.iloc[train_csv_index]
val_csv = csv.iloc[val_csv_index]

train_csv.to_csv('data/train_split.csv', index=False)
val_csv.to_csv('data/val_split.csv', index=False)
