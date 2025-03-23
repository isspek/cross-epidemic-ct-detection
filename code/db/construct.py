'''
Script for constructing datasets from Hoax and Hidden Agenda

conspiracy_or_not
0: does not contain a conspiracy theory 1: does contain a conspiracy theory

stance towards conspiracy theory:
0: does not contain a conspiracy theory 1: against the conspiracy theory 2: neutral towards the conspiracy theory 3: supportive towards the conspiracy theory

topic:
0: climate change 1: COVID origins 2: COVID vaccine 3: Epstein

'''

from pathlib import Path
import pandas as pd

input_folder = "data/covid/hoax_hidden"
output_folder = "data/collection/eval_sup_datasets/gold/non_epidemic"

data = pd.read_csv(Path(input_folder) / "data.csv", sep=",")
labels = pd.read_csv(Path(input_folder) / "labels.csv", sep=",")

labels.rename(columns={"tweet_ID": "id"}, inplace=True)

data = data.merge(labels, how='inner', on='id')

climate_consp = data[(data["topic"] == 0)]
print(f"Number of conspiracies about climate change: {len(climate_consp)}")


def assign_label(conspiracy_or_not, stance_towards_conspiracy):
    if conspiracy_or_not == 1 and (stance_towards_conspiracy == 2 or stance_towards_conspiracy == 3):
        return 1
    else:
        return 0


climate_consp['label'] = climate_consp.apply(
    lambda x: assign_label(x['conspiracy_or_not'], x['stance_towards_conspiracy']), axis=1)

# climate conspiracies 228 not conspiracy, 310 conspiracy
print(climate_consp.groupby(["label"])["text"].count())

epstein = data[data["topic"] == 3]
print(f"Number of conspiracies about epstein: {len(epstein)}")

epstein['label'] = epstein.apply(
    lambda x: assign_label(x['conspiracy_or_not'], x['stance_towards_conspiracy']), axis=1)

print(epstein.groupby(["label"])["text"].count())

full_data = pd.concat([climate_consp, epstein])

print("Full data stats")
print(full_data.groupby(["label"])["text"].count())

full_data.rename(columns={"id": "id_str"}, inplace=True)

full_data = full_data.sample(frac=1, random_state=0).reset_index(drop=True)
full_data = full_data[["id_str", "text", "label"]]

full_data.to_csv(Path(output_folder) / "data.tsv", sep="\t", index=False)