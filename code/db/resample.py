import pandas as pd
from pathlib import Path
from datetime import datetime

def resample_for_annotations():
    random_seed = 0
    num_samples = 500
    diseases = ['monkeypox']

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-Y-%H:%M:%S")
    # for disease in diseases:
    #     ref_data = pd.read_csv(f'data/claims/{disease}/data_for_labeling.tsv', sep='\t', engine='python')
    #     ref_samples = ref_data["cleaned_text"].tolist()
    #
    #     data = pd.read_csv(f'data/claims/{disease}/data_for_labeling_group_2.tsv', sep='\t', engine='python')
    #     data = data[~data["cleaned_text"].isin(ref_data["cleaned_text"].tolist())]
    #
    #     data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    #     N = int(len(data) / num_samples)
    #     [data.iloc[i * num_samples:(i + 1) * num_samples].to_csv(Path('data/claims') / disease / f'data_samples_{num_samples}_{dt_string}_{i}.tsv', index=False) for i in range(N + 1)]
    #     # resampled_data = data.sample(n=num_samples, random_state=random_seed)
    #     # resampled_data.drop_duplicates(subset=['cleaned_text'], keep='last', inplace=True)
    #     # print(f'Number of the samples {len(resampled_data)}')
    #     # resampled_data.to_csv(Path('data/claims') / disease / f'data_samples_{num_samples}_{dt_string}.tsv', sep='\t', index=False)
    #
    # for disease in diseases:
    #     ref_data_1 = pd.read_csv(f'data/claims/{disease}/data_for_labeling.tsv', sep='\t', engine='python')
    #     ref_data_2 = pd.read_csv(f'data/claims/{disease}/data_for_labeling_group_2.tsv', sep='\t', engine='python')
    #
    #     ref_data = pd.concat([ref_data_1, ref_data_2])
    #     ref_data.drop_duplicates(subset=['cleaned_text'], inplace=True)
    #
    #     ref_samples = ref_data["cleaned_text"].tolist()
    #
    #     data = pd.read_csv(f'data/claims/{disease}/data_for_labeling_group_3.tsv', sep='\t', engine='python')
    #     data = data[~data["cleaned_text"].isin(ref_data["cleaned_text"].tolist())]
    #     data.drop_duplicates(subset=['cleaned_text'], keep='last', inplace=True)
    #     data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    #     N = int(len(data) / num_samples)
    #     [data.iloc[i * num_samples:(i + 1) * num_samples].to_csv(Path('data/claims') / disease / f'data_samples_coco_{num_samples}_{dt_string}_{i}.tsv', index=False) for i in range(N + 1)]
    #     # # resampled_data = data.sample(n=num_samples, random_state=random_seed)
    #     # resampled_data.drop_duplicates(subset=['cleaned_text'], keep='last', inplace=True)
    #     # print(f'Number of the samples {len(resampled_data)}')
    #     # resampled_data.to_csv(Path('data/claims') / disease / f'data_samples_{num_samples}_{dt_string}.tsv', sep='\t', index=False)

    for disease in diseases:
        ref_data_1 = pd.read_csv(f'data/claims/{disease}/data_for_labeling.tsv', sep='\t', engine='python')
        ref_data_2 = pd.read_csv(f'data/claims/{disease}/data_for_labeling_group_2.tsv', sep='\t', engine='python')
        ref_data_3 = pd.read_csv(f'data/claims/{disease}/data_for_labeling_group_3.tsv', sep='\t', engine='python')

        ref_data = pd.concat([ref_data_1, ref_data_2])
        ref_data.drop_duplicates(subset=['cleaned_text'], inplace=True)

        ref_samples = ref_data["cleaned_text"].tolist()

        data = pd.read_csv(f'data/claims/{disease}/data_for_labeling_group_4.tsv', sep='\t', engine='python')
        data = data[~data["cleaned_text"].isin(ref_data["cleaned_text"].tolist())]
        data.drop_duplicates(subset=['cleaned_text'], keep='last', inplace=True)
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        N = int(len(data) / num_samples)
        [data.iloc[i * num_samples:(i + 1) * num_samples].to_csv(
            Path('data/claims') / disease / f'data_samples_literature_{num_samples}_{dt_string}_{i}.tsv', index=False)
         for i in range(N + 1)]
        # # resampled_data = data.sample(n=num_samples, random_state=random_seed)
        # resampled_data.drop_duplicates(subset=['cleaned_text'], keep='last', inplace=True)
        # print(f'Number of the samples {len(resampled_data)}')
        # resampled_data.to_csv(Path('data/claims') / disease / f'data_samples_{num_samples}_{dt_string}.tsv', sep='\t', index=False)

    # for disease in diseases:
    #     data = pd.read_csv(f'data/claims/{disease}/data_for_labeling.tsv', sep='\t', engine='python')
    #     resampled_data = data.sample(n=num_samples, random_state=random_seed)
    #     resampled_data.drop_duplicates(subset=['cleaned_text'], keep='last', inplace=True)
    #     print(f'Number of the samples {len(resampled_data)}')
    #     resampled_data.to_csv(Path('data/claims') / disease / f'data_samples_{num_samples}.tsv', sep='\t', index=False)


def resample_for_gold_annotations():
    random_seed = 0
    num_samples = 100
    diseases = ['monkeypox']

    pass


if __name__ == '__main__':
    resample_for_gold_annotations()
