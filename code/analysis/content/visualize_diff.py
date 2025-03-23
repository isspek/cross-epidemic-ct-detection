import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scattertext as st
import spacy
import requests, zipfile, io
from analysis.utils import clean_tweet_v2

if __name__ == '__main__':
	fpath = 'data/'
	diseases = ['ebola', 'zika', 'covid', 'monkeypox']
	nlp = spacy.load('en_core_web_trf')
	frame_classified_data = []
	output_path = f'{fpath}/frames.tsv'
	for disease in diseases:
		print(f'Disease: {disease}')
		data_path = f'{fpath}/{disease}/processed_v2/data.tsv'

		data = pd.read_csv(data_path, sep='\t')
		data['text'] = data['text'].apply(lambda x: clean_tweet_v2(x,
																   keep_numbers=False,
																   lowerize=True,
																   keep_punctuation=False,
																   replace_disease_names=False,
																   remove_disease_name=False))

		data = data[data['text'].apply(lambda x: all(len(token) > 1 for token in x.split()))]
		data.reset_index(drop=True, inplace=True)

		data['category'] = data['label'].apply(lambda x: 'CT' if x==1 else 'not CT')
		corpus = st.CorpusFromPandas(data, category_col='category', text_col='text', nlp=nlp).build()

		st_obj = st.produce_scattertext_explorer(corpus, category = 'CT',
											   category_name = 'CT',
											   not_category_name = 'not CT',
											   minimum_term_frequency=5,
											   width_in_pixels = 1000,
											   show_characteristic=True,
											   show_top_terms=True,
											   show_corpus_stats=False,
											   show_cross_axes=False,
											   include_term_category_counts=False,
											   return_scatterplot_structure=True
											   )
		fig = st.produce_scattertext_pyplot(st_obj,dpi=600)
		fig.savefig(f"{disease}_visualization.pdf", format='pdf', bbox_inches = "tight")