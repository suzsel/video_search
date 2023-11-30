import faiss
import pandas as pd
import numpy as np
import fastparquet
from autocorrect import Speller
from sentence_transformers import SentenceTransformer
from pyarrow.parquet import ParquetFile
import pyarrow as pa

"""
def Doc2Vec(table_name: str):
    return encoder.encode(df[table_name])

pf = ParquetFile("C:/Users/po_67/Downloads/train_data/videos.parquet")
a = next(pf.iter_batches(batch_size=1000))
df = pa.Table.from_batches([a]).to_pandas()
np.save("table_channel_title", df['channel_title'])
np.save("table_video_title", df['video_title'])

np.save("table_video_id", df['video_id'])

encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

index_arr = []
for i, el in enumerate([Doc2Vec('video_title'), Doc2Vec('channel_title')]):
    index_arr.append(faiss.IndexFlatL2(el.shape[1]))
    faiss.normalize_L2(el)
    index_arr[i].add(el)
np.save("index_arr", index_arr)

"""
print("Initialization started")
encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
index_arr = np.load('index_arr.npy', allow_pickle=True)
table_t = np.load('table_video_title.npy', allow_pickle=True)
table_i = np.load('table_video_id.npy', allow_pickle=True)
table_c = np.load('table_channel_title.npy', allow_pickle=True)
spell = Speller('ru')

print("Finding started")
search_text = spell('Секреты Творческого Прогресса')
search_vector = encoder.encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)

ann_arr = []
for i in range(len(index_arr)):
    ann_arr.append(index_arr[i].search(_vector, k=index_arr[i].ntotal))

for i in range(5):
    if ann_arr[0][0][0][i] <= ann_arr[1][0][0][i] * np.pi:
        print(table_i[ann_arr[0][1][0][i]] + " --- " + table_c[ann_arr[0][1][0][i]] + " --- " + table_t[ann_arr[0][1][0][i]])
    else:
        print(table_i[ann_arr[1][1][0][i]] + " --- " + table_c[ann_arr[1][1][0][i]] + " --- " + table_t[ann_arr[1][1][0][i]])