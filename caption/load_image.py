import os
import io
from PIL import Image
import pandas as pd
import base64

name = ["brazil_english_processed.tsv", "israel_english_processed.tsv", "japan_english_processed.tsv", "spain_english_processed.tsv"]
dst = ['brazil/', 'israel/', 'japan/', 'spain/']

dfs = [pd.read_csv(name[i], sep='\t') for i in range(4)]

for i in range(4):
    dst_path = "./images/" + dst[i]
    total = len(dfs[i])
    for j in range(total):
        img_b64 = dfs[i].iloc[j,1]
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(img_b64, "utf-8"))))
        img.save(dst_path + str(j) + '.png')

