from fastapi import FastAPI
from typing import List
import json
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from datetime import datetime

import re
import pandas as pd

import hdbscan
import umap

from dataclasses import dataclass


@dataclass
class Data:
    size: str
    ids: List


class News(BaseModel):
    data: list


app = FastAPI()
# //========================
# model_path = "kpfsbert-base" # sbert
model_path = "yunaissance/kpfbert-base"  # sbert
# model_path = "bongsoo/kpf-sbert-v1.1"  # sbert
# kpfSBERT 모델 로딩
model = SentenceTransformer(model_path)


def decode_unicode(data):
    return bytes(data, 'utf-8').decode('unicode_escape')


@app.post('/main')
def clusterNews(news: News):
    dicted_items = [item for item in news.data]
    json_data = json.dumps(dicted_items)
    return main(json_data)


# UMAP 차원축소 실행
def umap_process(corpus_embeddings, n_components=5):
    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=n_components,
                                metric='cosine').fit_transform(corpus_embeddings)
    return umap_embeddings


# HDBSCAN 실행
def hdbscan_process(corpus, corpus_embeddings, min_cluster_size, min_samples, umap=True, n_components=5,
                    method='eom'):
    if umap:
        umap_embeddings = umap_process(corpus_embeddings, n_components)
    else:
        umap_embeddings = corpus_embeddings

    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              min_samples=min_samples,
                              allow_single_cluster=True,
                              metric='euclidean',
                              core_dist_n_jobs=1,  # knn_data = Parallel(n_jobs=self.n_jobs, max_nbytes=None) in joblib
                              cluster_selection_method=method).fit(umap_embeddings)  # eom leaf

    docs_df = pd.DataFrame(corpus, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

    return docs_df, docs_per_topic


def main(json):
    cluster_mode = 'title'
    # Data Loading to clustering
    # DATA_PATH = 'data/newstrust/newstrust_sample.json'
    # df = pd.read_json(json, encoding='utf-8')
    df = pd.read_json(json, encoding='utf-8')

    # 카테고리별 클러스터링
    start = datetime.now()
    print('작업 시작시간 : ', start)
    previous = start
    bt_prev = start

    tot_df = pd.DataFrame()

    print(' processing start... with cluster_mode :', cluster_mode)

    category = df.category.unique()

    df_category = []
    for categ in category:
        df_category.append(df[df.category == categ])
    cnt = 0
    rslt = []
    topics = []
    # 순환하며 데이터 만들어 df에 고쳐보자
    for idx, dt in enumerate(df_category):
        try:
            corpus = dt[cluster_mode].values.tolist()
            # '[보통공통된꼭지제목]' 형태를 제거해서 클러스터링시 품질을 높인다.
            for i, cp in enumerate(corpus):
                corpus[i] = re.sub(r'\[(.*?)\]', '', cp)
            #     print(corpus[:10])
            corpus_embeddings = model.encode(corpus, show_progress_bar=True)

            docs_df, docs_per_topic = hdbscan_process(corpus, corpus_embeddings,
                                                      umap=False, n_components=2,  # 연산량 줄이기 위해 umap 사용시 True
                                                      method='leaf',
                                                      min_cluster_size=2,
                                                      min_samples=2,
                                                      )
            cnt += len(docs_df)

            rslt.append(docs_df)
            topics.append(docs_per_topic)
            dt['cluster'] = docs_df['Topic'].values.tolist()
            tot_df = pd.concat([tot_df, dt])

            bt = datetime.now()
            print(len(docs_df), 'docs,', len(docs_per_topic) - 1, 'clusters in', category[idx], ', 소요시간 :',
                  bt - bt_prev)
            bt_prev = bt
        except:
            print("category 수 부족")

    now = datetime.now()
    print('#Total docs :', cnt, 'in', len(rslt), 'Categories', ', 소요시간 :', now - previous)
    previous = now

    # cluster update

    df['cluster'] = tot_df['cluster'].astype(str)

    end = datetime.now()
    print('작업 종료시간 : ', end, ', 총 소요시간 :', end - start)

    res = []
    size = df.cluster.size
    print("size", size, "category", category)
    for cate in category:
        for i in range(0, size):
            print("cate", cate, "size_ ", str(i))

            condition = (df.category == cate) & (df.cluster == str(i))  # 조건식 작성
            test = df[condition]
            if len(test) == 0:
                break

            id_list = [item.replace('k', '') for item in test.id.values.tolist()]

            res.append(Data(size=len(test), ids=id_list))
    return res


def get_json_list():
    data_list = [
        {'title': 'John', "category": "IT 과학", 'age': 30},
        {'title': 'Jane', "category": "IT 과학", 'age': 25},
        {'title': 'Mike', "category": "IT 과학", 'age': 35}
    ]
    json_data = json.dumps(data_list)
    return json_data

# main(get_json_list())
