from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import io
import json
import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 非表示バックエンドを指定
import matplotlib.pyplot as plt

# 追加: torchvisionを用いた人物検出用モジュール
import torchvision
from torchvision import transforms as T

# -------------------------------
# グローバル設定・初期化
# -------------------------------
DATA_CSV_PATH = "data/merged_fashion.csv"         # CSVファイルのパス
EMBEDDINGS_DB_PATH = "embeddings.db"               # SQLite DBのパス
SIMILARITY_THRESHOLD = 0.7                         # グラフ作成用の閾値（必要に応じて調整）
TOP_K = 30                                       # 出力グラフのノード数（上位30件）
SAMPLE_SIZE = 60

# グローバル変数（データセット情報）
dataset_image_urls = []
dataset_ids = []
dataset_embeddings = []  # 各画像のembedding（numpy配列）
prefix_to_color = {}     # idのプレフィックスと色の対応

# CLIPモデルとプロセッサの読み込み（アプリ起動時に一度だけ読み込み）
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 追加: 人物検出モデルの読み込み（Faster R-CNN、COCOのpersonカテゴリを使用）
person_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
person_detector.eval()  # 推論モードへ

# FastAPIアプリケーションの初期化
app = FastAPI()

# -------------------------------
# ヘルパー関数
# -------------------------------
def detect_person(image: Image.Image) -> Image.Image:
    """
    入力画像から人物（person）を検出し、最も大きな領域をクロップして返す。
    検出できなかった場合は元の画像をそのまま返す。
    """
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = person_detector(img_tensor)[0]
    
    # COCOのpersonカテゴリはラベル1。信頼度0.8以上の候補を採用
    person_boxes = []
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if label == 1 and score > 0.8:
            person_boxes.append((box, score))
    
    if person_boxes:
        # 複数ある場合は、面積が最大の領域を選択
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        best_box, _ = max(person_boxes, key=lambda x: box_area(x[0]))
        best_box = best_box.tolist()
        # 画像をクロップ（[x_min, y_min, x_max, y_max]）
        cropped_image = image.crop(best_box)
        return cropped_image
    else:
        # 人物が検出できなかった場合は元の画像を返す
        return image

def compute_embedding(image: Image.Image) -> np.ndarray:
    """
    PIL ImageからCLIPのembeddingを計算する。
    画像全体ではなく、まず人物領域を検出してからembeddingを計算する。
    """
    # 人物領域の検出とクロップ
    person_image = detect_person(image)
    inputs = processor(images=person_image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    vector = outputs.detach().numpy().flatten()
    return vector

def load_dataset_and_embeddings():
    """
    CSVからデータ読み込み、SQLite DBを用いて各画像のembeddingを取得・計算、
    外れ値除去、色の割当を行う。
    """
    global dataset_image_urls, dataset_ids, dataset_embeddings, prefix_to_color

    # CSVから画像URLとIDを読み込み
    df = pd.read_csv(DATA_CSV_PATH)
    dataset_image_urls = df["image_url"].tolist()
    dataset_ids = df["id"].tolist()

    # SQLite DB接続（なければ作成）
    conn = sqlite3.connect(EMBEDDINGS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            image_url TEXT PRIMARY KEY,
            embedding TEXT
        )
    """)
    conn.commit()

    embeddings_list = []
    for url in dataset_image_urls:
        cursor.execute("SELECT embedding FROM embeddings WHERE image_url = ?", (url,))
        row = cursor.fetchone()
        if row is not None:
            # DBから読み込み
            vector = np.array(json.loads(row[0]))
            print(f"{url} のembeddingをDBから読み込みました。")
        else:
            try:
                response = requests.get(url, stream=True, timeout=10)
                image = Image.open(response.raw).convert("RGB")
            except Exception as e:
                print(f"{url} の画像取得エラー: {e}")
                vector = np.zeros(512)  # エラー時はゼロベクトル（必要に応じて処理変更）
            else:
                vector = compute_embedding(image)
                cursor.execute("INSERT INTO embeddings (image_url, embedding) VALUES (?, ?)",
                               (url, json.dumps(vector.tolist())))
                conn.commit()
                print(f"{url} のembeddingを計算し、DBに保存しました。")
        embeddings_list.append(vector)
    conn.close()

    # 全画像間の類似度を計算し、各画像の平均類似度が平均-2σより低いものを外れ値として除去
    similarity_matrix = cosine_similarity(embeddings_list)
    avg_similarities = np.mean(similarity_matrix, axis=1)
    mean_avg = np.mean(avg_similarities)
    std_avg = np.std(avg_similarities)
    threshold_outlier = mean_avg - 2 * std_avg
    print(f"全画像数: {len(dataset_image_urls)}, 除外する外れ値の画像数: {len(dataset_image_urls) - np.count_nonzero(avg_similarities >= threshold_outlier)}")

    non_outlier_indices = np.where(avg_similarities >= threshold_outlier)[0]

    # 外れ値除去後のリスト更新
    dataset_image_urls = [dataset_image_urls[i] for i in non_outlier_indices]
    dataset_ids = [dataset_ids[i] for i in non_outlier_indices]
    dataset_embeddings = [embeddings_list[i] for i in non_outlier_indices]

    # idのプレフィックスから色を割り当て（wearのユーザー名とみなす）
    prefixes = [id_str.split("_")[0] for id_str in dataset_ids]
    unique_prefixes = sorted(set(prefixes))
    prefix_to_color = {prefix: plt.cm.tab20(i % 20) for i, prefix in enumerate(unique_prefixes)}

# アプリ起動時にデータセットの読み込み・前処理を実施
@app.on_event("startup")
def startup_event():
    load_dataset_and_embeddings()
    print("Datasetとembeddingsのロード完了。")

# -------------------------------
# 入出力モデル
# -------------------------------
class QueryInput(BaseModel):
    image_base64: str  # 入力画像のbase64文字列

# similar_wear の各項目を表すモデル
class SimilarWearItem(BaseModel):
    username: str
    image_base64: str

# predict エンドポイントの出力形式（グラフ画像と類似wear情報）
class PredictResponse(BaseModel):
    graph_image: str
    similar_wear: List[SimilarWearItem]

# -------------------------------
# APIエンドポイント（predict）
# -------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(query: QueryInput):
    # 入力のbase64文字列から画像データに変換
    try:
        image_data = base64.b64decode(query.image_base64)
        query_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="無効なbase64画像データです。")
    
    # クエリ画像のembeddingを計算（人物領域を優先して抽出）
    query_vector = compute_embedding(query_image)
    
    # クエリ画像とデータセット各画像とのコサイン類似度計算
    dataset_emb_array = np.array(dataset_embeddings)
    query_similarities = cosine_similarity([query_vector], dataset_emb_array)[0]
    
    # --- ノード選択 ---
    # 類似度が最も高い上位5件のインデックス
    top_5 = list(np.argsort(query_similarities)[-5:][::-1])
    
    # 残りの全インデックス（トップ5以外）からランダムに25件を選択
    all_indices = list(range(len(dataset_embeddings)))
    remaining_indices = list(set(all_indices) - set(top_5))
    if len(remaining_indices) >= 25:
        random_25 = np.random.choice(remaining_indices, size=SAMPLE_SIZE, replace=False)
    else:
        random_25 = remaining_indices
    # グラフに用いるインデックス（トップ5 + ランダム25）
    graph_indices = list(top_5) + list(random_25)
    
    # --- グラフ構築（スター風レイアウト：類似度に応じた距離配置） ---
    G_query = nx.Graph()
    G_query.add_node("query", image_url=None, color="red")
    for idx in graph_indices:
        prefix = dataset_ids[idx].split("_")[0]
        node_color = prefix_to_color.get(prefix, "blue")
        G_query.add_node(idx, image_url=dataset_image_urls[idx], color=node_color)
        similarity = query_similarities[idx]
        G_query.add_edge("query", idx, weight=similarity)

    # レイアウト設定：クエリノードは中央に配置
    pos = {}
    pos["query"] = np.array([0, 0])  # クエリは中央

    # 各ノードに均等な角度を割り当てる
    theta = np.linspace(0, 2*np.pi, len(graph_indices), endpoint=False)
    # 類似度が高いほど近づけるための半径設定（例: 最小距離1.0、最大距離3.0）
    min_radius = 1.0
    max_radius = 3.0

    # 選択されたノードの類似度の最小値と最大値を計算
    selected_similarities = [query_similarities[idx] for idx in graph_indices]
    min_sim = min(selected_similarities)
    max_sim = max(selected_similarities)

    # 各ノードの位置を類似度に基づいて設定
    for i, idx in enumerate(graph_indices):
        similarity = query_similarities[idx]
        # 正規化（類似度が最高なら1、最低なら0になるように）
        if max_sim - min_sim > 1e-6:
            normalized_similarity = (similarity - min_sim) / (max_sim - min_sim)
        else:
            normalized_similarity = 1.0
        # 高い類似度ほどqueryに近い（normalized_similarity=1の場合: min_radius、0の場合: max_radius）
        node_radius = max_radius - normalized_similarity * (max_radius - min_radius)
        x = node_radius * np.cos(theta[i])
        y = node_radius * np.sin(theta[i])
        pos[idx] = np.array([x, y])
    
    # ノードの色とサイズのみ設定（ノードに名前は付けません）
    node_colors = []
    node_sizes = []
    for node in G_query.nodes():
        if node == "query":
            node_colors.append("red")
            node_sizes.append(1200)
        elif node in top_5:
            node_colors.append("orange")
            node_sizes.append(1000)
        else:
            # ラベルは使用せず、色のみユーザーごとの設定
            prefix = dataset_ids[node].split("_")[0]
            node_colors.append(prefix_to_color.get(prefix, "blue"))
            node_sizes.append(700)
    
    # グラフ描画（ラベル描画は行わず、ノードとエッジのみ表示）
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.set_facecolor("white")
    nx.draw_networkx_edges(
        G_query, pos, 
        width=1.0, alpha=0.4, edge_color="gray"
    )
    nx.draw_networkx_nodes(
        G_query, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black", linewidths=0.5, alpha=0.9
    )
    
    plt.title("Similar Items", fontsize=14, fontweight="bold", color="black")
    plt.axis("off")
    
    # グラフ画像をbase64に変換
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    # --- similar_wear: 上位5件の類似画像を返す ---
    similar_wear = []
    for idx in top_5:
        username = dataset_ids[idx].split("_")[0]  # idの先頭部分をユーザー名と仮定
        image_url = dataset_image_urls[idx]
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            sim_image = Image.open(response.raw).convert("RGB")
            buf_sim = io.BytesIO()
            sim_image.save(buf_sim, format="PNG")
            buf_sim.seek(0)
            image_sim_base64 = base64.b64encode(buf_sim.read()).decode("utf-8")
        except Exception as e:
            image_sim_base64 = ""
        similar_wear.append({
            "username": username,
            "image_base64": image_sim_base64
        })
    
    # 出力JSON：グラフ画像と類似wear情報（上位5件）を返す
    return PredictResponse(graph_image=graph_base64, similar_wear=similar_wear)

# -------------------------------
# メイン起動
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
