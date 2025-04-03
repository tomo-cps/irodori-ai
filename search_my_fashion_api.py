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
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 非表示バックエンドを指定
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams["font.family"] = "Noto Sans CJK JP"

# 追加: torchvisionを用いた人物検出用モジュール
import torchvision
from torchvision import transforms as T
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -------------------------------
# グローバル設定・初期化
# -------------------------------
DATA_CSV_PATH = "data/merged_fashion.csv"         # CSVファイルのパス
EMBEDDINGS_DB_PATH = "embeddings.db"               # SQLite DBのパス
SIMILARITY_THRESHOLD = 0.7                         # グラフ作成用の閾値（必要に応じて調整）
TOP_K = 30                                       # 出力グラフのノード数（上位30件）
SAMPLE_SIZE = 60
ICON_ZOOM = 0.06
QUERY_ICON_ZOOM = 0.3

# グローバル変数（データセット情報）
dataset_image_urls = []
dataset_ids = []
dataset_post_urls = []   # 追加：投稿URLを格納するリスト
dataset_embeddings = []  # 各画像のembedding（numpy配列）
prefix_to_color = {}
icon_image = {
    "riho0914": "https://images.wear2.jp/profile/rlipwVx5/esnOPkY6/1717473531_640.jpg",
    "osayu912abc": "https://images.wear2.jp/profile/EJiWRaL/vB5Tsa8Z/1735117297_640.jpg",
    "sensenakajima": "https://cdn.wimg.jp/profile/zbdysk/20210614202004353_640.jpg",
    "kuruminn61": "https://images.wear2.jp/profile/7binY4xl/X6bnqpkv/1720241651_640.jpg",
    "tyomoki": "https://images.wear2.jp/profile/8BiX8WOY/j25TWXtz/1715557148_640.jpg",
    "kyota0245": "https://images.wear2.jp/profile/EJiqj7RD/6c4azuGZ/1739180663_640.jpg",
    "moken": "https://cdn.wimg.jp/profile/e5gvvs/20210422190902624_640.jpg",
    "1107my": "https://images.wear2.jp/profile/nZizJXW8/m0biIIts/1714111268_640.jpg",
    "misane1209": "https://images.wear2.jp/profile/zaib64wl/MuQT1nEj/1710816720_640.jpg",
    "yusukeogura20020903": "https://images.wear2.jp/profile/zaijQaw5/V62sSXtC/1701199778_640.jpg",
    "coltwear": "https://images.wear2.jp/profile/Gri66bAY/5oSjAnxh/1701856953_640.jpg",
    "crewtiger": "https://images.wear2.jp/profile/p3iqGb7q/xwheHquD/1743326937_640.jpg",
    "itkwear": "https://images.wear2.jp/profile/PaikL34v/8hvHQhGn/1691053406_640.jpg",
    "11shion28": "https://images.wear2.jp/profile/Z8iaZlkD/FcuxbiuY/1730267967_640.jpg",
    "kkren9610": "https://images.wear2.jp/profile/gGizeg2J/dopOJL8C/1708341711_640.jpg",
    "maypikapi": "https://images.wear2.jp/profile/7biBr06o/9iFONlV7/1742532001_640.jpg",
    "maira0818": "https://images.wear2.jp/profile/lei2bGv7/VlD8B9NA/1731239452_640.jpg",
    "0116mn": "https://images.wear2.jp/profile/QWivp2Ox/o2IwPuNa/1677513159_640.jpg",
    "10momoon10": "https://images.wear2.jp/profile/GrigMn7m/Ci2Lbyg4/1655486994_640.jpg",
    "loveyxoxo": "https://images.wear2.jp/profile/08iXYanQ/YmhcAPjJ/1721650191_640.jpg"
}

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
    global dataset_image_urls, dataset_ids, dataset_post_urls, dataset_embeddings, prefix_to_color

    # CSVから画像URL, ID, 投稿URLを読み込み
    df = pd.read_csv(DATA_CSV_PATH)
    dataset_image_urls = df["image_url"].tolist()
    dataset_ids = df["id"].tolist()
    dataset_post_urls = df["post_url"].tolist()  # 追加：投稿URLの読み込み

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
                vector = np.zeros(512)  # エラー時はゼロベクトル
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
    dataset_post_urls = [dataset_post_urls[i] for i in non_outlier_indices]  # 追加
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
    query_width: int = None    # （任意）中央画像の希望横幅（ピクセル）
    query_height: int = None   # （任意）中央画像の希望縦幅（ピクセル）

class SimilarWearItem(BaseModel):
    username: str
    image_base64: str
    post_url: str   # 追加：投稿へのURLを含める

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
    
    # 残りの全インデックス（トップ5以外）からランダムにSAMPLE_SIZE件を選択
    all_indices = list(range(len(dataset_embeddings)))
    remaining_indices = list(set(all_indices) - set(top_5))
    if len(remaining_indices) >= SAMPLE_SIZE:
        random_sample = np.random.choice(remaining_indices, size=SAMPLE_SIZE, replace=False)
    else:
        random_sample = remaining_indices
    # グラフに用いるインデックス（トップ5 + ランダムサンプル）
    graph_indices = list(top_5) + list(random_sample)
    
    # --- グラフ構築（スター風レイアウト：類似度に応じた距離配置） ---
    G_query = nx.Graph()
    # クエリノードのimage_urlはNoneとする
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
    
    # ノードの色とサイズの設定
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
            prefix = dataset_ids[node].split("_")[0]
            node_colors.append(prefix_to_color.get(prefix, "blue"))
            node_sizes.append(700)
    
    # --- グラフ描画 ---
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.set_facecolor("white")
    nx.draw_networkx_edges(
        G_query, pos, 
        width=1.0, alpha=0.4, edge_color="gray"
    )
    
    # --- 中心画像のリサイズ ---
    # 入力画像の大きさに注意し、最大幅100px、最大高さ130pxに収まるようリサイズ（アスペクト比維持）
    MAX_CENTER_WIDTH = 100
    MAX_CENTER_HEIGHT = 130
    original_width, original_height = query_image.size
    scale_factor = min(MAX_CENTER_WIDTH / original_width, MAX_CENTER_HEIGHT / original_height, 1)
    resized_width = int(original_width * scale_factor)
    resized_height = int(original_height * scale_factor)
    query_image_draw = query_image.resize((resized_width, resized_height), Image.LANCZOS)
    
    # ノード描画（中心ノードはリサイズ済み画像、その他は既存アイコン画像）
    def crop_to_circle(img: Image.Image) -> Image.Image:
        """
        入力画像を正方形にクロップし、円形マスクを適用して円形画像を返す。
        """
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        img_cropped = img.crop((left, top, right, bottom))
        mask = Image.new("L", (min_dim, min_dim), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, min_dim, min_dim), fill=255)
        output = Image.new("RGBA", (min_dim, min_dim))
        output.paste(img_cropped, (0, 0), mask)
        return output

    for node in G_query.nodes():
        if node == "query":
            im = OffsetImage(np.array(query_image_draw), zoom=1)
            ab = AnnotationBbox(im, pos[node], frameon=False, pad=0.0)
            ax.add_artist(ab)
        else:
            username = dataset_ids[node].split("_")[0]
            if username in icon_image:
                icon_url = icon_image[username]
                try:
                    response = requests.get(icon_url, stream=True, timeout=10)
                    icon = Image.open(response.raw).convert("RGBA")
                    icon = crop_to_circle(icon)
                    print(f"ノード {node}（{username}）のアイコン画像サイズ: {icon.size[1]}x{icon.size[0]}")
                except Exception as e:
                    fallback_color = prefix_to_color.get(username, "blue")
                    circle = plt.Circle(pos[node], 0.1, color=fallback_color, zorder=10)
                    ax.add_artist(circle)
                    continue
                im = OffsetImage(np.array(icon), zoom=ICON_ZOOM)
                ab = AnnotationBbox(im, pos[node], frameon=False, pad=0.0)
                ax.add_artist(ab)
            else:
                fallback_color = prefix_to_color.get(username, "blue")
                circle = plt.Circle(pos[node], 0.1, color=fallback_color, zorder=10)
                ax.add_artist(circle)

    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    # --- similar_wear: 上位5件の類似画像を返す ---
    similar_wear = []
    for idx in top_5:
        username = dataset_ids[idx].split("_")[0]
        image_url = dataset_image_urls[idx]
        post_url = dataset_post_urls[idx]  # 追加：投稿URLを取得
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
            "image_base64": image_sim_base64,
            "image_url": image_url,
            "post_url": post_url  # 追加：投稿URLを出力データに含める
        })
    
    return PredictResponse(graph_image=graph_base64, similar_wear=similar_wear)

# -------------------------------
# メイン起動
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
