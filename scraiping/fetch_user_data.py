import requests
from bs4 import BeautifulSoup
import csv
import re
import os
import glob
import pandas as pd

def scrape_wear_image_urls(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # 画像URLの抽出＋フィルタ
    urls = sorted({
        tag['src']
        for tag in soup.find_all('img', src=True)
        if tag['src'].startswith("https://images.wear2.jp/coordinate/")
    })

    # 正規表現でユーザー名を抽出
    match = re.search(r"https?://wear\.jp/([^/]+)/?", url)
    username = match.group(1) if match else "wear_user"

    # CSVファイル名
    os.makedirs("data", exist_ok=True)
    csv_filename = f"data/fashion_{username}.csv"

    # CSVに保存（ID列にユーザー名を含める）
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "image_url"])
        for idx, img_url in enumerate(urls, start=1):
            writer.writerow([f"{username}_{idx}", img_url])

    print(f"画像URLを {csv_filename} に保存しました（{len(urls)} 件）。")

def merge_csv_files(output_file="data/merged_fashion.csv"):
    csv_files = glob.glob("data/fashion_*.csv")
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"\n✅ マージ完了: {output_file} に {len(merged_df)} 件の画像URLを保存しました。")

# --- 実行 ---
urls = [
    "https://wear.jp/yusukeogura20020903/",
    "https://wear.jp/tyomoki/",
    "https://wear.jp/kyota0245/",
    "https://wear.jp/osayu912abc/",
    "https://wear.jp/sensenakajima/",
    "https://wear.jp/moken/",
    "https://wear.jp/riho0914/",
    "https://wear.jp/1107my/",
    "https://wear.jp/misane1209/",
    "https://wear.jp/kuruminn61/"
]

for url in urls:
    scrape_wear_image_urls(url)

merge_csv_files()  # すべてのファイルをマージ
