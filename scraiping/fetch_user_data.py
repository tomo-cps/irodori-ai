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

    # ユーザー名の抽出
    match = re.search(r"https?://wear\.jp/([^/]+)/?", url)
    username = match.group(1) if match else "wear_user"

    # 画像と投稿URLの紐付け
    data = []
    seen = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        match = re.match(rf"^/{username}/(\d+)/$", href)
        if match:
            post_id = match.group(1)
            post_url = f"https://wear.jp{href}"
            img_tag = a_tag.find("img", src=True)
            if img_tag:
                img_url = img_tag["src"]
                if img_url.startswith("https://images.wear2.jp/coordinate/") and img_url not in seen:
                    seen.add(img_url)
                    data.append((img_url, post_url))

    # CSVファイル名
    os.makedirs("data", exist_ok=True)
    csv_filename = f"data/fashion_{username}.csv"

    # CSVに保存
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "user_name", "image_url", "post_url"])
        for idx, (img_url, post_url) in enumerate(data, start=1):
            writer.writerow([f"{username}_{idx}", username, img_url, post_url])

    print(f"画像URLを {csv_filename} に保存しました（{len(data)} 件）。")


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
    # "https://wear.jp/kyota0245/",
    # "https://wear.jp/osayu912abc/",
    # "https://wear.jp/sensenakajima/",
    # "https://wear.jp/moken/",
    # "https://wear.jp/riho0914/",
    # "https://wear.jp/1107my/",
    # "https://wear.jp/misane1209/",
    # "https://wear.jp/kuruminn61/",
    # "https://wear.jp/coltwear/",
    # "https://wear.jp/crewtiger/",
    # "https://wear.jp/itkwear/",
    # "https://wear.jp/11shion28/",
    # "https://wear.jp/kkren9610/",
    # "https://wear.jp/maypikapi/",
    # "https://wear.jp/maira0818/",
    # "https://wear.jp/0116mn/",
    # "https://wear.jp/10momoon10/",
    # "https://wear.jp/loveyxoxo/",
]

for url in urls:
    scrape_wear_image_urls(url)

merge_csv_files()  # すべてのファイルをマージ
