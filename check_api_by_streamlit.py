import base64
import requests
import streamlit as st
from io import BytesIO
from PIL import Image

def encode_image_to_base64_from_bytes(image_bytes: bytes) -> str:
    """画像のbytesからbase64文字列に変換する"""
    return base64.b64encode(image_bytes).decode("utf-8")

def decode_base64_to_image(base64_str: str) -> Image.Image:
    """base64文字列からPILのImageオブジェクトに変換する"""
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

# Streamlit のファイルアップローダーで画像を入力
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像データを読み込んでbase64エンコード
    image_bytes = uploaded_file.read()
    image_base64 = encode_image_to_base64_from_bytes(image_bytes)

    # APIエンドポイント（FastAPIサーバーが起動していることを確認してください）
    api_url = "http://localhost:8000/predict"

    # リクエストペイロード作成
    payload = {
        "image_base64": image_base64
    }

    # APIへPOSTリクエストを送信
    response = requests.post(api_url, json=payload)

    # レスポンスの確認とレンダリング
    if response.status_code == 200:
        result = response.json()

        # 取得したグラフ画像をデコードして表示
        graph_image_base64 = result.get("graph_image")
        if graph_image_base64:
            graph_image = decode_base64_to_image(graph_image_base64)
            st.image(graph_image, caption="Graph Image", use_column_width=True)
        else:
            st.warning("グラフ画像が取得できませんでした。")

        st.markdown("### Similar Wear")
        similar_wears = result.get("similar_wear", [])

        # レコメンドされた各画像を表示
        for item in similar_wears:
            username = item.get("username", "Unknown")
            similar_image_base64 = item.get("image_base64")
            if similar_image_base64:
                similar_image = decode_base64_to_image(similar_image_base64)
                st.image(similar_image, caption=f"Username: {username}", width=200)
            else:
                st.warning(f"{username} の画像が取得できませんでした。")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
else:
    st.info("画像をアップロードしてください。")
