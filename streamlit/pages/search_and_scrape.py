import os

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googleapiclient.discovery import build

import streamlit as st

load_dotenv()


@st.cache_resource
def search_and_scrape(query):
    output = []
    # Google Custom Search APIを使用して検索
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
    result = service.cse().list(q=query, cx=os.getenv("GOOGLE_CSE_ID")).execute()

    # スニペット取得
    items = result.get("items", [])
    for item in items:
        # ページの中身を取得
        page_url = item.get("link")
        if page_url:
            try:
                response = requests.get(page_url)
                if response.status_code == 200:
                    # page_content = response.text
                    # BeautifulSoupを使ってHTMLを解析
                    soup = BeautifulSoup(response.text, "html.parser")
                    page_content = soup.get_text()
                    lines = [line.strip() for line in page_content.splitlines()]
                    text = "\n".join(line for line in lines if line)
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page content: {e}")
                text = ""
        output.append([item.get("title"), item.get("snippet"), item.get("link"), text])
    return output


def main():
    st.set_page_config(page_title="Interweb Explorer", layout="wide")
    # 検索クエリを指定して実行
    query = st.text_input("`Ask a question:`")
    if (len(query) > 0) & (st.button("検索実行")):
        df_output = search_and_scrape(query)
        df_output = pd.DataFrame(df_output, columns=["タイトル", "スニペット", "URL", "コンテンツ"])
        st.session_state["df_output"] = df_output
    if "df_output" in st.session_state:
        df_output = st.session_state["df_output"]
        st.data_editor(df_output)


if __name__ == "__main__":
    main()
