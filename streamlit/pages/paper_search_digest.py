# ref
"""
https://zenn.dev/team_zenn/articles/117424abb5605b
https://zenn.dev/ozushi/articles/ebe3f47bf50a86
https://developers.notion.com/reference/post-page
https://info.arxiv.org/help/api/user-manual.html
https://api.semanticscholar.org/api-docs/graph
"""
import os
from dotenv import load_dotenv

import requests
import time
from datetime import datetime, timezone
    
from scipy.stats import rankdata
import numpy as np
import pandas as pd
import openai
import arxiv

import streamlit as st

# 環境変数の読み込み
load_dotenv()

# 現在時刻
now = datetime.now(timezone.utc)

#OpenAI Settings
openai.api_key = os.getenv('OPENAI_API_KEY')

OPENAI_PROMPT = """与えられた論文の要点を3点のみでまとめ、以下のフォーマットで日本語で出力してください。```
Titleの日本語訳
・要点1
・要点2
・要点3
```"""
OPENAI_PROMPT = """Please summarize the main points of the given paper in three points and output them in Japanese in the following format. ```
Title in Japanese
1.
2.
3.
```"""
OPENAI_PROMPT_DATA = """Please extract information about Dataset, Analysis method, Model, Metric, Result and Future work from the given paper and output it in Japanese in the following format. 
```
Dataset:
Analysis method:
Model:
Metric:
Result:
Future work:
```"""

#Notion Settings
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
NOTION_DATABASE_ID = os.getenv('NOTION_DATABASE_ID')
NOTION_API_URL = 'https://api.notion.com/v1/pages'

headers = {
    "Accept": "application/json",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
    "Authorization": "Bearer " + NOTION_API_KEY
}

payload = {
    "parent": {
        "database_id": NOTION_DATABASE_ID
    },
    "properties": {
        "Title": {
            "title": [
            {
                "text": {
                "content": ""
                }
            }
            ],
        },
        "タイトル": {
            "rich_text": [
            {
                "text": {
                "content": ""
                }
            }
            ],
        },
        "TLDR": {
            "rich_text": [
            {
                "text": {
                "content": ""
                }
            }
            ],
        },
        "Detail": {
            "rich_text": [
            {
                "text": {
                "content": ""
                }
            }
            ],
        },
        "Date": {
            "date": {
                "start": ""
                }
            },
        "Citation": {
            "number": 0
        },
        "InfluentialCitation": {
            "number": 0
        },
        "URL":{
            "url": ""
        },
        "arXiv query": {
            "rich_text": [
            {
                "text": {
                "content": ""
                }
            }
            ],
        },
    },
    "children":[
        {
            "object": 'block',
            "type": 'heading_1',
            "heading_1":{
                "rich_text": [
            {
                "text": {
                "content": "Abstract"
                }
            }
            ],
            }
        },
        {
            "object": 'block',
            "type": 'paragraph',
            "paragraph":{
                "rich_text": [
            {
                "text": {
                "content": ""
                }
            }
            ],
            }
        },
        {
            "object": 'block',
            "type": 'heading_1',
            "heading_1":{
                "rich_text": [
            {
                "text": {
                "content": "要約"
                }
            }
            ],
            }
        },
        {
            "object": 'block',
            "type": 'paragraph',
            "paragraph":{
                "rich_text": [
            {
                "text": {
                "content": ""
                }
            }
            ],
            }
        },
        {
            "object": 'block',
            "type": 'heading_1',
            "heading_1":{
                "rich_text": [
            {
                "text": {
                "content": "To do"
                }
            }
            ],
            }
        },
        {
            "object": 'block',
            "type": 'to_do',
            "to_do":{
                "rich_text": [
            {
                "text": {
                "content": ""
                }
            }
            ],
            "checked": False,
            "color": "default",
            }
        }
    ],
}


def create_notion_page(result,summary,response_semanticscholar):
    output = []
    # from arxiv API
    payload['properties']['Title']['title'][0]['text']['content'] = result.title
    output.append(result.title)
    payload['properties']['URL']['url'] = result.entry_id
    output.append(result.entry_id)
    payload['children'][1]['paragraph']['rich_text'][0]['text']['content'] = result.summary
    output.append(result.summary)
    payload['properties']['Date']['date']['start'] = result.published.strftime("%Y-%m-%d %H:%M:%S")
    output.append(result.published.strftime("%Y-%m-%d %H:%M:%S"))
    
    # from OpenAI API
    title_jp, *body = summary.split('\n')
    body = '\n'.join(body)
    
    payload['properties']['タイトル']['rich_text'][0]['text']['content'] = title_jp
    output.append(title_jp)
    payload['properties']['Detail']['rich_text'][0]['text']['content'] = body
    output.append(body)
    
    # from Semantic Scholar Academic Graph API
    if not 'error' in response_semanticscholar:
        payload['properties']['Citation']['number'] = response_semanticscholar['citationCount']
        payload['properties']['InfluentialCitation']['number'] = response_semanticscholar['influentialCitationCount']
        if not response_semanticscholar['tldr'] is None:
            payload['properties']['TLDR']['rich_text'][0]['text']['content'] = response_semanticscholar['tldr']['text']
        else:
            payload['properties']['TLDR']['rich_text'][0]['text']['content'] = ""
            
    else:
        payload['properties']['Citation']['number'] = 0
        payload['properties']['InfluentialCitation']['number'] = 0
        payload['properties']['TLDR']['rich_text'][0]['text']['content'] = ""
    
    output.append(payload['properties']['Citation']['number'])
    output.append(payload['properties']['InfluentialCitation']['number'])
    output.append(payload['properties']['TLDR']['rich_text'][0]['text']['content'])
    
    # from OpenAI API
    # summary_data = summary_data.split('\n')
    # for i in range(min(len(summary_data),6)):
    #     output.append(summary_data[i])
    # if len(summary_data)<6:
    #     for i in range(int(6-len(summary_data))):
    #         output.append("")
    
    response_notion = requests.post(NOTION_API_URL, json=payload, headers=headers)
    return response_notion,output


def main():
    # arxiv APIで最新の論文情報を取得する
    st.set_page_config(layout='wide', initial_sidebar_state='expanded')
    ARXIV_QUERY = st.text_input('クエリを入力','all:%22 reciprocal %22 AND all:%22 recommend %22')
    st.code("""
    - タイトル検索:ti
    'ti:%22 reciprocal %22 AND ti:%22 recommend %22'
    'ti:%22 strawberry %22'
    'ti:%22 tomato %22'
    'ti:%22 crop %22'
    - 全文検索:all
    'all:%22 yield %22 AND all:%22 predict %22'
    'all:%22 reciprocal %22 AND all:%22 recommend %22' # 相互推薦
    - 全文検索:all
    'all:cs.CV' # コンピュータービジョンとパターン認識
    'all:stat.ML' # 機械学習
    'all:q-fin.MF' # 数学ファイナンス
    'all:q-fin.PM' # ポートフォリオマネジメント
    'all:q-fin.PR' # 証券のプライシング
    'all:q-fin.RM' # リスク管理
    'all:q-fin.TR' # Trading and Market Microstructure
    - カテゴリ検索:cat
    'cat:cs.AI' # 人工知能
    """)
    ARXIV_NUM_PAPERS = st.number_input('arXivで検索する論文数を入力',value=500)
    RESULT_NUM_PAPERS = st.number_input('その中からOpenAI APIで要約する論文数を入力',value=10)
    payload['properties']['arXiv query']['rich_text'][0]['text']['content'] = ARXIV_QUERY
    if st.button('計算実行'):
        search = arxiv.Search(
            query=ARXIV_QUERY,  # 検索クエリ
            max_results=ARXIV_NUM_PAPERS,  # 取得する論文数
            sort_by=arxiv.SortCriterion.Relevance,  # 論文を投稿された日付でソートする(SubmittedDate,Relevance,LastUpdatedDate)
            sort_order=arxiv.SortOrder.Descending,  # 新しい論文から順に取得する(Descending,Ascending)
        )
        
        #searchの結果をリストに格納
        results = []
        arxiv_list = []
        for result in search.results():
            results.append(result)
            arxiv_list.append("ARXIV:"+result.entry_id.split("/")[-1].split("v")[0])
        
        # to Semantic Scholar Academic Graph API
        try:
            response_semanticscholar = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': 'citationCount,influentialCitationCount,tldr'},
            json={"ids": arxiv_list}
            )
            response_semanticscholar = response_semanticscholar.json()
        except:
            print("Semantic Scholar Academic Graph API Error")
            st.write("Semantic Scholar Academic Graph API Error")
    
        # 並び替え    
        order_list = []
        for i,s in enumerate(response_semanticscholar):
            if (s is None)|(s=='error'):
                order_list.append(0)
            else:
                # 日数あたりの引用回数
                order_list.append(-s['citationCount']/((now-results[i].published).days+1)) # citationCount,influentialCitationCount
        order_list = rankdata(order_list, method='ordinal')-1
        order_list = sorted(range(len(order_list)), key=lambda k: order_list[k])
        
        results = list(np.array(results)[order_list])
        response_semanticscholar = list(np.array(response_semanticscholar)[order_list])
        
        # 論文情報をNotion投稿する
        df_output = []
        for i,result in enumerate(results[:RESULT_NUM_PAPERS]):
            # API制限のため（3回/60秒）
            start = time.time()
            
            # to OpenAI API
            try:
                # text = f"Title: {result.title}\n Abstract: {result.summary}\n TLDR: {response_semanticscholar[i]['tldr']['text']}"
                text = f"Title: {result.title}\n Abstract: {result.summary}"
                response_chatgpt = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {'role': 'system', 'content': OPENAI_PROMPT},
                                {'role': 'user', 'content': text}
                            ],
                            temperature=0,
                        )
                summary = response_chatgpt['choices'][0]['message']['content']
                
                # response_chatgpt_data = openai.ChatCompletion.create(
                #             model="gpt-3.5-turbo",
                #             messages=[
                #                 {'role': 'system', 'content': OPENAI_PROMPT_DATA},
                #                 {'role': 'user', 'content': result.summary}
                #             ],
                #             temperature=0,
                #         )
                # summary_data = response_chatgpt_data['choices'][0]['message']['content']
                
            except:
                print("OpenAI API Error")
                        
            # to Notion API
            try:
                response_notion,output = create_notion_page(result,summary,response_semanticscholar[i])
                print(response_notion)
                df_output.append(output)
            except:
                print("Notion API Error")
                
            elapsed_time = time.time()-start
            if elapsed_time<=20:
                print(f"Sleep for {20-elapsed_time} seconds due to API limitation")
                time.sleep(20-elapsed_time)
        df_output = pd.DataFrame(df_output,columns=['Title','URL','Abstract','Date','タイトル','詳細','Citation','InfluentialCitation','TLDR',
                                                    # 'データセット','分析手法','モデル','評価指標','結果','今後の展望',
                                                    ])
        st.session_state['df_output'] = df_output
    if 'df_output' in st.session_state:
        df_output = st.session_state['df_output']
        st.table(df_output[['Title','タイトル','詳細','Date','Citation','TLDR','URL']])
        st.table(df_output)
        
        st.download_button(
            "結果出力",
            df_output.to_csv(index=False).encode('shift_jis','ignore'),
            "output.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main()