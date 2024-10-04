import configparser
import os
import time

import awswrangler as wr
import boto3
import pandas as pd

import streamlit as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource
def load_dataset():
    df = pd.read_parquet(
        "../../../../自己啓発/スマート農業/青果物市況情報/data_all.parquet"
    )
    # df = wr.s3.read_parquet("s3://mysample111/data_all.parquet")
    return df


@st.cache_resource
def load_client(platform="local"):
    if platform == "ec2":
        aws_access_key_id = None
        aws_secret_access_key = None
    elif platform == "local":
        # 認証情報のファイルパス
        credentials_path = "../../../../.aws/credentials"

        # ConfigParserオブジェクトを作成
        config = configparser.ConfigParser()
        config.read(credentials_path)

        # プロファイル名を指定して認証情報を取得
        aws_access_key_id = config["default"]["aws_access_key_id"]
        aws_secret_access_key = config["default"]["aws_secret_access_key"]

    # クライアントの作成
    client = boto3.client(
        "athena",
        region_name="ap-northeast-1",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    return client


def wait_for_query_to_complete(client, query_execution_id):
    while True:
        response = client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response["QueryExecution"]["Status"]["State"]
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            return status
        elif status == "QUEUED" or status == "RUNNING":
            st.info(f"Query is still {status}. Waiting before checking again.")
            time.sleep(10)


df = load_dataset()
client = load_client()

col1, col2, col3 = st.columns(3)
with col1:
    market_name = st.selectbox(label="市場名", options=df["市場名"].unique())
with col2:
    item_name = st.selectbox(label="品目名", options=df["品目名"].unique())
with col3:
    production_area = st.selectbox(label="産地名", options=df["産地名"].unique())

database = "sampledb.mysample111"
s3_output = "s3://mysample1111/Unsaved/"
query = f"""
SELECT *
FROM sampledb.mysample111
WHERE "市場名" = '{market_name}'
AND "品目名" = '{item_name}'
AND "産地名" = '{production_area}';
"""
st.write("クエリ")
st.info(query)

if st.button(label="クエリ実行"):
    # クエリの実行
    response = client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": s3_output},
    )
    query_execution_id = response["QueryExecutionId"]
    # クエリの実行状態をチェック
    wait_for_query_to_complete(client, query_execution_id)

    result = client.get_query_results(QueryExecutionId=query_execution_id)
    rows = result["ResultSet"]["Rows"]

    # カラム名を取得
    column_info = rows[0]["Data"]
    columns = [col["VarCharValue"] for col in column_info]

    # データを取得
    data = []
    for row in rows[1:]:
        data.append([col.get("VarCharValue", None) for col in row["Data"]])

    # Pandasデータフレームに変換
    df = pd.DataFrame(data, columns=columns)

    # 不要なインデックス列が存在する場合は削除
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])

    st.dataframe(df)
