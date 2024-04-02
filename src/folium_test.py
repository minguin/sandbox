# -*- coding: utf-8 -*-

import json
import urllib.parse

import folium
import folium.plugins
import numpy as np
import pandas as pd

# In[]Yahoo!ローカルサーチAPI
import requests
from pymongo import MongoClient
from sklearn import preprocessing

# APIキー等変数の指定
d = {}
# @マークは回避する必要がある。
d_qs = urllib.parse.urlencode(d, safe="@")
# APIのURLを得る
url = "https://map.yahooapis.jp/search/local/V1/localSearch?" + d_qs
# 実際にAPIにリクエストを送信して結果を取得する
r = requests.get(url)
# 結果はJSON形式なのでデコードする
data = json.loads(r.text)
df_all = []
print(data["ResultInfo"]["Total"])

for j in range(0, min(data["ResultInfo"]["Total"] // 100, 31)):
    print(j)
    if j >= 1:
        d["start"] = 100 * j + 1
        # @マークは回避する必要がある。
        d_qs = urllib.parse.urlencode(d, safe="@")
        # APIのURLを得る
        url = "https://map.yahooapis.jp/search/local/V1/localSearch?" + d_qs
        # 実際にAPIにリクエストを送信して結果を取得する
        r = requests.get(url)
        # 結果はJSON形式なのでデコードする
        data = json.loads(r.text)

    for i in range(0, len(data["Feature"])):
        Name = data["Feature"][i]["Name"]
        Genre = data["Feature"][i]["Property"]["Genre"][0]["Name"]
        (lng, lat) = data["Feature"][i]["Geometry"]["Coordinates"].split(",")
        CreateDate = data["Feature"][i]["Property"]["CreateDate"]
        df_all.append([Name, Genre, lat, lng, CreateDate])

df_all = pd.DataFrame(df_all)
df_all.columns = ["Name", "Genre", "lat", "lng", "CreateDate"]

# In[]
# Yahoo API
df = df_all
df[["lat", "lng"]] = df[["lat", "lng"]].astype("float64")
df_daiso = pd.read_csv(r"", engine="python")
df_daiso = df_daiso.rename(columns={"緯度": "lat", "経度": "lng", "col2": "Name"})

client = MongoClient("localhost", 27017)

# 範囲を示すmaxDistanceの単位は'度' ※１度 = 111.263km
# 2dsphereの場合は、メートルを指定するためいらない。
# KM = 1/111.263
pos = [35.681236, 139.767125]

radius = 15  # km


def get_geodata(dbname, loc):
    query = {
        "loc": {
            "$nearSphere": {
                "$geometry": {"type": "Point", "coordinates": [loc[1], loc[0]]},
                "$maxDistance": radius * 1000,
            }
        }
    }

    result = [r for r in dbname.find(query)]
    if not result:
        print("範囲内にデータはありませんでした。")
    else:
        return pd.DataFrame(result)


# メッシュ面積
def get_statistics(df_data):
    df_data = pd.concat(
        [
            df_data,
            pd.DataFrame(df_data.count()).T.rename(index={0: "count"}),
            pd.DataFrame(df_data.sum()).T.rename(index={0: "sum"}),
            pd.DataFrame(df_data.mean()).T.rename(index={0: "mean"}),
        ]
    )

    df_data.loc["調整係数"] = radius**2 * np.pi / (df_data.loc["count"] * 0.5**2)
    return df_data


# db = client.keizai.keizai
db = client.kokusei.kokusei
# db = client.syougyo.syougyo


df_estat = get_geodata(db, pos)
temp = pd.DataFrame(df_estat["loc"].tolist(), columns=["lng", "lat"])
df_estat = pd.concat([df_estat, temp], axis=1)
print(df_estat.columns)

tooltip = r"小売業計年間販売額（千万円）"
popup = r"小売業計年間販売額（千万円）"

tooltip = r"Ａ～Ｒ全産業（Ｓ公務を除く）"
popup = r"Ａ～Ｒ全産業（Ｓ公務を除く）"

tooltip = r"　人口総数"
popup = r"　人口総数"

# In[]
# cartodbpositron
m = folium.Map(
    location=pos,
    tiles="cartodbpositron",
    zoom_start=11,
)

# レイヤを追加
folium.TileLayer("openstreetmap").add_to(m)
folium.TileLayer("stamenterrain").add_to(m)
folium.TileLayer("stamentoner").add_to(m)

# minmax = preprocessing.scale(df_estat[tooltip].fillna(0))
minmax = preprocessing.minmax_scale(df_estat[tooltip].fillna(0))

for num in df_estat.index:
    E = df_estat.lng[num] + 11.25 / 3600
    W = df_estat.lng[num] - 11.25 / 3600
    S = df_estat.lat[num] - 7.5 / 3600
    N = df_estat.lat[num] + 7.5 / 3600
    upper_left = (N, W)
    upper_right = (N, E)
    lower_right = (S, E)
    lower_left = (S, W)

    color = (255, int(255 * (1 - minmax[num])), int(255 * (1 - minmax[num])))
    fill_color = "#%02X%02X%02X" % (color[0], color[1], color[2])
    line_color = "#%02X%02X%02X" % (0, 0, 0)

    edges = [upper_left, upper_right, lower_right, lower_left]

    folium.Rectangle(
        edges,
        tooltip=df_estat[tooltip][num],
        popup=df_estat[popup][num],
        color=line_color,
        fill_color=fill_color,
        fill_opacity=0.5,
        weight=0.1,
    ).add_to(m)

folium.Circle(pos, color="crimson", radius=radius * 1000).add_to(m)

#'''
for num in df.index:
    folium.CircleMarker(
        [df.lat[num], df.lng[num]],
        popup=df.Name[num],
        tooltip=df.Name[num],
        color="blue",
        radius=0.5,
    ).add_to(m)
#'''

#'''
for num in df_daiso.index:
    folium.CircleMarker(
        [df_daiso.lat[num], df_daiso.lng[num]],
        popup=df_daiso.Name[num],
        tooltip=df_daiso.Name[num],
        color="green",
        radius=0.5,
    ).add_to(m)
#'''

folium.LayerControl(collapsed=False).add_to(m)

minimap = folium.plugins.MiniMap(toggle_display=True, width=200, height=200)
m.add_child(minimap)
# m.add_child(folium.LatLngPopup())

m.save(r"index.html")
