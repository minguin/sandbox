import requests
from requests import Response


def _make_query(corporate_number: str) -> str:
    return f"""
        PREFIX hj: <http://hojin-info.go.jp/ns/domain/biz/1#>
        PREFIX ic: <http://imi.go.jp/ns/core/rdf#>
        SELECT DISTINCT ?name ?location FROM <http://hojin-info.go.jp/graph/hojin>
        WHERE {{
            ?s hj:法人基本情報 ?base .
            ?base ic:ID/ic:識別値 '{corporate_number}' .
            ?base ic:名称/ic:表記 ?name .
            OPTIONAL{{?base ic:住所/ic:表記 ?location .}}
        }}
    """


def _post(query: str) -> Response:
    print(query)
    response = requests.post(
        "https://api.info.gbiz.go.jp/sparql", data={"query": query}
    )
    response.raise_for_status()
    return response


def name_and_address(corporate_number: str) -> tuple[str, str]:  # type: ignore
    response = _post(_make_query(corporate_number))
    x = response.json()["results"]["bindings"][0]
    return (x["name"]["value"], x["location"]["value"])
