import responses

from gbizinfo.basic import _make_query, name_and_address


def test_make_query():
    assert (
        _make_query("7010401075212")
        == f"""
        PREFIX hj: <http://hojin-info.go.jp/ns/domain/biz/1#>
        PREFIX ic: <http://imi.go.jp/ns/core/rdf#>
        SELECT DISTINCT ?name ?location FROM <http://hojin-info.go.jp/graph/hojin>
        WHERE {{
            ?s hj:法人基本情報 ?base .
            ?base ic:ID/ic:識別値 '7010401075212' .
            ?base ic:名称/ic:表記 ?name .
            OPTIONAL{{?base ic:住所/ic:表記 ?location .}}
        }}
    """
    )


@responses.activate
def test_name_and_address():
    responses.add(
        responses.POST,
        "https://api.info.gbiz.go.jp/sparql",
        status=200,
        json={
            "results": {
                "bindings": [
                    {"name": {"value": "株式会社X"}, "location": {"value": "六本木3丁目"}}
                ]
            }
        },
        content_type="application/json",
    )
    assert name_and_address("7010401075212") == ("株式会社X", "六本木3丁目")
