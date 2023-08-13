# sandbox
## AutoML playground
### 概要
- 久しぶりにPyCaretを触ってみると色々出来ることが増えている気がする
- 改めて色々なEDA&AutoML&MLOpsライブラリを試してみたい
- Streamlitでサンプルデータも使いながら色々試せるものを構築

### 起動（Docker）
```
docker-compose up
```
####  Streamlit(http://localhost:8501/)
常時起動。UI
####  MLflow(http://localhost:5000/)
常時起動。PyCaretやAutogluon実行時に自動的にモデルパラメータや評価指標など記録。
####  ExplainerDashboard(http://localhost:8050/)
PyCaret実行時に起動、表示。クリックする度に立ち上がってしまう…（要修正）
####  H2O(http://localhost:54321/)
選択時に起動。

### 起動（pip）
```
pip install -r requirements.txt
python -m streamlit run app.py
# 別の所から起動(http://localhost:5000/)
mlflow ui
```

### 使い方
- プリセットデータでirisがセットされているので、そのまま分析実行を押下
- 他のサンプルデータ（テキストとか）を試す際には「詳細を確認」して概要を見て選択
- H2Oを用いる際にはデータアプロードが必要なので、「データダウンロード（option）」を押下して手元にダウンロードしておく

### 試せるライブラリ
※AutoML関係についての参考記事
https://atmarkit.itmedia.co.jp/ait/series/24323/

- 🕵‍♂️PyGWalker(EDA)…Tableau風にデータ操作できる
https://github.com/Kanaries/pygwalker

- 🕵‍♂️Sweetviz(EDA) …基礎データ統計に加えて、Train,Testデータの分布の比較みたいなのが便利。（今回の構築ではまだ使えない）
https://github.com/fbdesignpro/sweetviz

- 👨🏻‍💻PyCaret(AutoML)…回帰、分類に加えて、クラスタリング、時系列、異常検知もカバー。サンプルデータも豊富。テキスト画像は扱えない？
https://pycaret.org/

    ##### PyCaret上で使えるもの
    - 🔀MLflow…
    https://mlflow.org/

    - 👨‍🏫Deepchecks(モデルのテスト)…
    https://deepchecks.com/

    - 📰ExplainerDashboard(XAI)…
    https://explainerdashboard.readthedocs.io/en/latest/

    ##### 今回は搭載していない（別のEDAライブラリを搭載）が、本来PyCaret上で使えるもの
    - 🕵‍♂️Autoviz(EDA)…
    https://github.com/AutoViML/AutoViz

    ##### PyCaretを実行するとMLFlowのartifactに出力されるもの（要setup引数）
    - 🕵‍♂️ydata-profiling(EDA)…旧pandas-profiling。いつの間にか名前が変わってた
    https://ydata-profiling.ydata.ai/docs/master/index.html

- 👨🏻‍💻AutoGluon(AutoML)…PyCaretでは扱えない、テキスト、画像なんかも扱える。マルチモーダルなモデル構築が可能。気軽には使えるが、PyCaretほど他とのライブラリ連携はなさそう？
https://auto.gluon.ai/stable/index.html

- 👨🏻‍💻H2O(AutoML)…UI上でデータをアップロードしたり、EDAしたり。モデルの種類は相当搭載しているみたい
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/flow.html

- 👨🏻‍💻auto-sklearn(AutoML)…サンプルデータセットによるメタ学習が気になったが、他のAutoMLとの兼ね合いで断念…。Windows非対応。
https://automl.github.io/auto-sklearn/master/

## gbizinfo
https://speakerdeck.com/stakaya/2020nian-xin-ren-yan-xiu-zi-liao-nauteyankunapythonkai-fa-ru-men

追加でpoetry add pytest types-requests --devしてる