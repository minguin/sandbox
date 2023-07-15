# %%
# ライブラリインポート
import os

import pandas as pd
from sklearn.model_selection import train_test_split

import streamlit as st
import streamlit.components.v1 as components

import mlflow

import pygwalker as pyg
import sweetviz as sv

from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.time_series import TSForecastingExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.anomaly import AnomalyExperiment

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.multimodal import MultiModalPredictor

import h2o
# %%
# 各種キャッシュ関数
@st.cache_data
def get_sweetviz_result(df_data):
    report = sv.analyze(df_data, pairwise_analysis="off")
    report.show_html("eda.html", open_browser=False, layout="widescreen", scale=0.9)
    with open('eda.html', encoding='utf-8') as f:
        html = f.read()
    return html

@st.cache_data
def get_pycaret_model_result(_s):
    best = _s.compare_models()
    results = _s.pull()
    return best, results

@st.cache_data
def get_pycaret_predict_result(_best):
    predict = s.predict_model(_best)
    return predict

@st.cache_data
def get_deepchecks_result(_best):
    report = s.deep_check(_best, check_kwargs={})
    report.save_as_html('deepchecks.html')
    with open('deepchecks.html', encoding='utf-8') as f:
        html = f.read()
    os.remove('deepchecks.html')
    return html
# %%
# ページ全体の設定
st.set_page_config(layout="wide")
st.title('AutoML playground')

# 初期セッション変数の保存
if 'init' not in st.session_state:
    st.session_state['init'] = True
    st.session_state['index'] = get_data('index')
    st.session_state['data_list'] = {}
    st.session_state['data_list']['iris'] = get_data('iris')
    st.session_state['analysis_list'] = {}
    st.session_state['plot_model'] = pd.read_csv('plot_model.csv', encoding='cp932')
# %%
# サイドバーの設定
st.sidebar.header('1. データインプット')
uploaded_file = st.sidebar.file_uploader(label='ファイルアップロード', label_visibility='collapsed')
if uploaded_file is not None:
    st.session_state['data_list'][uploaded_file.name] = pd.read_csv(uploaded_file)

st.sidebar.header('2. サンプルデータを追加（任意）')
if st.sidebar.button(label='詳細を確認'):
    st.write(st.session_state['index'])
sample_data = st.sidebar.selectbox(
    label='サンプルデータを選択',
    options=list(st.session_state['index']['Dataset']), 
    label_visibility='collapsed',
    )
if st.sidebar.button(label='追加'):
    st.session_state['data_list'][sample_data] = get_data(sample_data)

st.sidebar.header('3. 分析対象データを選択')
select_data = st.sidebar.radio(
    label='分析対象データを選択',
    options=list(st.session_state['data_list'].keys()),
    horizontal=True,
    label_visibility='collapsed',
    )

st.sidebar.download_button(
    label="データダウンロード（option）",
    data=st.session_state['data_list'][select_data].to_csv(index=False).encode('utf-8'),
    file_name=f"{select_data}.csv",
    mime="text/csv",
    key='download-csv',
)
if st.sidebar.button(label='分析開始'):
    st.session_state['analysis_list'][select_data] = select_data
# %%
# ページ全体の設定
if select_data in st.session_state['analysis_list'].keys():
    df_data = st.session_state['data_list'][select_data]
    # 変数インプット
    set_col1, set_col2, set_col3, set_col4 = st.columns(4)
    with set_col1:
        target_col = st.selectbox(label="target列を選択",options=df_data.columns)
    with set_col2:
        ml_task = st.radio(
            label='タスクを選択',
            options=['Classification','Regression','Time Series','Clustering','Anomaly Detection'],
            horizontal=True
            )
    with set_col3:
        train_size = st.number_input(label='学習データの割合', min_value=0.0, max_value=1.0, value=0.7)
    with set_col4:
        time_limit = st.number_input(label='学習時間の上限（単位：分。0の場合は無制限）', min_value=0)
        if time_limit==0:
            time_limit = None
        st.write(time_limit)
    # タブメニュー
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🕵‍♂️PyGWalker(EDA)", 
        "🕵‍♂️Sweetviz(EDA)", 
        "👨🏻‍💻PyCaret(AutoML)",
        "👨🏻‍💻AutoGluon(AutoML)",
        "👨🏻‍💻H2O(AutoML)",
        "👨🏻‍💻auto-sklearn(AutoML)",
        ])

    # PyGWalker
    with tab1:
        if st.button("PyGWalker"):
            st.session_state['PyGWalker'] = True
        if 'PyGWalker' in st.session_state:
            pyg.walk(df_data, env='Streamlit')

    # Sweetviz
    with tab2:
        if st.button("Sweetviz"):
            st.session_state['Sweetviz'] = get_sweetviz_result(df_data)
        if 'Sweetviz' in st.session_state:
            components.html(st.session_state['Sweetviz'], height=800, scrolling=True)

    # PyCaret
    with tab3:
        # 変数インプット
        set_col1, set_col2, set_col3 = st.columns(3)
        with set_col1:
            fold_strategy = st.radio(label='CVを選択', options=['kfold','groupkfold','timeseries'], horizontal=True)
            if fold_strategy=='groupkfold':
                fold_groups = st.selectbox(label="groups列を選択", options=df_data.columns)
            else:
                fold_groups = None
        with set_col2:
            fold = st.number_input(label='Fold数を選択', min_value=2, value=5)
        with set_col3:
            fold_shuffle = st.radio(label='CVのシャッフルを行うか', options=[True,False], horizontal=True)

        # 計算実行
        if st.button('計算実行', key='PyCaret_計算実行'):
            with st.spinner('PyCaret実行中…'):
                # タスクの選択
                if ml_task=='Classification':
                    s = ClassificationExperiment()
                if ml_task=='Regression':
                    s = RegressionExperiment()
                if ml_task=='Time Series':
                    s = TSForecastingExperiment()
                if ml_task=='ClusteringExperiment':
                    s = ClusteringExperiment()
                if ml_task=='Anomaly Detection':
                    s = AnomalyExperiment()
                    
                # セットアップ
                s.setup(
                    data = df_data,
                    target = target_col, 
                    train_size = train_size,
                    fold_strategy = fold_strategy,
                    fold = fold,
                    fold_shuffle = fold_shuffle,
                    fold_groups = fold_groups,
                    html=False,
                    session_id=0,
                    log_experiment=True, 
                    experiment_name=f'{select_data}_pycaret',
                    log_plots=True,
                    log_profile=True,
                    log_data=True,
                    profile=True,
                    )
                # モデルの比較
                best = s.compare_models(budget_time=time_limit)
                results = s.pull()

                # セッション変数の保存
                st.session_state['pycaret_setup'] = s
                st.session_state['best'] = best
                st.session_state['results'] = results

        if 'pycaret_setup' in st.session_state: 
            # セッション変数の読込
            s = st.session_state['pycaret_setup']
            best = st.session_state['best']
            results = st.session_state['results']

            # 結果出力
            pycaret_tab1, pycaret_tab2, pycaret_tab3, pycaret_tab4, pycaret_tab5, pycaret_tab6 = \
            st.tabs([
                '📋️結果一覧',
                '📈予測結果',
                '📊結果可視化',
                '🔀MLflow',
                '👨‍🏫Deepchecks(モデルのテスト)',
                '📰ExplainerDashboard(XAI)'
                ])
            with pycaret_tab1:
                st.write(results)
            with pycaret_tab2:
                st.write('予測結果（最右列のprediction_label&prediction_score）')
                st.write(get_pycaret_predict_result(best))
            with pycaret_tab3:
                plot_model_list = st.session_state['plot_model']
                plot_list = plot_model_list.loc[plot_model_list['Modules']==ml_task,'ID'].values
                col1, col2 = st.columns([2,1])
                with col1:
                    plot_contents = st.radio(label='可視化項目を選択', options=plot_list, horizontal=True)
                    st.write(
                        plot_model_list.loc[
                            (plot_model_list['Modules']==ml_task)&
                            (plot_model_list['ID']==plot_contents),'Name'].values[0]
                            )
                with col2:
                    try:
                        s.plot_model(best, plot = plot_contents, display_format="streamlit")
                    except Exception as e:
                        st.write(e)
            with pycaret_tab4:
                st.write("クリックで別窓表示→[🔀MLflow](http://localhost:5000/)")
                components.iframe("http://localhost:5000", width=None, height=800, scrolling=True)
            with pycaret_tab5:
                if st.button("Deepchecks"):
                    st.session_state['Deepchecks'] = get_deepchecks_result(best)
                if 'Deepchecks' in st.session_state:
                    components.html(st.session_state['Deepchecks'], height=600, scrolling=True)
            with pycaret_tab6:
                # ExplainerDashboardは随時サーバーが立ち上がってしまうので、ボタンクリック時のみ表示する
                # TODO:shutdown実装
                st.write("ExplainerDashboardを起動してからクリックで別窓表示→[📰ExplainerDashboard](http://localhost:8050/)")
                if st.button('ExplainerDashboardの起動'):
                    components.iframe("http://localhost:8050", width=None, height=800, scrolling=True)
                    s.dashboard(best)

    # AutoGluon
    with tab4:
        # 変数インプット
        set_col1, set_col2, set_col3 = st.columns(3)
        with set_col1:
            select_presets = st.radio(
                label='presetsを選択',
                options=['best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment', 'interpretable', 'ignore_text'],
                index=3, # medium_quality
                horizontal=True
                )
        with set_col2:
            tabular_or_multimodel = st.radio(
                label='扱うデータ種別を選択',
                options=['表形式','マルチモーダル'],
                horizontal=True
                )
        with set_col3:
            if ml_task=='Classification':
                ml_tasks_autogluon = st.radio(label='分類タスクを選択',options=['binary','multiclass'],horizontal=True)
            if ml_task=='Regression':
                ml_tasks_autogluon = 'regression'
        # 計算実行
        if st.button(label='計算実行', key='AutoGluon_計算実行'):
            if time_limit is not None:
                time_limit = int(time_limit*60)
            with st.spinner('AutoGluon実行中…'):
                # こちらのmlflowは参考程度（autologだし、一部のモデルのみ）
                mlflow.set_experiment(f'{select_data}_autogluon')
                mlflow.autolog()

                train, test = train_test_split(df_data, train_size=train_size, shuffle=True, random_state=0)
                train_data = TabularDataset(train)
                if tabular_or_multimodel=='表形式':
                    predictor = TabularPredictor(label=target_col, problem_type=ml_tasks_autogluon).fit(train_data, presets=select_presets, time_limit=time_limit)
                if tabular_or_multimodel=='マルチモーダル':
                    predictor = MultiModalPredictor(label=target_col, problem_type=ml_tasks_autogluon).fit(train_data, presets=select_presets, time_limit=time_limit)
                test_data = TabularDataset(test)
                y_pred = predictor.predict(test_data.drop(columns=[target_col]))
                summary = predictor.fit_summary()
                    
                st.session_state['autogluon_predictor'] = predictor
                st.session_state['autogluon_y_pred'] = y_pred
                st.session_state['autogluon_summary'] = summary
                st.session_state['autogluon_test_data'] = test_data

                if tabular_or_multimodel=='表形式':
                    train_leaderboard = predictor.leaderboard()
                    test_leaderboard = predictor.leaderboard(test_data)
                    st.session_state['train_leaderboard'] = train_leaderboard
                    st.session_state['test_leaderboard'] = test_leaderboard
                
        if 'autogluon_predictor' in st.session_state: 
            # セッション変数の読込
            predictor = st.session_state['autogluon_predictor']
            y_pred = st.session_state['autogluon_y_pred']
            summary = st.session_state['autogluon_summary']
            test_data = st.session_state['autogluon_test_data']

            # 結果出力
            autogluon_tab1, autogluon_tab2 = st.tabs(['📋️結果一覧','📈予測結果'])
            with autogluon_tab1:
                if tabular_or_multimodel=='表形式':
                    st.write(f'評価指標：{predictor.eval_metric}')
                    train_leaderboard = st.session_state['train_leaderboard']
                    st.write(train_leaderboard)
                st.write(summary)
            with autogluon_tab2:
                st.write('予測結果（最右列のy_pred）')
                st.dataframe(test_data.assign(y_pred=y_pred))
                if tabular_or_multimodel=='表形式':
                    test_leaderboard = st.session_state['test_leaderboard']
                    st.write('（参考）testデータの結果')
                    st.write(test_leaderboard)
            
            mlflow.autolog(disable=True)
            
    # H2O
    with tab5:
        st.write("H2O Flowを起動してからクリック→[💧H2O Flow](http://localhost:54321/)")
        if st.button(label='H2O Flowの起動'):
            # TODO:要デコードエラー対応
            try:
                h2o.init(bind_to_localhost=False)
            except:
                h2o.init(bind_to_localhost=False)            
        if st.button(label='H2O Flowの終了'):
            h2o.shutdown()

    # # auto-sklearn
    # # TODO:依存関係
    # with tab6:
    #     # 計算実行
    #     if st.button(label='計算実行', key='auto-sklearn_計算実行'):
    #         with st.spinner('auto-sklearn実行中…'):
    #             # タスクの選択
    #             import autosklearn.classification
    #             import autosklearn.regression
    #             train, test = train_test_split(df_data, train_size=train_size, shuffle=True, random_state=0)
    #             if ml_task=='Classification':
    #                 automl = autosklearn.classification.AutoSklearnClassifier()
    #             elif ml_task=='Regression':
    #                 automl = autosklearn.regression.AutoSklearnRegressor()     
    #             else:
    #                 st.write(f"{ml_task}は対応していません。")           
    #             automl.fit(train.drop(target_col,axis=1), train[target_col])
                
    #         # 結果出力
    #         auto_sklearn_tab1, auto_sklearn_tab2 = st.tabs(['📋️結果一覧','📈予測結果'])
    #         with auto_sklearn_tab1:
    #             st.write('trainデータの結果')
    #             st.write(automl.leaderboard())
    #             st.write('testデータの結果')
    #             st.write(automl.show_models())
    #         with auto_sklearn_tab2:
    #             predictions = automl.predict(test.drop(target_col,axis=1))
    #             st.dataframe(test.assign(y_pred=y_pred))