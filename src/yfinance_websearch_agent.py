"""
Smolagents株価アナリストレポート生成ツール

このスクリプトは、HuggingfaceのSmolagentsを使用して、特定の企業の株価データを
yfinanceから取得し、Web検索ツールを使用して最新の情報を収集し、
それらの情報に基づいて包括的なアナリストレポートを生成します。
"""

import os
import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool

# 必要なAPI_KEYを設定（HuggingFace APIキーを使用する場合）
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "あなたのHuggingFace APIキー"

class StockAnalystReportGenerator:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        StockAnalystReportGeneratorの初期化
        
        Args:
            model_id: 使用するHuggingFaceのモデルID
        """
        # カスタムツールの定義
        @tool
        def get_stock_data(ticker: str, period: str = "6mo") -> str:
            """特定の株式ティッカーの株価データを取得します。
            
            Args:
                ticker: 株式ティッカーシンボル（例：AAPL, MSFT）
                period: データ期間（例：1d, 1mo, 6mo, 1y, 5y）
            
            Returns:
                株価データの要約文字列
            """
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                # 基本的な統計情報を計算
                summary = {
                    "最新終値": data['Close'][-1],
                    "期間始値": data['Close'][0],
                    "変化率": ((data['Close'][-1] / data['Close'][0]) - 1) * 100,
                    "期間高値": data['High'].max(),
                    "期間安値": data['Low'].min(),
                    "平均出来高": data['Volume'].mean(),
                }
                
                # グラフの保存パスを作成
                graph_path = f"{ticker}_price_chart.png"
                
                # 株価チャートを作成して保存
                plt.figure(figsize=(10, 6))
                plt.plot(data['Close'])
                plt.title(f"{ticker} Stock Price - {period}")
                plt.xlabel("Date")
                plt.ylabel("Price (USD)")
                plt.grid(True)
                plt.savefig(graph_path)
                plt.close()
                
                return f"""
                株価データ要約 ({ticker}):
                期間: {period}
                最新日付: {data.index[-1].strftime('%Y-%m-%d')}
                最新終値: ${summary['最新終値']:.2f}
                期間始値: ${summary['期間始値']:.2f}
                変化率: {summary['変化率']:.2f}%
                期間高値: ${summary['期間高値']:.2f}
                期間安値: ${summary['期間安値']:.2f}
                平均出来高: {int(summary['平均出来高']):,}株
                
                株価チャートは '{graph_path}' に保存されました。
                """
            except Exception as e:
                return f"エラー: 株価データの取得に失敗しました - {str(e)}"

        @tool
        def get_stock_news(ticker: str, limit: int = 5) -> str:
            """特定の株式ティッカーに関するニュースを取得します。
            
            Args:
                ticker: 株式ティッカーシンボル（例：AAPL, MSFT）
                limit: 取得するニュース記事の最大数
            
            Returns:
                ニュース記事の要約文字列
            """
            try:
                stock = yf.Ticker(ticker)
                news = stock.news[:limit]
                
                if not news:
                    return f"最近の{ticker}に関するニュースは見つかりませんでした。"
                
                news_summary = [
                    f"タイトル: {item['title']}\n"
                    f"発行日: {datetime.datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"出典: {item['publisher']}\n"
                    f"リンク: {item['link']}\n"
                    for item in news
                ]
                
                return f"最新ニュース ({ticker}):\n\n" + "\n".join(news_summary)
            except Exception as e:
                return f"エラー: ニュースの取得に失敗しました - {str(e)}"

        @tool
        def get_stock_info(ticker: str) -> str:
            """特定の株式ティッカーの企業情報を取得します。
            
            Args:
                ticker: 株式ティッカーシンボル（例：AAPL, MSFT）
            
            Returns:
                企業情報の要約文字列
            """
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # 基本的な企業情報を抽出
                company_info = {
                    "会社名": info.get("longName", "不明"),
                    "ティッカー": ticker,
                    "セクター": info.get("sector", "不明"),
                    "業種": info.get("industry", "不明"),
                    "時価総額": info.get("marketCap", "不明"),
                    "従業員数": info.get("fullTimeEmployees", "不明"),
                    "国": info.get("country", "不明"),
                    "通貨": info.get("currency", "不明"),
                    "ウェブサイト": info.get("website", "不明"),
                    "事業概要": info.get("longBusinessSummary", "情報なし")
                }
                
                # 財務指標を抽出
                financial_metrics = {
                    "PER": info.get("trailingPE", "不明"),
                    "PBR": info.get("priceToBook", "不明"),
                    "配当利回り (%)": info.get("dividendYield", "不明"),
                    "52週高値": info.get("fiftyTwoWeekHigh", "不明"),
                    "52週安値": info.get("fiftyTwoWeekLow", "不明"),
                    "平均出来高 (10日)": info.get("averageVolume10days", "不明"),
                    "ベータ": info.get("beta", "不明"),
                }
                
                # 時価総額を読みやすい形式に変換
                if isinstance(company_info["時価総額"], (int, float)):
                    if company_info["時価総額"] >= 1e12:
                        company_info["時価総額"] = f"${company_info['時価総額'] / 1e12:.2f}兆"
                    elif company_info["時価総額"] >= 1e9:
                        company_info["時価総額"] = f"${company_info['時価総額'] / 1e9:.2f}十億"
                    elif company_info["時価総額"] >= 1e6:
                        company_info["時価総額"] = f"${company_info['時価総額'] / 1e6:.2f}百万"
                
                # PERとPBRを読みやすい形式に変換
                if isinstance(financial_metrics["PER"], (int, float)):
                    financial_metrics["PER"] = f"{financial_metrics['PER']:.2f}倍"
                if isinstance(financial_metrics["PBR"], (int, float)):
                    financial_metrics["PBR"] = f"{financial_metrics['PBR']:.2f}倍"
                
                # 配当利回りを読みやすい形式に変換
                if isinstance(financial_metrics["配当利回り (%)"], (int, float)):
                    financial_metrics["配当利回り (%)"] = f"{financial_metrics['配当利回り (%)'] * 100:.2f}%"
                
                # データをフォーマット
                company_info_str = "\n".join([f"{k}: {v}" for k, v in company_info.items() if k != "事業概要"])
                financial_str = "\n".join([f"{k}: {v}" for k, v in financial_metrics.items()])
                business_summary = f"事業概要: {company_info['事業概要']}"
                
                return f"""
                企業情報 ({ticker}):
                
                【基本情報】
                {company_info_str}
                
                【財務指標】
                {financial_str}
                
                【事業概要】
                {business_summary}
                """
            except Exception as e:
                return f"エラー: 企業情報の取得に失敗しました - {str(e)}"

        @tool
        def get_stock_financials(ticker: str) -> str:
            """特定の株式ティッカーの財務情報を取得します。
            
            Args:
                ticker: 株式ティッカーシンボル（例：AAPL, MSFT）
            
            Returns:
                財務情報の要約文字列
            """
            try:
                stock = yf.Ticker(ticker)
                
                # 四半期財務情報を取得
                quarterly_financials = stock.quarterly_financials
                
                # 年間財務情報を取得
                annual_financials = stock.financials
                
                # 四半期と年間のキーメトリクスを取得
                quarterly_balance_sheet = stock.quarterly_balance_sheet
                annual_balance_sheet = stock.balance_sheet
                
                quarterly_cashflow = stock.quarterly_cashflow
                annual_cashflow = stock.cashflow
                
                # 直近の四半期財務情報をフォーマット
                if not quarterly_financials.empty:
                    latest_quarter = quarterly_financials.columns[0]
                    quarter_fin_str = f"直近四半期 ({latest_quarter.strftime('%Y-%m-%d')}):\n"
                    
                    # 重要な指標を選択
                    key_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
                    for metric in key_metrics:
                        if metric in quarterly_financials.index:
                            value = quarterly_financials.loc[metric, latest_quarter]
                            if isinstance(value, (int, float)):
                                value_str = f"${value / 1e6:.2f}百万" if value >= 1e6 else f"${value:.2f}"
                            else:
                                value_str = str(value)
                            quarter_fin_str += f"{metric}: {value_str}\n"
                else:
                    quarter_fin_str = "四半期財務情報は利用できません。\n"
                
                # 直近の年間財務情報をフォーマット
                if not annual_financials.empty:
                    latest_year = annual_financials.columns[0]
                    annual_fin_str = f"直近年間 ({latest_year.strftime('%Y-%m-%d')}):\n"
                    
                    # 重要な指標を選択
                    key_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
                    for metric in key_metrics:
                        if metric in annual_financials.index:
                            value = annual_financials.loc[metric, latest_year]
                            if isinstance(value, (int, float)):
                                value_str = f"${value / 1e9:.2f}十億" if value >= 1e9 else f"${value / 1e6:.2f}百万"
                            else:
                                value_str = str(value)
                            annual_fin_str += f"{metric}: {value_str}\n"
                else:
                    annual_fin_str = "年間財務情報は利用できません。\n"
                
                return f"""
                財務情報 ({ticker}):
                
                【四半期財務情報】
                {quarter_fin_str}
                
                【年間財務情報】
                {annual_fin_str}
                """
            except Exception as e:
                return f"エラー: 財務情報の取得に失敗しました - {str(e)}"
            
        # Smolagentsのエージェントを初期化
        self.agent = CodeAgent(
            tools=[
                get_stock_data,
                get_stock_news,
                get_stock_info,
                get_stock_financials,
                DuckDuckGoSearchTool()
            ],
            model=HfApiModel()
        )
    
    def gather_stock_information(self, ticker):
        """
        特定の銘柄に関する情報を収集します。
        
        Args:
            ticker: 株式ティッカーシンボル（例：AAPL, MSFT）
            
        Returns:
            収集した情報の文字列
        """
        try:
            # ステップ1: 基本的な株価と企業情報の取得
            prompt_basic = f"""
            次のツールを使用して、{ticker}株の基本情報を収集してください：
            1. get_stock_info - 企業の基本情報を取得
            2. get_stock_data - 株価チャートデータを取得
            """
            basic_info = self.agent.run(prompt_basic)
            
            # ステップ2: 財務情報とニュースの取得
            prompt_financials = f"""
            次のツールを使用して、{ticker}株の財務情報とニュースを収集してください：
            1. get_stock_financials - 最新の財務指標を取得
            2. get_stock_news - 最近のニュース記事を取得
            """
            financial_news = self.agent.run(prompt_financials)
            
            # ステップ3: Web検索を使用した最新のアナリストコメントと市場動向の取得
            prompt_search = f"""
            DuckDuckGoSearchToolを使用して、{ticker}株に関する以下の情報を検索してください：
            1. 最新のアナリストレーティングと目標株価
            2. 最近の市場動向と業界分析
            3. 将来の成長予測と潜在的なリスク
            
            検索結果から重要な情報を抽出し、要約してください。
            """
            market_analysis = self.agent.run(prompt_search)
            
            # すべての情報を結合
            all_info = f"""
            ===== {ticker}株の総合情報 =====
            
            {basic_info}
            
            {financial_news}
            
            {market_analysis}
            """
            
            return all_info
            
        except Exception as e:
            return f"エラー: 情報収集中に問題が発生しました - {str(e)}"
    
    def generate_analyst_report(self, ticker):
        """
        アナリストレポートを生成します。
        
        Args:
            ticker: 株式ティッカーシンボル（例：AAPL, MSFT）
            
        Returns:
            生成されたアナリストレポート
        """
        try:
            # 情報収集
            print(f"{ticker}の情報を収集中...")
            stock_info = self.gather_stock_information(ticker)
            
            # レポート生成
            print("アナリストレポートを生成中...")
            prompt_report = f"""
            以下の情報に基づいて、{ticker}株の包括的なアナリストレポートを作成してください：
            
            {stock_info}
            
            レポートには以下のセクションを含め、マークダウン形式で作成してください：
            
            # {ticker}株 アナリストレポート
            ## 作成日: {datetime.datetime.now().strftime('%Y-%m-%d')}
            
            ## 1. 会社概要
            [会社概要、ビジネスモデル、市場ポジションなど]
            
            ## 2. 最近の業績と株価動向
            [直近の株価パフォーマンス、チャート分析、重要な価格レベルなど]
            
            ## 3. 財務ハイライト
            [重要な財務指標、収益成長率、利益率、キャッシュフローなど]
            
            ## 4. アナリスト評価
            [ウォールストリートのアナリストの見解、目標株価、推奨事項など]
            
            ## 5. 市場動向と競合分析
            [業界動向、競合他社との比較、市場シェアなど]
            
            ## 6. リスク要因と投資機会
            [短期・長期的なリスク、潜在的な成長機会など]
            
            ## 7. 投資判断
            [推奨（買い/売り/保持）、目標株価、投資期間など]
            
            ## 付録: 最近のニュースハイライト
            [重要なニュース記事と発表のリスト]
            
            レポートは事実に基づき、バランスの取れた見解を提供するようにしてください。データと分析に基づいた明確な投資判断を含めてください。
            """
            
            report = self.agent.run(prompt_report)
            
            # レポートを保存
            report_filename = f"{ticker}_analyst_report_{datetime.datetime.now().strftime('%Y%m%d')}.md"
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(report)
            
            print(f"レポートが {report_filename} に保存されました。")
            
            return report, report_filename
            
        except Exception as e:
            error_msg = f"エラー: レポート生成中に問題が発生しました - {str(e)}"
            print(error_msg)
            return error_msg, None


# 使用例
if __name__ == "__main__":
    # ティッカーシンボルの入力を受け付ける
    ticker = input("分析したい企業のティッカーシンボルを入力してください (例: AAPL, MSFT): ").strip().upper()
    
    # レポート生成器の初期化
    # 注: モデルIDは環境によって変更可能です
    # ローカルOllamaモデルを使用する場合: "ollama/mistral"
    # Hugging Face APIを使用する場合: "mistralai/Mistral-7B-Instruct-v0.2" など
    report_generator = StockAnalystReportGenerator()
    
    # レポートの生成
    report, filename = report_generator.generate_analyst_report(ticker)
    
    # レポートの表示（オプション）
    if filename:
        print(f"\nレポートの概要:\n{report[:500]}...\n")
        print(f"完全なレポートは {filename} にあります。")