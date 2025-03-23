from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
import asyncio
from dotenv import load_dotenv

load_dotenv()

# browser = Browser(
#     config = BrowserConfig(
#          chrome_instance_path=r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"  # Windowsの場合
#     )
# )

text = "https://outlook.live.com/mail/0/ を開いて、迷惑メールフォルダに遷移して、そのメールをすべて削除してください。"
text = "https://www.microsoft365.com/launch/powerpoint?auth=1 を開いて、空白のプレゼンテーションを新規作成し、新しいスライドでタイトルとコンテンツのスライドを追加し、適当に文章と箇条書きを入れてください。"
text = "https://www.nikkei.com/nkd/company/?scode=8411&ba=1 を開いて、株価タブをクリックし、株価テーブルデータを取得してください。続いて、ニュースタブをクリックし、表示されているニュースヘッドラインを取得してください。個別の記事詳細ページへは遷移しなくてOKです。最後に、株価のデータとニュースヘッドラインの情報を用いて、アナリストレポートを生成してください。"

async def main():
    agent = Agent(
        task=text,
        llm=ChatOpenAI(model="gpt-4o-mini"),
        # browser=browser,
    )
    await agent.run()

asyncio.run(main())