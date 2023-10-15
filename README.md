# README
## CIテスト

## テンプレート
## プロンプト
Summarize this page as follows.
- What is the research?

- What's great about it compared to previous research?

- What is the key to the technique or method?

- How was it validated?

- What are the future challenges or unsolved problems?

- What paper should I read next?

このページを要約してください。
## Title

- 研究内容

- 先行研究との比較

- 技術や方法のポイント

- 検証方法

- 今後の課題

- 次に読むべき論文


## KeyGraph: Automatic Indexing by Co-occurrence Graph based on Building Construction Metaphor
- 研究内容
    - 建築メタファー(土台、屋根、柱)に基づく共起グラフによる文書の自動インデックス作成のための新しいアルゴリズム

- 先行研究との比較
    - 利点コーパスや自然言語処理ツールなどの外部知識に依存することなく、著者の要点を表すキーワードを抽出できる

- 技術や方法のポイント
    - ステミング、ストップワードの除去による文章の前処理
    - 用語の共起グラフを基本概念に対応するクラスタに分割し、複数のクラスタと強い関係を持つ用語をキーワードとして選択

- 検証方法
    - 人工知能に関する文書コレクションを対象に、提案手法の再現率と精度を従来の2つの手法（TFIDF、NGRAM）と比較し、提案手法が特別な興味や考えを持つユーザに対してより良いパフォーマンスを示すことを示した

- 今後の課題
    - TFIDFはテキストの構文構造や意味構造を考慮しないため、頻度が低かったり、他と共起していない重要な用語を見落とす可能性がある。また、キーワードの品質に影響を与える可能性のある経験的なパラメータに依存する

- 次に読むべき論文

## From technology opportunities to ideas generation via cross-cutting patent analysis: Application of generative topographic mapping and link prediction
https://www.sciencedirect.com/science/article/abs/pii/S0040162523002500
- 研究内容
    - アイデア創出を伴う技術機会分析（TOA）は、競争力を維持し、将来的に業界をリードするための重要な活動
    - しかし既存のTOAには、技術機会からアイデア創出までの道筋が不明確であること、自動化されたTOA手法と専門家ベースの手法との統合が曖昧であること、技術機会の詳細なスキームが欠如していることなど、いくつかの問題がある
    - 横断的な特許分析、生成的トポグラフィックマッピング（GTM）、リンク予測を用いて、技術機会を特定し、創造的なアイデアを生み出すための新しい体系的アプローチを提案する
    - ※GTMアルゴリズムは、高次元のデータを低次元の空間に写像する次元削減技術。自己組織化マップ（SOM）と比較すると、データの分布の正確な反映、形状やサイズにロバスト、各ノードが一意に定まる。
- 先行研究との比較
    - リンク予測を用いることで、ドメイン固有のスキームではなく、ドメイン横断的に技術機会からアイデア創出までの明確な道筋を提供している点
    - また、アイデア生成の細かい粒度と技術機会の自動識別を組み合わせている
- 技術や方法のポイント
    1. Fタームを通じて、ターゲット技術と参照技術の横断的関係を確立する
    2. 特許を収集・加工してターゲット技術と参照技術の特許-キーワードベクトル行列をそれぞれ構築する
    3. 生成的トポグラフィーマッピング(GTM)に基づいて対象技術と参照技術の二次元マップを作成し、対象技術の空白領域（技術機会）と参照技術の非空白領域（既存特許）を識別
    4. コサイン類似度とリンク予測を用いて、対象技術の空白領域と参照技術の非空白領域間にリンク関係を構築し、参照技術から推奨される特許を提示
- 検証方法
    - 天然ガスハイドレート（NGH）と炭層メタン（CBM）の両分野における開発技術に関する実証的なケーススタディによって検証
    - 参照技術から価値の高い特許にリンクさせることで、効果的に技術機会を特定し、ターゲット技術のための創造的なアイデアを生み出すことができることが示された
- 今後の課題
    - 科学論文やニュース記事など、特許以外の他のタイプのデータソースに拡張すること
    - より多くの要因や特徴を考慮することで、リンク予測手法の精度と効率を向上させること
    - アイデアの質と新規性を高めるために、アイデア生成プロセスにユーザーからのフィードバックと評価を組み込むこと
- 次に読むべき論文


## Identifying the technology convergence using patent text information: A graph convolutional networks (GCN)-based approach
https://www.sciencedirect.com/science/article/abs/pii/S0040162522000099
- 研究内容
    - 特許のテキスト情報とグラフ畳み込みネットワーク（GCN）ベースのアプローチを用いて、技術的収束を特定・監視するもの
    - ※技術的収束（テクノロジーコンバージェンス）：もとは無関係であった技術がその進歩に伴って次第に1つの機器や媒体に統合されていくこと
- 先行研究との比較
    - 特許と技術キーワードの関係を捉えることができる新しい意味論的手法を提案している点であり、引用や共同分類情報を用いた既存の手法よりも優れた性能を示す
- 技術や方法のポイント
    - GCNベースのグラフオートエンコーダー（GAE）を用いて特許と技術キーワードのベクトル表現を生成し、それを用いて技術収束を測定する指標を設計する
    - 収束レベル：特定の技術キーワードが他の技術キーワードとどれだけ近い位置にあるか
    - 成長率：特定の技術キーワードが時間的にどれだけ変化したか
    - GAEは特許と技術キーワードのノードからなるグラフ構造を学習し、特許と技術キーワードの関係性を捉えることができる
    - ※提案手法は半教師あり学習を行い、一部の特許に対して技術キーワードのラベルを与えることで、特許と技術キーワードの関係性を学習する
- 検証方法
    - 分類タスクを用いて既存のセマンティック手法と比較し、特許引用とIPCの共同分類情報に基づくいくつかの仮説を検証することで、提案手法を検証
    - 技術的収束を予測するための分類タスクを用いて、提案手法を既存のセマンティック手法と比較
    - また人工知能(AI)と分散台帳技術(DLT)の融合に関するケーススタディに適用することで、手法の有用性を実証
- 今後の課題
    - 科学論文、ニュース記事、ソーシャルメディアへの投稿など、他のタイプのデータソースにこの手法を拡張すること
    - 時間情報をモデルに組み込むこと
    - 技術収束の因果メカニズムを探ること
- 次に読むべき論文


## Topic-based technology mapping using patent data analysis: A case study of vehicle tires
https://www.sciencedirect.com/science/article/abs/pii/S0040162523002615
- 研究内容
    - タイヤ産業について、将来の自動車・交通産業における重要な役割について調査
    - 特許データ分析と機械学習を利用して、自動車タイヤ業界の技術分野とトレンドを特定し、予測
    - エアレスタイヤとインテリジェントタイヤの技術分野は将来的に非常に歓迎され、将来的にタイヤ業界の支配的かつ広範囲に使用される技術になることが示された
- 先行研究との比較
    - 世界中の大規模かつ包括的な特許データベースを使用し、タイヤ技術の世界的な革新状況を把握
    - 特許分析においてLDAを適用し、事前知識やラベルを必要とせずに特許抄録の隠れたトピックや技術分野を自動的に発見
- 技術や方法のポイント
    - 検索文字列とIPCコードを用いてDerwent Innovation Indexデータベースから過去20年間の全てのタイヤ関連特許証書を抽出し、教師なし機械学習法であるLDAを用いて、関連する技術分野を抽出
    - 2000年～2009年、2010年～2019年の2つの10年間における各技術のシェアと成長率、産業とバリューチェーンに関連するトレンドと技術指標の調査について、技術分析と将来技術分野の予測を行った
    - 技術分野の現状と成長率から、優位、新興、飽和、衰退の4グループに分類するフレームワークを提案
    - 特許動向とその影響要因の分析に基づき、タイヤ産業の将来技術を予測
- 検証方法
    - coherenceにてトピック数を決定
    - 現在の特許シェアと年平均成長率(CAGR)の2つの指標を用いて、各技術分野の特許活動の経時的傾向を評価
- 今後の課題
    - 特に特許件数が少なく不確実性の高い新興技術について、精度と信頼性をいかに向上させるか
    - 科学論文、市場レポート、SNSなど他の情報源をどのように取り入れるか
    - 環境規制、消費者の嗜好、経済状況などの外部要因がタイヤ技術の開発と普及に与える影響を考慮する方法
    - 潜在的な機会、脅威、競合他社、協力者の特定など、タイヤ業界の意思決定者により具体的で実行可能な洞察を提供する方法
- 次に読むべき論文


## Technology identification from patent texts: A novel named entity recognition method
https://www.sciencedirect.com/science/article/abs/pii/S0040162522006813
- 研究内容
    - 特許テキストに記載された技術を特定することを目的とした名前固有表現認識（NER）の利用について検討
    - ガゼッタベース、ルールベース、深層学習ベースの3つの異なるアプローチを比較
    - ※ガゼッタ：場所、組織、技術などの既知のエンティティのリスト
    - その3つを組み合わせた、特許文書から技術を識別するための新しい固有表現認識法を提案。3つのアプローチを組み合わせることで、4つの異なるIPCクラスから成る1600件の特許から4500件以上の技術を抽出し、最良の結果が得られることを実証
    - 一般的な用語や無関係な用語を避け、きめ細かく特定の技術を抽出できると主張
    - 4つの異なる特許クラスをケーススタディとして、精度、再現率、計算時間の観点から手法のパフォーマンスを評価
- 先行研究との比較
    - 特許文書から技術領域とその進化をマッピングするための文献や実践におけるギャップを埋めることが主な貢献
- 技術や方法のポイント
    - ガゼッタベースNER：WikipediaやO*NETなどのオンラインソースから既知の技術のリストを使用して、テキスト中のエンティティにマッピング
    - ルールベースNER：正規表現や形態統語的パターンを使用して、エンティティの構文的・意味的特徴に基づいて技術を抽出します。
    - 分布型NER：特許文書で事前学習したBERTモデルを微調整して、単語や句の文脈的表現を学習
- 検証方法
    - 異なる技術領域からの4つの異なる特許クラス（ボードゲーム、ホットディッピングプロセス、床構造、周波数分割多重システム）に適用し、各アプローチの精度、再現率、計算時間を測定し、互いに比較
    - 抽出された技術の例や関連性や具体性について議論（可視化には、単語埋め込みとt-SNEを用いている。分析では、各グループにおける最も頻出する技術や最も類似する技術を提示）
- 今後の課題
    - 高品質で包括的な技術のガゼッタを入手する困難さ
    - ルールベースと分布型の方法の間の精度と再現率のトレードオフ
    - 方法の特許領域や言語に対するバイアスと依存性
    - より大きく多様な特許データセットに対する方法の評価と検証の必要性
    - 技術予測や分析のための他のテキスト分析技術との方法の統合
- 次に読むべき論文
    - [TechNet: Technology semantic network based on patent data](https://www.sciencedirect.com/science/article/abs/pii/S0957417419307122)


## Wikipedia記事間の関係を考慮したTriplet Networkに基づくBERTのFine-tuning
https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_3Rin476/_article/-char/ja/
- 研究内容
    - Wikipediaの記事間の関係を考慮したTriplet Networkに基づくBERTのFine-tuningに関する論文
    - Wikipediaの記事間の関係をハイパーリンクのネットワークで定量化し、Triplet Networkの構成や損失関数を改良して、BERTをFine-tuningする方法を提案
- 先行研究との比較
    - 先行研究では、Wikipediaの記事からランダムにTripletを作成していたが、この研究では、Wikipediaの記事間の関係を考慮してTripletを作成。これにより、文の意味的な近さをより正確に反映したTripletが得られる
    - また先行研究では、Triplet Networkの損失関数は固定（距離が一定と仮定）されていたが、この研究では、記事間の関係に応じて損失関数を変化させている。これにより、文の分散表現がより細かく調整される。
- 技術や方法のポイント
    - Wikipediaの記事間の関係をハイパーリンクのネットワークで、Zerohop, Onehop, Twohop, Randomの4種類のカテゴリーに定量化
    - Zerohopは同じ記事内の文、Onehopは直接リンクされた記事内の文、Twohopは間接的にリンクされた記事内の文、Randomは無関係な記事内の文
- 検証方法
    - Wikipediaから抽出した約120万件のTripletでBERTをfine-tuning
    - 文章間類似度推定タスク（STS-B）とテーマ類似度推定タスク（wikipedia-sections-triplet）で評価
    - STS-Bでは、先行研究と同等の性能を示し、wikipedia-sections-tripletでは、先行研究よりも高い性能を示した
- 今後の課題
    - Triplet Network以外の教師なし学習法や他言語への適用
    - Wikipedia以外のコーパスや知識グラフなどの活用
- 次に読むべき論文
    - [Learning Thematic Similarity Metric Using Triplet Networks](https://aclanthology.org/P18-2009.pdf)

## Hyperbolic Relevance Matching for Neural Keyphrase Extraction
https://arxiv.org/pdf/2205.02047v1.pdf
- 研究内容
    - フレーズと文書を同じ双曲線空間で表現し、各フレーズの重要スコアとしてポアンカレ距離を介してフレーズと文書の関連性を明示的に推定する新しい双曲線マッチングモデル（HyperMatch）を設計
- 先行研究との比較
    - 双曲空間を使用するため、ユークリッド空間よりも潜在的な階層構造とそれらの間の関連性をより効果的に捉えることができる点が優れている
    - ポアンカレ距離によってフレーズと文書間の関連性を明示的にモデル化することで、キーフレーズ抽出の精度を向上させることができる
- 技術や方法のポイント
    - 指数写像を用いてユークリッド空間から双曲空間へフレーズと文書の表現をマッピングし、ポアンカレ距離を用いて各候補フレーズの重要度スコアとしてフレーズと文書の関連性を推定する
    - ハイパーボリックマージンベースのトリプレットロスを使用して、キーフレーズを抽出するモデルを最適化（関連するフレーズを関連しないフレーズよりも上位にランク付けする）
- 検証方法
    - 6つのベンチマークキーフレーズ抽出データセットで実験を行い、ユークリッド空間に基づく教師なし手法や教師あり手法を含むいくつかのベースラインとHyperMatchの性能を比較することで検証
    - 最近のBERTベースのキーフレーズ抽出モデルよりも良い結果を得た
    - 異なるドメインや言語間で頑健な性能を示した
    - 特にゼロショットデータセットにおいて、ベースラインをアウトパフォーム
- 今後の課題
    - フレーズや文書の潜在的な階層構造を明示的にモデル化するために外部知識（WordNetなど）を導入すること
    - キーフレーズ抽出のための他の双曲線演算や空間を探索すること
- 次に読むべき論文
    - [Probing BERT in Hyperbolic Spaces](https://arxiv.org/abs/2104.03869)


## Technical Phrase Extraction for Patent Mining: A Multi-level Approach
http://home.ustc.edu.cn/~wuhanhan/papers/yeliu2020.pdf
- 研究内容
    - 特許文書から技術フレーズを抽出する教師なし手法を提案し、技術的な角度から特許を要約して表現すること
- 先行研究との比較
    - 技術フレーズと特許文書のマルチレベル構造の特徴を兼ね備えており、高価な人間によるラベリングを必要としない点が優れている。また、抽出されたフレーズを表現能力の観点から評価するために、情報検索効率(IRE)と呼ばれる新しい指標を設計
- 技術や方法のポイント
    - 意味的指標と統計的指標を組み合わせ、特許文書のマルチレベル構造を活用した教師なしマルチレベル技術フレーズ抽出（UMTPE）モデルを開発
    
- 検証方法
    - 電気工学と機械工学という2つのドメインの実世界の特許データを用いて、いくつかの最先端のフレーズ抽出手法と比較することで検証、最先端のベースラインを上回った
    - 情報検索効率（IRE）と呼ばれる新しいメトリックを設計し、抽出されたフレーズを表現能力の観点から評価することで、PrecisionやRecallのような従来のメトリックを補完
- 今後の課題
    - 他のドメインや言語に拡張すること
    - より多くの外部知識ソースを組み込むこと
    - モデルの頑健性と効率を改善すること
- 次に読むべき論文

- UMTPEモデルについて
    - 特許文書の階層構造（タイトル、要約、クレーム）を利用して、レベルごとに技術的なフレーズを抽出するモデル
    - 各レベルで抽出されたフレーズは、次のレベルの抽出をガイドする役割を果たし、また既存の特許分類システム（CPCグループ）の記述も初期のガイドとして使用される
    - ※CPC：Cooperative Patent Classificationの略で、「共同特許分類」とも呼ばれ、欧州特許庁（EPO：European Patent Office)と米国特許商標庁（USPTO：United States Patent and Trademark Office）で共通に使われる特許分類体系
- UMTPEモデルの構成
    1. 候補生成：複数のフレーズ抽出ツールと名詞句抽出規則を用いて、大規模な候補フレーズプールを生成
    2. フレーズ埋め込み：スキップグラムモデルで単語の埋め込みを学習。複数語からなるフレーズの埋め込みは単語の埋め込みの平均として計算
    3. トピック生成：CPCグループや前のレベルで抽出された高信頼度のフレーズを埋め込み空間に写像し、階層的クラスタリング法でトピックを求める
    4. 候補スコア：候補フレーズ間のコサイン類似度に基づいてグラフを構築し、候補フレーズに対して意味的・統計的な指標を設計してスコア付けを行う。意味的な指標には、トピック関連性、意味関係、意味独立性など、統計的な指標には、自己長さ、影響範囲など
    5. 候補ランクと選択：NE-rankアルゴリズムで候補グラフ上でランキングを行い、上位K個の候補を技術的なフレーズとして選択。Kは文書中の文数に応じて調整。また、最も信頼度の高い候補（上位1位）は次のレベルへ送られる

- IRE（情報検索効率）について
    - Information Retrieval Efficiencyの略で、抽出したフレーズの表現能力を評価するための新しい指標
    - 精度や再現率などの従来の評価指標を補完するものであり、抽出したフレーズが特許文書の技術情報をどれだけよく表現できるかを測ることができる
- IREの計算方法
    1. ラベル付きの特許文書100件を含む1,000件の特許文書でIRタスクを実施
    2. 各抽出フレーズをクエリとして使用し、すべての文書をマッチング度に基づいてランキング
    3. そのフレーズが出てきた文書が上位10件の文書セットに含まれていれば、そのフレーズに1点を付与し、そうでなければ0点とする
    4. 各文書のスコアを抽出したフレーズの数で平均を計算
    5. 抽出したフレーズの数が少ない場合に影響を受けないように、PF（ペナルティファクター）を導入する。PFは文書中の参照技術的なフレーズの数rとモデルが抽出したフレーズの数pに応じて調整される
    6. PFで修正されたスコアを用いて、100件のラベル付き文書の平均値を求め、これをIREとする
    

## Identifying potential technological spin-offs using hierarchical information in international patent classification
https://www.sciencedirect.com/science/article/pii/S016649722030064X
- 研究内容
    - 国際特許分類(IPC)の階層情報を利用した潜在的な技術スピンオフの特定に関する研究
    - 技術的スピンオフ：ある産業で生まれた技術が予想外の分野で利用される現象
- 先行研究との比較
    - 抽象度の異なるIPCの共起ネットワークに基づくリンク予測の新しい手法を使用しており、異なる階層からの特徴が予測性能と解釈を改善できることを示している点
- 技術や方法のポイント
    - IPCの階層情報を用いて複数の技術層における仮想的な共起ネットワークを構築することと、IPC間の類似性を測定するために重み付けされた指標を用いること
    - 各特許のIPCコードを抽出する。IPC階層の異なる層に対して共起ネットワークを構築し、各IPCコードのペアに対して4つの重み付けリンク予測指数を計算
- 検証方法
    - IPCの階層情報を利用したリンク予測のモデルの性能を、分類問題と回帰問題の両方で評価
    - 炭素繊維強化プラスチック（CFRP）と機能性傾斜材料（FGM）という2つのケーススタディにこの手法を適用し、既存の手法や専門家の意見と結果を比較することで検証
    - CFRPにおいてはC08G006942とD02G000344というIPCの組み合わせが最も高い予測スコア、具体的には芳香族ポリマー繊維を含む難燃性の糸や布地や衣服の製造方法に関する特許技術
    - FGMにおいてはA61K0036とA61P001というIPCの組み合わせが最も高い予測スコア、具体的には藻類や地衣類や菌類や植物から得られる物質を含む医薬品に関する特許技術
- 今後の課題
    - CPC（Cooperative Patent Classification）やUSPC（United States Patent Classification）のような他の特許分類にこの手法を拡張すること
    - 時間的情報や動的ネットワーク分析を取り入れること
    - 技術スピンオフの因果メカニズムを探る
- 次に読むべき論文
    - [A network approach to topic models](https://www.science.org/doi/pdf/10.1126/sciadv.aaq1360)


## A network approach to topic models
https://www.science.org/doi/pdf/10.1126/sciadv.aaq1360
- 研究内容
    - トピックモデルと、複雑なネットワークにおいて類似した接続パターンを持つノードのグループを見つける手法であるコミュニティ検出法を結びつける統一的なフレームワークを提案
    - テキスト・コーパスを文書と単語の二分割ネットワークとして表現する方法と、テキストの潜在的なトピック構造を推測するためにコミュニティ検出法を適用する方法を示している。
- 先行研究との比較
    - LDAは、ディリクレプリオールの正当性の欠如、トピック数の選択不能、実際のテキストの統計的特性との不整合など、概念的・実際的な問題に悩まされていると論じている
- 技術や方法のポイント
    - テキストコーパスを文書と単語の二分割ネットワークとして表現し、確率的ブロックモデル（SBM）に基づく既存のコミュニティ検出法をトピックモデリングの実行に適応させること
    - 文書集合を文書と単語の二部グラフとして表現し、既存のコミュニティ検出法（特に、階層的確率ブロックモデル）をトピックモデルに適用することで、以下の利点が得られる：
        - データに適した事前分布を自動的に選択できる
        - トピックの数や階層構造を自動的に決定できる
        - 単語だけでなく文書もクラスタリングできる
        - データの多様性や多尺度性を捉えられる
    - SBMはグループの数や階層構造を事前に決める必要があるが、階層的なグループ構造を自動的に推定するhSBMという手法を提案。
        - hSBMは、文書と単語の両方を階層的にクラスタリングし、データに適した事前分布やトピックの数を自動的に選択する
- 検証方法
    - 手法を人工コーパスと実コーパスに適用し、トピックモデリングの最先端手法である潜在ディリクレ配分法（LDA）と比較することで検証
    - hSBMをWikipediaの記事に適用し、その結果を図示する。三つの科学分野（化学物理学、実験物理学、計算生物学）から記事を選び、文書と単語の階層的なクラスタリング結果を示す。hSBMは、分野やサブトピックに応じて記事や単語を分類し、機能語や専門用語なども自動的に識別する
- 今後の課題
    - ソーシャルメディアの投稿、画像、動画など他のタイプのテキストデータに拡張し、構文、セマンティクス、メタデータなどテキストの他の特徴を取り入れること
- 次に読むべき論文
    - [Hierarchical Stochastic Block Model for Community Detection in Multiplex Networks](https://arxiv.org/abs/1904.05330)


## Hierarchical Stochastic Block Model for Community Detection in Multiplex Networks
https://arxiv.org/pdf/1904.05330.pdf
- 研究内容
    - 複数の種類のエッジや関係を持つネットワークである多重ネットワークにおけるコミュニティ検出のための新しいベイジアンモデルに関するもの
    - このモデルは階層的確率的ブロックモデル（HSBM）と呼ばれ、ネットワークの異なる層にわたる様々なコミュニティや、それらの間の依存関係を捉えることができる
- 先行研究との比較
    - 複雑で異質なネットワーク・データを扱うことができ、データに基づいて各層のコミュニティ数を自動的に選択できる
    - また、固定ノードを前提とする既存の多くのモデルよりも現実的であり、レイヤー間でノード数やノードセットを変えることができる
- 技術や方法のポイント
    - 階層的ディリクレ過程（hierarchical Dirichlet process）を用いて層間のコミュニティ・ラベルをモデル化する
    - 確率的ブロック・モデルを用いて各層内のネットワーク構造をモデル化する
    - 階層的事前分布はレイヤー間の情報共有と依存性を可能にし、確率的ブロックモデルはレイヤー内のコミュニティ構造を捉える
- 検証方法
    - シミュレートデータと実データで検証され、単層モデルよりも優れたパフォーマンスと、実ネットワークの興味深い構造を発見する能力を実証した
    - 実データの例としては、Twitterの多重ソーシャルネットワーク、arXivの多重共著ネットワーク、ヨーロッパの多重経済ネットワークなど
- 今後の課題
    - 重み付きネットワークや有向ネットワークを扱うためにどのようにモデルを拡張するか
    - ノードの属性や共変数をどのように組み込むか
    - 非常に大規模なネットワークを扱うためにどのようにモデルを拡張するか
- 次に読むべき論文