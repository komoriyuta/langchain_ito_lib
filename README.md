# Ito Game - LangGraph Implementation

LangGraphを使用した**Ito**カードゲームの再利用可能な実装。

## 概要

Itoは、秘密の数字（1-100）を持つプレイヤーが、お題に基づいた単語だけでコミュニケーションし、全員が昇順でカードを出すことを目指す協力型ゲームです。

## インストール

### 仮想環境の作成とインストール（推奨）

```bash
cd ito_game_standalone
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### 依存関係

このライブラリは以下に依存しています：

- `langchain>=1.1.0` - LLMインターフェース
- `langchain-openai>=1.1.0` - OpenAI LLM
- `langchain-google-genai>=2.0.0` - Google Gemini LLM  
- `langgraph>=1.0.4` - グラフ状態管理
- `pydantic>=2.12.5` - データ検証
- `python-dotenv>=1.2.1` - 環境変数管理

## 環境設定

`.env`ファイルを作成してAPIキーを設定します：

```bash
cp .env.example .env
```

`.env`にAPIキーを設定：

```bash
# OpenAI API
OPENAI_API_KEY=your-openai-api-key-here

# または Google Gemini API
GOOGLE_API_KEY=your-gemini-api-key-here

# LLMプロバイダー選択（openai または gemini）
ITO_PROVIDER=openai

# モデル指定（任意）
ITO_MODEL=gpt-4o-mini
# ITO_SPEAKER_MODEL=gpt-4o
# ITO_ESTIMATOR_MODEL=gpt-4o-mini
# ITO_DISCUSSION_MODEL=gpt-4o-mini

# Mockモード（LLMなしでテスト）
# ITO_FORCE_MOCK=true
```

## 使い方

### 基本使用

```python
from ito_graph import create_game_graph

# 3人のAIエージェントでゲーム作成
game = create_game_graph(
    agent_ids=["Alice", "Bob", "Charlie"],
    theme="動物の大きさ",
    debug=True,
    reveal_hands=True  # デバッグ用に全員の手札を表示
)

# ゲーム実行
result = game.run(verbose=True)

print(f"ステータス: {result['status']}")
print(f"プレイされたカード: {result['played_cards']}")
print(f"ターン数: {result['turn_count']}")
```

### インタラクティブモード（人間プレイヤー）

```python
from ito_graph import create_game_graph

# 人間プレイヤー+ AIエージェント
game = create_game_graph(
    agent_ids=["Human", "Alice", "Bob"],
    theme="生き物の強さ",
    human_agent_id="Human",  # 人間プレイヤーのID
    max_turns=15
)

# 実行 - 人間プレイヤーは入力を求められます
result = game.run(verbose=True)
```

### LangGraphアプリへのアクセス

```python
from ito_graph import create_game_graph

game = create_game_graph(
    agent_ids=["X", "Y"],
    theme="乗り物の速さ"
)

# LangGraphアプリを直接操作
app = game.get_app()

# グラフ可視化（graphvizが必要）
try:
    from IPython.display import Image
    Image(app.get_graph().draw_mermaid_png())
except:
    print("可視化するにはIPythonとgraphvizをインストールしてください")
```

## ゲームフロー

LangGraphは以下のノードでゲームを管理します：

```
setup → speaking → voting
                       ↓ (PLAYがある)
                  execute_play → (終了チェック)
                       ↓
                   (まだACTIVE)
                   → voting
                       ↓ (全員WAIT)
                 wait_round → voting
```

1. **setup**: カードを配り、テーマを選択
2. **speaking**: 各エージェントがカードに応じた単語を生成
3. **voting**: 各エージェントが PLAY または WAIT を決定
4. **execute_play**: 最小のカードを持つエージェントがプレイ
5. **wait_round**: 全員WAITの場合、質問・回答フェーズ

ゲーム終了条件：
- 全員がカードを出した (SUCCESS)
- 昇順でないカードを出した (FAILED)
- 最大ターン数に達した (FAILED)

## データ構造

### GameState

```python
{
    "theme": str,              # 現在のお題
    "history": List[str],       # ゲーム履歴
    "played_cards": List[int],  # プレイされたカード
    "last_played_card": int,  # 最後にプレイされたカード
    "utterances": Dict[str, str],  # {エージェントID: 単語}
    "turn_count": int,         # 現在のターン数
    "status": Literal["ACTIVE", "FAILED", "SUCCESS"],  # ゲーム状態
    "deck": List[int],         # 残りのデッキ
    "agents": List[str],       # エージェントIDリスト
    "hands": Dict[str, int],  # {エージェントID: カード数字}
    "votes": Dict[str, Literal["PLAY", "WAIT"]],  # 投票結果
    "finished_agents": List[str],  # プレイ完了したエージェント
    "speaker_reasonings": Dict[str, str],  # 発話の理由
    "estimator_thoughts": Dict[str, str],  # 行動の思考
    "debug": bool,
    "reveal": bool,
    "theme_override": str,
}
```

## GRPOフレームワーク統合

この実装はGRPOトレーニングに最適化されています：

### 特徴

1. **LangGraphベース**: 状態管理・ノード間の遷移が自動化
2. **完全な状態アクセス**: `game.run(verbose=False)` でプログラム的制御
3. **グラフ可視化**: `app.get_graph().draw_mermaid_png()` でフロー可視化
4. **再利用可能**: エージェント関数を個別に置き換え可能

### GRPOトレーニング例

```python
from ito_graph import create_game_graph

# カスタムポリシーモデル（GRPOトレーニング済み）
policy_model = YourGRPOModel(...)

# エージェント関数をオーバーライド
from agents import speaker, estimator, discussion

speaker.set_speaker_llm(policy_model)
estimator.set_estimator_llm(policy_model)
discussion.set_discussion_llm(policy_model)

# ゲーム実行（出力なしでトレーニング用）
game = create_game_graph(
    agent_ids=["Agent1", "Agent2"],
    theme="動物の大きさ"
)

result = game.run(verbose=False)

# 報酬信号を抽出
reward = 1.0 if result["status"] == "SUCCESS" else -1.0

# 状態履歴から学習データを構築
# history = result["history"]
# hands = result["hands"]
# played_cards = result["played_cards"]
```

### トレーニングループの例

```python
import os

# 複数ゲームでトレーニング
for episode in range(100):
    # グラフ作成
    game = create_game_graph(
        agent_ids=[f"Agent_{i}" for i in range(4)],
        theme=random.choice([
            "動物の大きさ",
            "食べ物の辛さ",
            "キャラの強さ"
        ])
    )
    
    # 実行
    result = game.run(verbose=False)
    
    # 報酬計算
    if result["status"] == "SUCCESS":
        reward = 1.0
    elif result["status"] == "FAILED":
        # どれだけ進めたかで部分的な報酬
        reward = len(result["played_cards"]) / len(result["agents"]) - 0.5
    else:
        reward = -1.0
    
    # GRPO更新
    grpo_update(reward, result)
    
    print(f"Episode {episode}: Status={result['status']}, Reward={reward:.2f}")
```

## API リファレンス

### `create_game_graph()`

```python
def create_game_graph(
    agent_ids: List[str],
    human_agent_id: Optional[str] = None,
    theme: Optional[str] = None,
    max_turns: int = 20,
    debug: bool = False,
    reveal_hands: bool = False,
) -> ItoGameGraph
```

**パラメータ:**
- `agent_ids`: エージェントIDのリスト
- `human_agent_id`: インタラクティブモードの人間プレイヤーID（任意）
- `theme`: お題の固定（省略時はランダム）
- `max_turns`: 停滞と見なす最大ターン数
- `debug`: デバッグ出力を有効化
- `reveal_hands`: 全員の手札を表示（デバッグ用）

### `ItoGameGraph`

```python
class ItoGameGraph:
    def run(self, initial_state: Optional[GameState] = None, verbose: bool = True) -> GameState
    def get_app(self) -> CompiledGraph
```

## テスト

```bash
# テスト実行
python3 -m venv venv
source venv/bin/activate
pip install -e .
python test.py
```

## 使用例

```bash
# Mockモードでの基本例
source venv/bin/activate
python example.py basic

# インタラクティブモード
python example.py interactive

# 履歴分析
python example.py history
```

## ディレクトリ構成

```
ito_game_standalone/
├── models/          # データモデル・プロンプト
│   ├── schemas.py   # GameState定義
│   ├── prompts.py   # LLMプロンプト
│   └── themes.py    # テーマリスト
├── agents/          # エージェント実装
│   ├── speaker.py   # 単語生成
│   ├── estimator.py  # 行動決定
│   └── discussion.py # 質問・回答生成
├── utils/           # ユーティリティ
│   ├── deck.py      # カードデッキ管理
│   ├── llm.py       # LLMファクトリ
│   └── parsing.py   # JSONパース
├── ito_graph.py     # LangGraphベースのメイン実装
├── ito_game.py     # 以前の実装（非推奨）
├── test.py          # テストスイート
├── example.py       # 使用例
├── pyproject.toml   # 依存関係
├── .env.example     # 環境変数テンプレート
└── README.md        # このファイル
```

## 注意点

1. **LangGraph必須**: この実装はLangGraphに依存しています。GRPO統合などでLangGraphの利点が必要ない場合、`ito_game.py`のシンプルな実装を使用できます。

2. **Pythonバージョン**: Python 3.10以上が必要です。Python 3.14ではPydantic V1の互換性警告が出る場合がありますが、動作に影響しません。

3. **APIキー**: OpenAIまたはGoogle APIキーが必要です（Mockモードを除く）。

## ライセンス

MIT
