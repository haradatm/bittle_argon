# Gemini 4 Argon × MuJoCo による四足歩行ロボット (Petoi Bittle) の制御

Gemini 4 Argon と MuJoCo を組み合わせ、自然言語で四足歩行ロボット Petoi Bittle を制御する実験プロジェクトです。[MarcHesse/bittle-mujoco](https://github.com/MarcHesse/bittle-mujoco) の物理シミュレーションモデルを利用し、動作の生成・実行・確認からスキルの保存と再利用までを扱います。

## 概要

[解説ノートブック](./bittle_gemini_walkthrough.ipynb) では、次の3つのアプローチを段階的に実装します。

| パート | アプローチ | 内容 |
| --- | --- | --- |
| Part 1 | プリセット動作による制御 | OpenCat の歩容や姿勢を Gemini の Function Calling で組み合わせ、前進・旋回・伏せなどを実行します。 |
| Part 2 | 直接関節制御 | プリセットを使わず、8関節の角度、キーフレーム補間、CPG (中枢パターン生成器) によって新しい姿勢や周期運動を合成します。 |
| Part 3 | スキルの蓄積と再利用 | 姿勢・キーフレーム列・CPG パラメータを名前付きスキルとして JSON に保存し、次回以降の指示で再利用します。 |

Gemini がツールを呼び出してシミュレーションを進め、レンダリング画像をもとに動作を確認します。実行結果は動画として保存できます。ノートブックには、二足立ちや前足を振る動作、保存済みスキルを組み合わせた二足歩行を試す例も含まれます。

ここでの「学習」は、モデルの再訓練ではなく、生成した動作パラメータを保存・再利用する仕組みを指します。

## ファイル構成

- [bittle_gemini_walkthrough.ipynb](./bittle_gemini_walkthrough.ipynb): 共通セットアップ、各方式の実装解説、実行例と結果の表示。
- [bittle_gemini_control.py](./bittle_gemini_control.py): プリセット動作を利用する制御スクリプト。
- [bittle_gemini_direct.py](./bittle_gemini_direct.py): プリセットを利用しない直接関節制御スクリプト。
- [bittle_gemini_skills.py](./bittle_gemini_skills.py): スキルの保存・読み込み・再利用を行うスクリプト。

実行時には、スキル保存用の `learned_skills.json` や動画ファイルが生成されます。ノートブックのスキル学習例では、別の `notebook_learned_skills.json` を使用します。

## 実行準備

Python 環境、ノートブック実行環境、MuJoCo のレンダリングが利用できる環境、および Gemini API へのアクセスが必要です。

1. リポジトリ直下にシミュレーションモデルを配置します。

   ```bash
   git clone https://github.com/MarcHesse/bittle-mujoco.git bittle-mujoco
   ```

2. 使用する Python 環境に依存ライブラリをインストールします。

   ```bash
   python -m pip install mujoco numpy google-genai python-dotenv pillow imageio imageio-ffmpeg ipython ipykernel
   ```

3. リポジトリ直下の `.env` に API キーを設定します。API キーは Git にコミットしないでください。

   ```dotenv
   GEMINI_API_KEY=your_api_key
   ```

4. VS Code などでノートブックを開き、準備した Python 環境をカーネルに選択して、共通セットアップから順にセルを実行します。作業ディレクトリはリポジトリ直下にしてください。

モデル ID はノートブックでは `gemini-4-argon` を使用しています。利用可能なモデル ID とアクセス権を確認し、必要に応じてノートブックの `model_name` またはスクリプトの `--model` で変更してください。

## コマンドラインでの実行例

```bash
# プリセット動作を組み合わせる
python bittle_gemini_control.py --model gemini-4-argon --prompt "前に3歩進んでから伏せて" --video preset_demo.mp4

# 関節角度から動作を合成する
python bittle_gemini_direct.py --model gemini-4-argon --prompt "お座りして右前足を3回振って" --video direct_demo.mp4

# スキルを保存し、以後の実行で再利用する
python bittle_gemini_skills.py --model gemini-4-argon --prompt "お座りして右前足を3回振って" --skills-file learned_skills.json --video skills_demo.mp4
```

## 注意事項

- 対象は MuJoCo 上のシミュレーションであり、実機への接続や制御は含みません。
- 生成された動作の成功や安定性は保証されません。動画やシミュレーション状態で結果を確認してください。
- Gemini API の利用には、利用条件に応じた料金が発生する場合があります。
- シミュレーションの画像や指示は Gemini API に送信されます。機密情報を含めないでください。
