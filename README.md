# rnnoise-mlx

Apple Silicon/macOS向けのRNNoise系ノイズ抑制実装です。学習には
[MLX](https://github.com/ml-explore/mlx)、実時間ニューラル推論にはBNNS、
特徴抽出とゲイン適用にはupstream RNNoiseのDSPを使用します。

このリポジトリで生成する配布モデルは、独自に生成した学習データから学習します。
Xiph/upstreamのモデルは検証用に限り、重み・出力・蒸留ターゲットを成果物モデルへ
混入させません。

## 状態

- `training/`: MLXモデル、RNNoise `.f32` データセット、損失、学習CLI
- `bnns/`: 1フレーム推論APIと重み読込を実装するSwift package（実行グラフは作業中）
- `upstream/`: 固定するRNNoise revisionと取り込み方針
- `models/`: MLX/BNNS間のモデル契約
- `docs/`: 再現性、評価、ライセンス方針

## 学習の開始

```sh
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/rnnoise-mlx-train data/features/train.f32 runs/smoke \
  --batch-size 8 \
  --sequence-length 2000 \
  --max-updates 320 \
  --segmented-tbptt-length 100 \
  --segmented-tbptt-state carry \
  --eval-features data/features/eval.f32
```

`features.f32` はupstream `dump_features`と同じ、1フレーム98個の
float32（特徴65、ゲイン32、VAD 1）です。

開発中のコーパス追加と音質スクリーニングにはTBPTT長100を使い、固定した評価WAVを
ユーザーが主観評価します。lossは破綻検出の補助指標であり、長さ100の結果だけで本番品質を
判断しません。有望な候補は長さ500、Conv/GRU state継続、境界`stop_gradient`、
2,000フレーム後にoptimizerを1回更新する条件で再確認します。

```sh
.venv/bin/rnnoise-mlx-train data/features/train.f32 runs/promoted-500 \
  --batch-size 8 \
  --sequence-length 2000 \
  --segmented-tbptt-length 500 \
  --segmented-tbptt-state carry \
  --eval-features data/features/eval.f32 \
  --max-updates 320
```

100/250/500とstate保持の比較は
[docs/segment-length-100-250-500-experiment.md](docs/segment-length-100-250-500-experiment.md)、
学習データ作成と320更新テストの再現手順は
[docs/test-training.md](docs/test-training.md)を参照してください。

## 生成物

`data/`、`runs/`、`experiments/`、`.venv/`、Swiftの`.build/`はGit管理しません。
SafeTensors、評価JSON、WAVなどの生成物はこれらのディレクトリへ保存してください。

## テスト

```sh
.venv/bin/python -m pytest -q
swift test --package-path bnns
```

## ライセンス

本体はBSD-3-Clauseです。upstream由来コードには原著作権表示を保持します。
詳細は [docs/licensing.md](docs/licensing.md) を参照してください。
