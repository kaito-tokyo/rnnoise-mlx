# MLX test training result — 2026-07-11

Apple M2 Pro（32 GB）で、`docs/test-training.md`の256系列・320更新試験を実行した。
これはパイプライン検証用モデルであり、音質モデルとして配布しない。

## Data

| split | speech | sequences | bytes |
|---|---|---:|---:|
| train | LibriTTS-R train-clean-100 | 256 | 200,704,000 |
| eval | LibriTTS-R dev-clean | 64 | 50,176,000 |

dev-clean archive MD5は`2c1f5312914890634cc2d15783032ff3`で、OpenSLR配布一覧と一致した。
ノイズはXiph background noise v2 / foreground noise、RIRはmeasured_rirs-v2を使用した。
既存モデルの重み・推論出力・蒸留ターゲットは使用していない。

## Configuration

- MLX 0.32.0、Python 3.14、Metal
- batch 8、sequence length 2,000、10 epochs、320 updates
- AdamW、learning rate 1e-3、betas (0.8, 0.98)、gamma 0.25
- Conv1d 65→128→256、GRU 256 × 3、gain 32、VAD 1

## Result

| metric | initial | trained | reloaded |
|---|---:|---:|---:|
| total loss | 0.563442 | 0.185812 | 0.185812 |
| gain loss | 0.562722 | 0.185565 | 0.185565 |
| VAD loss | 0.719916 | 0.247472 | 0.247472 |

- first 32 update train-loss mean: 0.401861
- last 32 update train-loss mean: 0.189594（52.8%低下）
- minimum observed train loss: 0.108562
- NaN / Inf: none
- gain / VAD output range: 0〜1内
- SafeTensors save/reload: 評価値が完全一致
- 計測時間: 1,094.1秒（0.292 updates/s。現計測は最終評価・再読込評価を含むため僅かに保守的）

既存PyTorch CPU版の同じ256系列・batch 8の実測は1 epoch 57.8秒、約0.553 updates/s
（10 epochs換算578秒）。したがって現MLX実装は約1.89倍遅い。主因候補は、GRUをPythonから
2,000 step明示展開している点である。次の性能作業ではMLXのcompiled scan/RNN primitiveへ置換し、
同じ損失・重み形式を維持したまま再計測する。

## Verdict

320更新完走、train/eval loss改善、有限性、出力範囲、保存再読込の全機能条件を満たした。
速度条件は測定済みだが、現時点ではPyTorch CPU版より遅く、最適化が必要である。
