# MLX training benchmark — 2026-07-11

> Historical benchmark. The current policy is chunk 100 for subjective screening and chunk 500 with state carry for promoted runs. See [segment-length-100-250-500-experiment.md](segment-length-100-250-500-experiment.md).

Apple M2 Pro（32 GB）、MLX 0.32.0、Python 3.14、Metalで現行`nn.GRU`学習器を測定した。
全条件でbatch 8、320 updates、seed 0、compiled training step、AdamWを使用。trainは256系列、
evalは別の64系列を使用し、evalは2,000フレーム全長で実行した。記載時間は学習loopのみで、
初期・学習後・再読込後evalを含まない。

## Throughput and loss

| chunk | training time | updates/s | processed frames | frames/s | audio seconds/s | eval total |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 11.11 s | 28.799 | 256,000 | 23,039 | 230.39 | 0.229661 |
| 200 | 29.32 s | 10.915 | 512,000 | 17,464 | 174.64 | 0.208147 |
| 400 | 93.13 s | 3.436 | 1,024,000 | 10,995 | 109.95 | 0.203946 |
| 800 | 311.14 s | 1.028 | 2,048,000 | 6,582 | 65.82 | 0.195729 |

`frames/s`はbatchを含む。`audio seconds/s = frames/s × 0.01`で、学習が1秒あたり何秒分の
音声フレームを処理したかを表す。

## Loss details

| chunk | first 32 train | last 32 train | eval gain | eval VAD |
|---:|---:|---:|---:|---:|
| 100 | 0.436106 | 0.241134 | 0.229321 | 0.340618 |
| 200 | 0.443506 | 0.232190 | 0.207846 | 0.300382 |
| 400 | 0.426948 | 0.220189 | 0.203677 | 0.268257 |
| 800 | 0.418393 | 0.204904 | 0.195468 | 0.261367 |

全条件でNaN / Infなし、gain/VADは0〜1内、SafeTensors再読込後の評価値は一致した。

## Observations

- 100は最も高いthroughputで、回帰テストに適する。
- 200は100に対してeval total lossを9.4%改善し、約29秒で320更新できる。
- 400は200よりeval total lossを2.0%改善する一方、学習時間は3.18倍。
- 800は最良eval lossだが、400より3.34倍、200より10.61倍の時間が必要。
- chunk長が8倍になる100→800で`frames/s`は71.4%低下する。MLX 0.32のPython展開
  `nn.GRU`では、長い静的グラフのコストが線形より大きい。

同じ320更新では総処理フレームが異なるため、品質効率の最終比較ではない。100=回帰、
200=開発、400=中間、800=本学習暫定候補という用途を維持し、次は状態連続型TBPTT実装後に
等フレーム・等wall-clock条件を測定する。

## Async evaluation and one-batch prefetch

chunk 200、320更新で同期版と`mx.async_eval`＋1 batch先読み版を比較した。

| execution | time | updates/s | frames/s | eval total |
|---|---:|---:|---:|---:|
| per-step synchronization | 29.32 s | 10.915 | 17,464 | 0.208147 |
| async + prefetch 1 | 28.40 s | 11.267 | 18,027 | 0.208147 |

全320件のloss履歴と学習後評価値は一致した。全体では3.1%の時間短縮。32更新の短い測定では
約17%改善したが、長い測定ではcheckpoint保存時の同期などが残るため差が縮小した。
`mx.async_eval`は既定とし、lossのPython化は10更新ごと、データ先読みは1 batchに限定する。

## Artifacts

各条件の`training.json`とSafeTensorsはGit管理外の`runs/training-benchmark/chunk-{length}/`
に保存した。
