# TBPTT length sweep — 2026-07-11

> Historical sweep using the earlier single-random-chunk method. It is retained for reproducibility and superseded by [segment-length-100-250-500-experiment.md](segment-length-100-250-500-experiment.md) for current decisions.

打ち切り長の最終決定を目的とせず、候補を絞るための320更新スイープ。全条件でseed 0、
batch 8、同じ初期重み、同じ256 train / 64 eval系列、compiled training stepを使用した。
評価は2,000フレーム全体で行った。

## Results

| chunk | duration | updates/s | frames/s | first 32 train | last 32 train | eval total | eval gain | eval VAD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 (1 s) | 11.53 s | 27.743 | 22,194 | 0.436106 | 0.241134 | 0.229661 | 0.229321 | 0.340618 |
| 200 (2 s) | 29.39 s | 10.887 | 17,418 | 0.443506 | 0.232190 | 0.208147 | 0.207846 | 0.300382 |
| 400 (4 s) | 93.28 s | 3.431 | 10,978 | 0.426948 | 0.220189 | 0.203946 | 0.203677 | 0.268257 |
| 800 (8 s) | 313.26 s | 1.022 | 6,538 | 0.418393 | 0.204904 | 0.195729 | 0.195468 | 0.261367 |

全条件で初期eval total lossは0.563211。NaN / Infなし、出力は0〜1内、SafeTensors
再読込後の評価値は一致した。

## Interpretation

- 100→200: 2.55倍の時間でeval total lossが9.4%改善。VADの改善も大きい。
- 200→400: 3.17倍の時間でeval total lossが2.0%改善。VAD lossは10.7%改善。
- 400→800: 3.36倍の時間でeval total lossが4.0%改善。gain改善が主で、VAD改善は2.6%。
- chunkが長いほどGPU上のframes/sが低下する。MLX 0.32のPython展開GRUでは、長い静的
  グラフのcompile/execution costが線形には増えない。

同じ320更新でも処理したフレーム数は異なる。100は256,000、200は512,000、400は
1,024,000、800は2,048,000フレームなので、この結果だけで長いchunkの改善を時間依存の
効果と追加データ量の効果に分離できない。長時間学習前に、候補間で総処理フレーム数または
wall-clockを揃えた比較が必要になる。

## Candidates

- **100 frames**: 高速回帰テスト専用。
- **200 frames**: 開発・ハイパーパラメータ探索用。320更新を約29秒で反復できる。
- **400 frames**: 品質と速度の中間候補。total lossの追加改善は小さいが、VAD lossが200より良い。
- **800 frames**: 本学習の暫定候補。今回の最良eval lossだが、400の3.36倍の時間が必要。

800は有力だが最終決定ではない。状態連続型TBPTT、等フレーム・等時間比較、WAV確認後に
固定する。

## Compatibility contract

状態連続型TBPTT、`mx.compile`、scheduler、同期削減、先読みは学習方法だけの変更であり、
推論モデルの構造は変更しない。次の契約を維持する。

- 入力特徴量65次元
- Conv1d × 2、GRU × 3
- GRUゲート順`reset, update, new`
- gain 32次元、VAD 1次元
- 重みshapeとSafeTensors形式
- BNNS/upstream C向け変換
- Conv/GRU状態の時系列更新

MLX、BNNS、C間の小さな数値差は許容する。bit-perfectではなく、入力・状態・出力の意味論、
フレームalignment、有限性、範囲、音質を検証する。

## Stateful TBPTT proposal

現行実装は2,000フレームからランダムな1 chunkだけを選び、hiddenをゼロ初期化する。次は
全2,000フレームを順番に処理し、状態を次chunkへ渡し、境界で`stop_gradient`する方式を
比較する。

```text
chunk 1 → state → chunk 2 → state → ... → chunk N
             ↑ 境界でstop_gradient
```

引き継ぐ状態:

| state | shape |
|---|---:|
| conv1 history | 2 × 65 |
| conv2 history | 2 × 128 |
| gru1 hidden | 256 |
| gru2 hidden | 256 |
| gru3 hidden | 256 |

kernel size 3のConvを2段通すため、過去4入力と新規K入力からK出力を得られる。Conv各層の
履歴を明示的に持てば重複loss除外は通常不要で、Cのストリーミング推論にも近づく。
chunkごとにoptimizer更新する方式と、複数chunkの勾配を蓄積して1回更新する方式は別条件で
比較する。

## Fair comparison budgets

同じ320更新でも総学習フレーム数が異なる。

| chunk | 320更新の総フレーム |
|---:|---:|
| 100 | 256,000 |
| 200 | 512,000 |
| 400 | 1,024,000 |
| 800 | 2,048,000 |

800の改善には長い勾配範囲と追加データ量の両方が含まれる。2,048,000フレームへ揃える
等フレーム比較は次の条件になる。

| chunk | updates | total frames |
|---:|---:|---:|
| 200 | 1,280 | 2,048,000 |
| 400 | 640 | 2,048,000 |
| 800 | 320 | 2,048,000 |

約313秒へ揃える等wall-clock比較の概算は200が約3,400更新、400が約1,070更新、800が
320更新。実測時は`updates/s`に加えて以下を記録する。

```text
frames/s = updates/s × chunk length × batch size
audio seconds/s = frames/s × 0.01
```

## Scheduler and execution optimization

等フレーム比較ではupdate基準のdecayは公平でないため、処理フレーム基準schedulerを
比較候補とする。

```text
processed_frames += chunk_length × batch_size
lr = initial_lr / (1 + decay_per_frame × processed_frames)
```

学習step全体の`mx.compile`とMLX arrayベースschedulerは導入済み。今後は毎更新の
`mx.eval(model.parameters(), optimizer.state, loss)`と`loss.item()`によるCPU同期を減らす。

- `mx.async_eval`でstateとlossをschedule
- `loss.item()`は10〜32更新ごとに限定
- device上で区間lossを累積
- 1 batch先読み
- 固定chunk shape
- 端数はpaddingと正しいloss maskで処理
- warm-up後だけMetal captureを検討

## Training operation proposal

- 開発確認: chunk 200、320更新
- ハイパーパラメータ探索: chunk 200
- 本番前確認: chunk 800、320〜1,000更新
- 本学習候補: chunk 800、数千〜10,000更新以上
- checkpointごとに2,000フレーム全長eval
- best checkpointはeval lossとWAVの両方で選択

800ではlearning rate `1e-3 / 5e-4`とgradient clipping有無も比較する。
