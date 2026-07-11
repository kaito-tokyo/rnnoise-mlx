# State diagnostics — 2026-07-11

> Historical diagnostics for the earlier random/stateful checkpoints. Current training decisions are recorded in [segment-length-100-250-500-experiment.md](segment-length-100-250-500-experiment.md).

lossを順位付けに使わず、未学習、random chunk 200/800、stateful chunk 200/800の5モデルを
固定してstateを観測した。eval 64系列を集計し、先頭8系列についてreset ablationのtraceを
保存した。chunk長は100/200/400/800。

## Correctness

- 全モデル・全chunk長で全長forwardとchunk forwardのgain/VAD最大絶対誤差は0.0。
- Conv1履歴`2 × 65`、Conv2履歴`2 × 128`、GRU hidden各256を維持。
- Conv履歴resetによるConv2への直接影響は全条件で境界後offset 0〜3に限定された。
- 全指標は有限、gain/VADは0〜1内。
- stale-state診断前後で全checkpointのSHA-256は不変。

## State distribution

chunk長を変えてもresetなしのtraceは同じため、以下はchunk 200の集計。飽和率は`|h| > 0.9`。

| model | GRU1 RMS / sat | GRU2 RMS / sat | GRU3 RMS / sat |
|---|---:|---:|---:|
| untrained | 0.183 / 0.0% | 0.118 / 0.0% | 0.081 / 0.0% |
| random 200 | 0.673 / 21.9% | 0.612 / 14.9% | 0.649 / 36.7% |
| random 800 | 0.660 / 20.4% | 0.610 / 14.6% | 0.655 / 31.6% |
| stateful 200 | 0.744 / 35.4% | 0.699 / 29.3% | 0.623 / 25.2% |
| stateful 800 | 0.687 / 25.6% | 0.590 / 15.7% | 0.549 / 22.2% |

学習済みモデルではstate振幅と飽和率が大きく増える。特にstateful 200のGRU1/2とrandom
200のGRU3が高い。直ちにstate clippingは導入せず、将来のlearning rate・gradient clipping
比較で観測対象とする。

通常フレームに対するchunk境界のstate変化比は学習済みモデルで概ね0.85〜1.08。resetなし
では境界だけに不連続はなく、Conv/GRU履歴の実装が連続している。

## Reset memory horizon

chunk 800で全stateをresetし、連続実行との差がreset直後の50%・10%・1%以下になるまでの
フレーム数を測定した。`>800`は次境界までに閾値へ到達しなかったことを表す。

| model | GRU1 50/10/1% | GRU2 50/10/1% | GRU3 50/10/1% | gain 50/10/1% | VAD 50/10/1% |
|---|---:|---:|---:|---:|---:|
| untrained | 3/6/11 | 4/8/14 | 5/10/16 | 3/6/12 | 1/4/10 |
| random 200 | 10/179/>800 | 26/493/>800 | 21/>800/>800 | 4/67/733 | 5/5/>800 |
| random 800 | 14/250/>800 | 32/580/>800 | 47/696/>800 | 4/139/>800 | 7/240/>800 |
| stateful 200 | 14/197/>800 | 19/399/>800 | 13/387/>800 | 4/95/693 | 14/129/>800 |
| stateful 800 | 20/318/>800 | 45/326/>800 | 36/408/>800 | 5/140/555 | 29/251/>800 |

学習済みstateの影響は数秒残り、1%基準では8秒を超える場合が多い。短いburn-inだけで
連続stateを置き換えるのは難しい。層別resetでもGRU2/3のgain/VAD影響が長く残るため、
特定1層だけを継続する方式より、3層すべてを継続する方式を優先する。

Conv履歴resetのConv2直接影響は4フレーム以内だが、その摂動がGRUへ入るためgain/VADの
1%影響は最大242フレーム程度残った。これはalignment不具合ではなく再帰状態による伝播。

## Stale-state after one optimizer update

prefixを更新前重みで処理して得たstateと、1回更新後の重みで同じprefixを再計算したstateを
比較した。表はchunk 200。値は相対L2誤差。

| model | GRU1 | GRU2 | GRU3 | next gain | next VAD |
|---|---:|---:|---:|---:|---:|
| untrained | 58.0% | 66.6% | 73.4% | 0.58% | 0.15% |
| random 200 | 39.1% | 45.7% | 23.6% | 4.68% | 2.68% |
| random 800 | 37.1% | 44.8% | 38.8% | 12.27% | 12.97% |
| stateful 200 | 26.7% | 32.7% | 31.8% | 6.28% | 1.50% |
| stateful 800 | 46.4% | 55.3% | 35.1% | 7.06% | 9.42% |

学習済みモデルでも1更新でstateが大きく変わり、特にrandom 800では次chunk出力差が約12〜13%。
「更新前stateを更新後モデルへ渡す」現行chunk単位更新の不整合は無視できない。

## Candidate ranking

1. **同じ重みで複数chunkを処理してからoptimizer更新**。stale-state誤差が最大のシグナルで、
   状態連続型の精度悪化を直接説明できる。まず1系列全chunkでstateを連続させ、chunkごとの
   勾配を蓄積して系列末尾で1回更新する方式を比較する。
2. **stateful＋長いwarm-up/burn-in**。reset影響が数百フレーム残るため、採用するなら短い
   10〜50フレームではなく、少なくとも200〜400フレームを候補にする。warm-up区間はlossから
   除外する。
3. **3層すべてのstate継続**。部分statefulより優先する。GRU2/3の影響が長く、1層だけの継続は
   情報を大きく失う可能性が高い。
4. **ランダムchunk＋burn-in**。実装は簡単だが、8秒以内に1%へ収束しないstateがあるため、
   完全な代替ではなく比較候補に留める。

Conv reset範囲は正常なのでalignment修正は不要。次段階は候補1と、比較用の候補2/4だけを
学習実験し、同じ総処理フレームとWAVで判断する。

詳細な機械可読結果はGit管理外の`runs/state-diagnostics/summary.json`、
`state_metrics.csv`、`traces/*.npz`に保存した。
