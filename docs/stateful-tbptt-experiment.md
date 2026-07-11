# Stateful TBPTT experiment — 2026-07-11

> Historical experiment that updated the optimizer after every chunk. The current method processes all 2,000 frames with unchanged weights and updates once; see [segment-length-100-250-500-experiment.md](segment-length-100-250-500-experiment.md).

精度とストリーミング互換性を重視し、Conv/GRU状態を2,000フレーム内のchunk間で引き継ぐ
実験を行った。実験機能は`--stateful-tbptt`で有効化し、既定のランダムchunk方式は変更しない。

## State and alignment

引き継ぐ状態はConv1入力履歴`2 × 65`、Conv2入力履歴`2 × 128`、3層のGRU hidden
各256。最初のchunkはK-4出力、以降は履歴＋K入力からK出力を生成する。教師alignmentは
最初が`[3:K-1]`、以降が`[start-1:end-1]`で、全2,000入力に対して1,996出力を一度ずつ
学習する。chunk境界では`mx.stop_gradient`を適用する。

600フレームを200 × 3へ分割したforwardと全長forwardを比較し、gain/VADとも最大絶対誤差
0.0を確認した。推論モデル、重みshape、SafeTensors、BNNS/C変換契約は変更していない。

## Results

| method | chunk | updates | frames | time | frames/s | eval total | gain | VAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random chunk | 200 | 320 | 512,000 | 29.32 s | 17,464 | 0.208147 | 0.207846 | 0.300382 |
| stateful | 200 | 320 | 512,000 | 29.37 s | 17,434 | 0.271462 | 0.271102 | 0.359500 |
| stateful | 800/800/400 | 96 | 512,000 | 82.72 s | 6,190 | 0.363985 | 0.363604 | 0.381121 |
| stateful | 800/800/400 | 320 | 1,708,800 | 263.54 s | 6,484 | 0.232768 | 0.232436 | 0.332699 |
| random chunk | 800 | 320 | 2,048,000 | 311.14 s | 6,582 | 0.195729 | 0.195468 | 0.261367 |

全条件でNaN / Infなし、出力は0〜1内、SafeTensors再読込後の評価値は一致した。

## Interpretation

状態とalignmentの実装は全長forwardと一致するが、chunkごとにoptimizer更新する今回の方式は
精度を改善しなかった。候補原因:

- 同じ系列の相関が強いchunkを連続して更新する
- 次chunkのstateは更新前の重みで計算され、モデル重みはchunk間で変化する
- random方式はepochごとに系列と区間を再抽出し、更新ごとの多様性が高い
- 800の比較では更新回数または総フレーム数も完全には同時に揃わない

状態連続型を既定にはしない。次は精度優先で、1系列の複数chunkについて勾配を蓄積し、
optimizer更新を系列単位または数chunk単位へ遅らせる方式を比較する。これは以前保留した一般的な
勾配蓄積とは分け、状態連続学習の整合性を確認するための限定実験として扱う。random 200/800を
基準に、等フレーム、等更新、WAVで判断する。
