# MLX nn.GRU training result — 2026-07-11

MLX 0.32の`nn.GRU`へ3層すべてを置換した。`nn.GRU`も内部ではPythonで時系列を
展開するため、2,000フレーム全体のBPTTは高速化しなかった。学習時のみ200フレームを
固定seedで抽出するtruncated BPTTへ変更し、評価は2,000フレーム全体で実施した。

## Speed

| implementation | BPTT frames/update | updates/s | 320 updates |
|---|---:|---:|---:|
| explicit custom GRU | 2,000 | 0.292 | 1,094.1 s |
| MLX `nn.GRU` | 2,000 | 0.131 | 243.7 s / 32 updates |
| MLX `nn.GRU` + truncated BPTT | 200 | 8.214 | 39.0 s |

200フレームTBPTTの32更新比較では、学習step全体の`mx.compile`によりeagerの
7.520 updates/sから8.746 updates/sへ16.3%改善した。compileは既定で有効にする。

320更新の最終測定ではcompiled版は29.10秒、10.996 updates/sだった。eager版の
38.96秒、8.214 updates/sに対して25.3%の時間短縮、33.9%の更新速度向上となった。

最終方式の更新速度は旧テスト学習比で約28.1倍。ただし1更新の逆伝播長が1/10なので、
同一FLOP条件の比較ではない。2,000フレーム`nn.GRU`単純置換は旧実装より遅いため採用しない。

## 320-update validation

| metric | initial | trained | reloaded |
|---|---:|---:|---:|
| total loss | 0.564459 | 0.208413 | 0.208413 |
| gain loss | 0.563788 | 0.208099 | 0.208099 |
| VAD loss | 0.670966 | 0.313635 | 0.313635 |

- first 32 update mean: 0.430909
- last 32 update mean: 0.230247（46.6%低下）
- NaN / Infなし
- gain / VADは0〜1内
- SafeTensors再読込後の評価値は完全一致

全長BPTT版の学習後eval total loss `0.185812`よりは高い。長時間学習では更新回数を増やし、
chunk長100/200/400と音質を比較して、速度と長期依存のトレードオフを決める。
