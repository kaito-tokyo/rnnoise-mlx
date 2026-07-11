# Roadmap and acceptance gates

1. MLX smoke training: 320 updates, finite loss, and a downward loss trend.
2. Held-out features: report gain loss, VAD loss, and inference throughput.
3. Freeze model format and verify sequence-vs-frame numerical parity.
4. Implement the BNNS graph and compare its outputs with MLX within tolerance.
5. Connect pinned RNNoise C DSP and measure end-to-end real-time factor.
6. Train for thousands to 10,000+ updates and perform objective/listening tests.

The 320-update checkpoint proves pipeline viability only; it is not a release
quality model.

## Training optimization order

1. コーパス追加と高速な主観スクリーニングにはchunk 100を使う。
2. 固定した評価音源から同じ手順でWAVを生成し、ユーザーがA/B判定する。
3. 有望なデータ構成だけをchunk 500、state継続で再学習する。
4. Conv履歴と3層GRU stateを継続し、境界では勾配だけを切る。
5. 500フレーム4区間を同じ重みで処理し、optimizer更新は最後に1回だけ行う。
6. 長時間学習では数千〜10,000更新以上を実行し、eval lossとWAVの両方で選ぶ。
7. 必要になった段階でbatch size、learning rate、gradient clippingを比較する。

100/250/500の同一フレーム・同一更新数比較では500が最良のeval lossだった。500のstate保持は
resetよりeval totalを約1.90%改善し、64評価系列のpaired bootstrap区間は0を跨がなかった。
一方100は500より約2.27倍高速なため、開発用の既定とする。詳細は
[segment-length-100-250-500-experiment.md](segment-length-100-250-500-experiment.md)を参照する。

次の速度最適化候補は、500フレーム用compiled graphを4回再利用し、device上で勾配を累積して
最後に1回だけoptimizerを更新する方式である。現在の一体型graphはM2 ProでGPU使用率96%、
GPU割当メモリ約25 GBだったため、まず短いA/Bベンチマークで有効性を確認する。

混合精度は小型モデルでは逆効果の可能性があるため当面採用しない。Metal profilingとMLX
fused GRU primitiveは通常の学習最適化を完了した後の候補とする。
