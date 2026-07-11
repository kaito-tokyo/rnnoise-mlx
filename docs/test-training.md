# MLX / BNNS向けテスト学習手順

この試験は音質モデルの完成ではなく、RNNoise形式のMLX学習が成立することを確認する。
学習と評価には別の発話集合を使い、256系列・batch 8・10 epoch（320更新）で実行する。

## データ

- train: LibriTTS-R `train-clean-100`から固定seedで1,000 clips
- eval: LibriTTS-R `dev-clean`から固定seedで250 clips
- noise: Xiph `background_noise_v2.sw`と`foreground_noise.sw`
- RIR: Xiph `measured_rirs-v2`

LibriTTS-RはCC BY 4.0。取得物、PCM、特徴量、重みはGit管理しない。
`scripts/prepare_libritts_raw.sh`は各clipを48 kHz mono s16leへ変換し、間に100 msの
無音を挿入し、選択一覧を`.clips.txt`へ保存する。

upstream `dump_features`には、RIR用65536点FFTについて`kiss_fft.c`の
`fstride[MAXFACTORS + 1]`修正と、`dump_features.c`の巨大作業配列をheapへ移す修正を
適用する。macOSでのビルドは次のとおり。

```sh
./autogen.sh
lt_cv_sys_max_cmd_len=262144 ./configure
make -j4 dump_features
```

特徴量は98 float32/frame（特徴65、ゲイン32、VAD 1）、2,000 frames/系列。
期待サイズはtrain 200,704,000 bytes、eval 50,176,000 bytes。

## 実行

```sh
rnnoise-mlx-train data/features/train.f32 runs/test-training \
  --eval-features data/features/eval.f32 \
  --batch-size 8 --sequence-length 2000 \
  --segmented-tbptt-length 100 --segmented-tbptt-state carry \
  --epochs 10 --max-updates 320
```

`training.json`へ初期／学習後／再読込後のtotal・gain・VAD loss、有限性、出力範囲、
時間、総処理フレーム、`frames/s`、audio seconds/sを記録する。成功条件は320更新完走、train loss低下、eval loss改善、NaN/Infなし、
出力が0〜1、再読込後の評価一致、PyTorch CPU版との速度比較である。

320更新はパイプライン検証に限る。WAV歪みは失敗条件とせず、C exportやBNNS統合は
この試験へ含めない。

MLX 0.32の`nn.GRU`は時系列をPythonループで展開する。開発時は2,000フレームを100フレーム
ずつ処理し、Conv/GRU stateを継続しながら境界で勾配だけを切る。全20区間を同じ重みで処理し、
optimizerは最後に1回だけ更新する。chunk 100のモデルは固定評価WAVによるユーザー主観の
高速スクリーニングに使い、有望な候補だけchunk 500で再確認する。

forward、backward、AdamW更新は`model.state`と`optimizer.state`を入出力として
`mx.compile`する。学習率scheduleはoptimizerへcallableとして渡し、MLX arrayのstepから
計算する。compileを診断目的で無効化する場合のみ`--no-compile`を指定する。

モデル・optimizer stateとlossは`mx.async_eval`でscheduleし、lossをPythonへ取り出すのは
10更新ごととする。データは専用threadで1 batchだけ先読みする。診断時は`--sync-eval`、
`--no-prefetch`で個別に無効化できる。
