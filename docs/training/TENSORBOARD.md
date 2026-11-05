# TensorBoard

学習 YAML に任意の監視設定を追加する。

```yaml
logging:
  tensorboard:
    enabled: true
    scalar_interval: 10
    audio:
      enabled: true
      max_cases: 3
      max_seconds: 5
    spectrogram: true
```

既定では無効である。有効にするには、学習環境に `tensorboard` をインストールする。一次記録は引き続き `metrics.jsonl` で、イベントは rank 0 が `checkpoint_dir/tensorboard` に書く。Scalar の step は Optimizer の更新回数であり、検証値は検証が実行されたときだけ記録する。学習率は更新と、その step に epoch の Scheduler 更新があればそれも終わった後の Optimizer の値である。次の更新の開始時に学習率を変更するモデルでは、次回使用する値と異なる場合がある。欠損指標は記録しない。

音声プレビューには case ID の昇順で先頭の発話を使う。失敗した場合は `preview_errors` に理由をテキストで記録する。再生用の音声は [-1, 1] にクリップする。スペクトログラムは clean、noisy、enhanced を左から共通の表示範囲で並べる。可視化用 STFT は Hann 窓、512 サンプル、hop 128、振幅の常用対数を使う。

リモートホストでは、例えば次のように localhost で TensorBoard を起動する。

```bash
tensorboard --logdir data/results/training --host localhost --port 6006
```

手元の端末から `ssh -L 6006:localhost:6006 HOST` で接続し、ブラウザで `http://localhost:6006` を開く。`--bind_all` を使う場合は、ネットワークからのアクセスが適切な環境に限る。複数 run の共通親ディレクトリを指定すると、TensorBoard 上で run を比較できる。異なる目的関数の学習損失の絶対値ではなく、共通の検証指標でモデルを比較する。

再開時は、checkpoint の step より後のイベントを表示履歴から除く。JSONL の原本や既存イベントを変更せず、別のディレクトリに Scalar を再生成することもできる。

```bash
python -m src.training.core.tensorboard path/to/metrics.jsonl path/to/new-event-directory
```

ホスト側の更新時間とスループットの計測区間は `train_update` のみである。CUDA 処理は非同期なので、これらはホスト側の時間であり、同期を取った GPU ベンチマークではない。CUDA メモリは rank 0 の PyTorch allocator の値で、全 GPU の合計ではない。

## 検証範囲

2026-11-05 の小規模テストで、次を確認した。いずれも論文設定での長時間学習や公式出力との一致を示すものではない。

- CPU の Mock 2 更新と縮小 MP-SENet 1 更新では、TensorBoard の有無による学習状態・乱数の一致を確認した。event の読み戻しと、再開時の古い Scalar の除去も確認した。
- TF-GridNet、TF-Locoformer、SEMamba++、DeepFilterNet3、MossFormerGAN は縮小構成・各 1 更新と検証を CPU と `tensor` の A100 80GB 各 1 GPU で実行した。学習・検証損失と学習率の event を JSONL と照合し、欠損 Scalar の不在、音声 3 種と画像の記録を確認した。SEMamba++ は reference scan である。
- `shannon` の A100 80GB 2 枚で NCCL による 2 プロセス DDP を実行した。Mock では両 Optimizer の学習率と損失が rank 0 から 1 回だけ記録されること、縮小 TF-GridNet では 1 更新・検証と Scalar・音声・画像が記録されることを確認した。GPU DDP で他のモデルを実行したことは意味しない。
- 同日の追加試験では、MP-SENet（reconstruction と公式損失）、TF-Locoformer、SEMamba++、DeepFilterNet3、MossFormerGAN を CPU の 2 プロセス DDP で各 2 更新・検証した。勾配蓄積 2 の場合も、両 rank の更新後パラメータ、event の step と JSONL 値、音声・画像を確認した。MP-SENet の公式損失では遅延 backward の Generator 勾配同期、MossFormerGAN では Discriminator の BatchNorm バッファと 2 回の backward に対する同期、DeepFilterNet3 では未使用パラメータの検出を修正した。MP-SENet の公式損失は短い合成波形であり、PESQ ターゲットを使う条件の再現ではない。使用中でない同一ホストの GPU が 2 枚見つからなかったため、これら 5 モデルの GPU DDP はこの追加試験では未検証である。
- `tensor` の DeepFilterNet3 試験は、共有の `libdf_rfft.so` が同ホストの glibc と互換でなかったため、同ホストの一時ディレクトリで `realfft` カーネルをビルドし、`DF_RFFT_LIBRARY` で指定して行った。共有ライブラリは上書きしていない。

再実行用テストは `tests/training/test_tensorboard_scalars.py` と `tests/training/test_tensorboard_integration.py`。後者の実モデル試験は既定で CPU を使い、`AUDIO_DB_TB_DEVICE=cuda` で単一 GPU を使う。DDP 試験は既定で CPU、`AUDIO_DB_TB_DDP_DEVICE=cuda` で 2 GPU を使う。`AUDIO_DB_TB_DDP_MODELS` で対象モデルを、`AUDIO_DB_TB_DDP_GLOBAL_BATCH=4` で勾配蓄積 2 を指定できる。GPU 実行前に空きを `nvidia-smi` で確認し、`CUDA_VISIBLE_DEVICES` を指定する。
