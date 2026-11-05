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

追加の CPU 検証では、管理停止からの継続と古い checkpoint への巻き戻しで、同一 step の train と validation を含む JSONL と表示 Scalar を照合した。途中停止の注入で checkpoint 公開後の記録欠落を検出したため、JSONL 書き込みと event flush の後に checkpoint を公開するよう修正し、保存直後、JSONL 書き込み後、event flush 後の例外から再開する試験を加えた。これは例外を注入し writer を正常に閉じる疑似中断試験であり、強制終了、電源断やストレージ障害の試験ではない。JSONL の再生成は Scalar のみで、音声と画像は復元しない。

同じ CPU・seed・合成データを使い、縮小 MP-SENet の reconstruction と公式損失、TF-GridNet、TF-Locoformer、SEMamba++（reference scan）、DeepFilterNet3、MossFormerGAN で、TensorBoard OFF と音声・画像プレビューを含む ON を比較した。各 2 更新直後と検証後の task state（モデル、Optimizer、Scheduler）と CPU の Python / NumPy / PyTorch RNG は厳密一致し、損失・検証値も一致した。公式 MP-SENet 損失は短い波形のため PESQ ターゲットの再現ではない。これは単一 CPU の試験であり、論文学習や GPU の決定性を示さない。再実行用の長めの試験は `tests/training/reference/tensorboard_equivalence.py` に置いた。

2026-11-05 の TSUBAME 4.0 `node_h=1` では、H100 2 枚（MIG 無効、各 95830 MiB、確保時の使用量 0 MiB）に `CUDA_VISIBLE_DEVICES` を限って NCCL の 2 プロセス DDP を実行した。縮小した実モデルで global batch 4、micro batch 1、勾配蓄積 2、Optimizer 更新 2 回、step 2 の validation を通した。対象は MP-SENet の reconstruction と公式損失、MossFormerGAN、DeepFilterNet3、SEMamba++（reference scan）、TF-Locoformer、TF-GridNet である。両 rank の更新後パラメータの最大絶対差は 0 で、許容は rtol `1e-6`、atol `1e-7` だった。rank 0 の event は 1 ファイルで、学習損失・学習率・有限な validation 損失は JSONL と一致し、音声 3 件と画像 1 件があり、`preview_errors` はなかった。`step_00000002.pt` を新規の一時ディレクトリへ保存した。詳細とジョブ ID は [最終検証](TENSORBOARD_FINAL_VERIFICATION.md) に記録した。SEMamba++ の fused scan、GPU 上の DDP 再開、論文設定の長時間学習はこの実行に含まれない。

最終検証の条件と再実行手順は [TENSORBOARD_FINAL_VERIFICATION.md](TENSORBOARD_FINAL_VERIFICATION.md)。

再実行用テストは `tests/training/test_tensorboard_scalars.py` と `tests/training/test_tensorboard_integration.py`。後者の実モデル試験は既定で CPU を使い、`AUDIO_DB_TB_DEVICE=cuda` で単一 GPU を使う。DDP 試験は既定で CPU、`AUDIO_DB_TB_DDP_DEVICE=cuda` で 2 GPU を使う。`AUDIO_DB_TB_DDP_MODELS` で対象モデルを、`AUDIO_DB_TB_DDP_GLOBAL_BATCH=4` で勾配蓄積 2 を指定できる。TF-GridNet は既定で 1 更新で、`AUDIO_DB_TB_GRID_STEPS=2` にすると 2 更新と step 2 の validation になる。`AUDIO_DB_TB_DDP_SAVE_PARAMETERS=1` のとき両 rank の Generator パラメータを照合する。GPU 実行前に空きを `nvidia-smi` で確認し、`CUDA_VISIBLE_DEVICES` を指定する。
