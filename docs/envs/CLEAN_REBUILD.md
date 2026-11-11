# クリーン再構築

現行の作り直しは [セットアップ](SETUP.md) の `make setup PLATFORM=tsubame` または `make setup PLATFORM=shinoda_lab` である。依存は `requirements/<role>.in` と `requirements/<role>.lock.txt`。`requirements/clean-rebuild.txt` はリポジトリに無い。2026-11-04 の `.venvs/clean-20261104` は、その日の一回の記録であり、現行の手順ではない。

同じ `make setup` を再実行する。成功済みの仮想環境は、lock のハッシュが一致し import できるときやり直さない。`FORCE_SETUP=1` は lock の同期からやり直すが、仮想環境ディレクトリは消さない。

## 融合カーネルと libdf

`mamba-ssm` の注意、`make df-rfft`、Cargo 1.99 と公式 `Cargo.lock`、HDF5、`patchelf` はセットアップの手順に書いてある。パッチは `patches/mamba_ssm-2.3.2.post1-init.patch` で、`scripts/apply_mamba_init_patch.py` が当てる。

## 2026-11-04 の記録

計算ノード `r4n11`（対話ジョブ `115437`、H100 MIG 4g.47gb）で `.venvs/clean-20261104` を作ったときの記録である。現行の `.venvs/se` で取り直してはいない。

- そのノードのツールは Python 3.12.13（uv）、rustc / cargo 1.99.1、`module load cuda/12.3.2`、`module load hdf5-parallel/1.14.3/gcc11.4.1`、`patchelf`。PyTorch は cu124 の wheel。当時の cubin は `sm_90` だけである。別ノード、別アーキテクチャでの再ビルドは未実施。未実施の理由はジョブを増やさないためであり、依存関係やコンパイラの失敗ではない。
- 同日、同じノードで `make setup PLATFORM=tsubame` の終了コードは 0。診断は `PASS=47 FAIL=0 SKIP=0 NOT_TESTED=0`。PyTorch の CUDA 加算と `causal_conv1d_fn` は通った。`deepfilternet` は、Cargo 1.99 が公式 `Cargo.lock` を `--locked` で更新できないため、既存 wheel を入れ直した。公式 checkout の `Cargo.lock` は変えていない。SHINODA-LAB の同じ入口は、この記録の時点では未実行である。
- 公式 DeepFilterNet の短い学習では、`num_workers = 0` のとき epoch 開始直後に `TimeoutError` になり、プロセスは戻らなかった。`num_workers = 2` と `overfit = true` では 2 更新が完了した。公式再現の完了ではない。
- ESPnet の `espnet2/bin/enh_train.py` を短時間動かした記録は、batch 1、worker 0、更新 2、chunk 16000、1 epoch で、VoiceBank を 16 kHz に変換して公式 DataLoader に渡したものである。公式レシピの再現ではない。当時足りなかった `typeguard==4.4.4`、`configargparse`、`humanfriendly`、`torch-complex`、`omniio`、`sentencepiece`、`fast-bss-eval` は、現行の `requirements/official.in` にある。`flash_attn` は無く、ESPnet は自前の注意実装へ戻る、という記録がある。MS-SNSD の公式設定は 16 kHz、`chunk_length` 48000、`n_fft` 512、`stride` 256、Adam、`mr_l1_tfd` である。
- SEMamba++ の `train.py` は epoch 数を引数にしない。ピン留めツリーは `utils.py` しか無いのに `dataset.py` が未使用の `utils.util` を import する。当時の短時間実行は公式ファイルを編集せず、外部 launcher が `main` のソース文字列だけを差し替えて `exec` し、同じ関数を `sys.modules` に登録した。公式 SEMamba++ は `mamba_ssm.models.mixer_seq_simple` を直接 import するため、その入口には `transformers==4.46.3` が要る。現行の `requirements/official.in` は `transformers==4.46.3` を含む。

## 2026-11-02 の旧仮想環境

`.venvs/tsubame`、`.venvs/tsubame-audio`、`.venvs/tsubame-h100-build` を手で作る手順は、現行の `make setup` に置き換わっている。次のスモークは `.venvs/se` で取り直してはいない。

- `gpu_h=1` で H100 MIG 3g.47gb、47,488 MiB、compute capability 9.0、driver 580.105.08 を確認し、PyTorch 2.6.0+cu124 の CUDA 演算が成功した。
- 小さな合成データでは、5 モデルの有限勾配、更新、checkpoint の保存と読込が成功した、という記録がある。SEMamba++ は reference scan。5 モデルとも保存した重みを推論 API に読み直し、短い合成波形を処理した。公式バッチ、融合カーネル、forward と backward 別のメモリピークは未検証である。構築時と step 全体の peak はジョブログに出すが、直前のモデルの reserved メモリを含み得る。
- 同じスモークは、再作成した `.venvs/tsubame` の 5 分の `gpu_h=1` でも成功した、と記録されている。ログのパスは `.venvs/tsubame/logs/rebuild-gpu-smoke.log`。このファイルが今もあるかは確認していない。
