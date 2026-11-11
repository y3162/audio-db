# TSUBAME 4.0

構築は次のとおり。ホスト差の詳細と依存の正本は [セットアップ](SETUP.md)。

```bash
cp .env.example .env
make setup PLATFORM=tsubame
make doctor
```

`scripts/setup/platforms/tsubame.sh` が読むモジュールは `cuda/12.3.2` と `hdf5-parallel/1.14.3/gcc11.4.1`。融合カーネルの `TORCH_CUDA_ARCH_LIST` は `9.0`。PyTorch の wheel は cu124。cargo が PATH に無いと、この準備の段階で止まる。求める Rust は 1.76 以上。

GPU の計算と CUDA 拡張のビルドは、スケジューラで確保した計算ノードで行う。研究室でビルドした融合カーネルは使わない。fused 用の環境は `.venvs/semambapp` で、`make setup` がこのノードの `TORCH_CUDA_ARCH_LIST` で作る。

## 資源

`gpu_h` は CPU 4 コア、ホストメモリ 48 GB、H100 の半分の MIG を割り当てる。GPU メモリの容量はジョブ内で `nvidia-smi` と PyTorch から確認する。`gpu_1` は GPU 1 枚である。詳細は [公式ジョブ手引き](https://www.t4.cii.isct.ac.jp/docs/all/handbook.ja/jobs/) と [ソフトウェア手引き](https://www.t4.cii.isct.ac.jp/docs/all/handbook.ja/software/) を参照する。

ジョブ内で `nvidia-smi` の空きと MIG を確認してから計算する。デバイス番号を他ノードの物理番号に固定しない。

## 投入

長時間の学習は `qsub`、短時間の開発・デバッグは `qrsh` を使う。いずれも優先度は、ユーザーから別の指示がなければ `-p -5` とする。30 分程度以上のジョブは、投入前にそのジョブを含むユーザーの明示的な指示または追加承認が必要である。`h_rt` を 30 分未満にして、この確認を避けない。

利用する TSUBAME グループを指定して GPU を確保する。

## 学習ジョブ

学習は `scripts/experiments/job.sh CONFIG [NPROC]` で起動する。中身は audio-db の Launcher で、研究室サーバーやローカルと同じである。Launcher はジョブを投入せず、再投入もしない。リポジトリのルートから投入する。

```bash
# GPU 2 枚の 1 ノード（node_h）、DDP 2 rank、20 分
qsub -g GROUP -p -5 -l node_h=1 -l h_rt=0:20:00 \
  -v AUDIO_DB_STOP_AFTER=600,AUDIO_DB_GRACE=300 \
  scripts/experiments/job.sh configs/reproduction/experiments/variants/CONFIG.yaml 2

# 続き
qsub -g GROUP -p -5 -l node_h=1 -l h_rt=0:20:00 \
  -v AUDIO_DB_RESUME=data/results/experiments/RUN,AUDIO_DB_STOP_AFTER=600 \
  scripts/experiments/job.sh configs/reproduction/experiments/variants/CONFIG.yaml 2
```

- Launcher は 1 ノードの中で rank を起動する。資源タイプの個数は 1 にし、必要な GPU 数を持つタイプを選ぶ（`gpu_1` は 1 枚、`node_h` は 2 枚、`node_f` は 4 枚）。`gpu_1=2` は 1 枚ずつの 2 ノードになり、この Launcher では使えない。
- `-v NAME=VALUE` は Grid Engine の標準オプションである。TSUBAME の手引きの主なオプションには載っておらず、ここでは実ジョブで確かめていない。使えないときは、変数を `export` してから `job.sh` を呼ぶ自分のジョブスクリプトを書く。
- `AUDIO_DB_STOP_AFTER` 秒で安全な停止を要求し、`AUDIO_DB_GRACE` 秒（既定 300）待つ。両者の和を `h_rt` より短くする。時間切れで kill される前に Latest を保存して終わるためである。
- 終了コードは 0 が完了、3 が安全な停止、1 が失敗。3 と 1 は、次のジョブで `AUDIO_DB_RESUME` に出力ディレクトリを渡して続ける。
- `qrsh` の中では、同じ行を `qsub` なしで実行する。
- 状態は `.venvs/se/bin/python scripts/experiments/show_status.py OUTPUT_DIRECTORY` で見る。

`qdel` と `h_rt` 切れがジョブへ送るシグナルに対する動作は、TSUBAME の実ジョブでは未確認である。確認済みなのは、研究室サーバーでの SIGTERM、SIGINT、SIGHUP と `--stop-after` である。仕様は [Launcher](../experiments/LAUNCHER.md)。
