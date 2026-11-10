# audio-db

## 概要

音声強調の研究基盤である。複数の SE モデルを共通の手続きで学習し、固定した ASR の WER（`wer-norm-v1`、micro）で比べる。PESQ / STOI は補助指標で、ASR の重みは更新しない。Dataset の再現条件と Experiment の中断再開を保持する。

## 目次

- [セットアップ・環境構築](docs/envs/SETUP.md)
- [データベース](docs/db/SETUP.md)
- [データセットと再現プロトコル](docs/db/CORPORA.md)
- [音声強調モデル](docs/se/MODELS.md)
- [音声認識モデル](docs/asr/MODELS.md)
- [学習・実験](docs/experiments/RUN.md)
- [評価](docs/evaluation/PIPELINE.md)
- [コマンド・チートシート](docs/CHEATSHEET.md)
