# ドキュメント

概要は [README](../README.md)。エージェントの制約は [AGENTS.md](../AGENTS.md) と [.agents/](../.agents/PROJECT.md)。

## 使う

- [チートシート](CHEATSHEET.md)
- [DB](db/SETUP.md)、[レシピ](db/RECIPE.md)、[コーパス](db/CORPORA.md)
- [評価](evaluation/PIPELINE.md)
- [SE モデル](se/MODELS.md)、[ASR](asr/MODELS.md)
- [環境](envs/SETUP.md)

## 守る

- 評価の入力は名前付きセットだけ。実行時に Manifest を読まない。
- `na` と `unavailable` を 0 にしない。
- CHiME3 の CH0 は clean ではない。real と元 WSJ は時間整合したペアではない。
- 学習 checkpoint は `audio-db-experiment-v1`。`data/` 以下の実データは削除しない。
- `dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked。

学習の式、更新順、再開の条件は [.agents/ARCHITECTURE.md](../.agents/ARCHITECTURE.md)。データの不変条件は [.agents/DATA.md](../.agents/DATA.md)。
