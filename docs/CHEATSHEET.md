# チートシート

```bash
make setup PLATFORM=shinoda_lab
make doctor
make db-init
make reproduction-init-db
make experiment-preflight CONFIG=configs/experiments/mp_senet_gan.yaml
make experiment CONFIG=configs/experiments/official/mp_senet_gan.yaml
make test-cpu
```

```bash
.venvs/se/bin/python -m src.experiments.cli --config configs/experiments/mp_senet_gan.yaml --preflight
.venvs/se/bin/python -m src.evaluation.cli --config configs/evaluation/smoke.yaml --limit 2 --device cpu
```

評価用の `configs/evaluation/smoke.yaml` が作業ツリーに無いときは、そのコマンドは実行できない。DB とスナップショットは `docs/db/SETUP.md`。

`make db-ingest` は `corpora.audio_files` を作り直す。明示されたときだけ、DB を退避してから実行する。
