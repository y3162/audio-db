# 学習中の研究評価とモデル選択

公式の検証、scheduler、公式の推論用重みは各 Trainer が従来どおり実行する。この文書の評価と選択は、その完了後に追加する研究用の経路である。公式の損失式、停止、scheduler、`best_generator.pt`、`valid.loss.best.pt`、`valid.loss.ave_*best.pt`、`valid.si_snr.best.pt`、`{N}epoch.pt` は置き換えない。

指標の計算、集約、選択判定は Python が行う。YAML は指標、データ、タイミング、選択基準を宣言する。

## 役割

| 役割 | クラスの例 | すること |
|---|---|---|
| `metric` | `src.experiments.metrics.SISDRMetric`、`PESQMetric`、`STOIMetric`、`ESTOIMetric`、`LossMetric`、`WerMetric` | 数値を計算する。状態は保存しない。`state.save: true` は構築時に拒否する |
| `evaluator` | `src.experiments.evaluation.AudioPairEvaluator` | 指定した split で指標を実行し、rank 間で集計する。重み、Optimizer、学習カーソルは変えない |
| `other` | `src.experiments.selection.BestMetricSelection` | 有限な値の best を決める。ファイルは書かない。`state.save: true` が必須 |
| `checkpoint.selector` | `src.experiments.selectors.TopKSelector` | Archive の保持。選択ポリシーとは別 |

`source` は `validation` または `train_eval`。`train` は更新時の損失であり、Evaluator の split ではない。`test` は Evaluator の source にできない。

学習損失の平均を `train_eval.loss` として記録しない。`train_eval` は `eval()` と、学習カーソルを進めない `data.train_eval_epoch(epoch)` を使う。`validation` は従来の `validation_epoch(epoch)` を使う。

## 指標

- `LossMetric` は注入した objective のバッチ平均を、発話数で加重平均する。非有限は失敗であり、0 にしない。公式の検証損失とは別の名前で記録する。`validation.loss` は公式検証が橋渡しした値である。
- `SISDRMetric` は `src.evaluation.metrics.score_si_sdr` の正の dB。`si_sdr_loss` の符号を反転した訓練損失ではない。完全再構成は有限値にしない。
- `PESQMetric`、`STOIMetric`、`ESTOIMetric` は同じ評価実装を使う。`na`、`unavailable`、失敗は平均に入れず、0 にも公式の −1 にもしない。
- `WerMetric` は `wer-norm-v1` のコーパス micro `(S+D+I)/N`。認識器は `transcribe(audio, sample_rate)` を注入する。参照テキストは `data.reference_transcript(sample_id)`。参照欠落、仮説の失敗、空のトークン列は失敗または除外であり、0.0 にしない。1 件でも欠けると status は `incomplete` で、`value` は None。この値では順位を付けない。

## タイミング

`every` は `unit` と `interval`。

| unit | 動くとき |
|---|---|
| `epoch` | 公式の境界で、epoch が interval の倍数のとき。epoch 0 を含む。Trainer がその progress キーを出すときだけ使える |
| `logical_step` | 完了した step または境界で、logical step が interval の倍数のとき。Trainer がその progress キーを出すときだけ使える |
| `boundary` | 公式の検証境界。interval は 1。カウンタは要らない |
| `end` | 学習完了時。interval は 1。カウンタは要らない |

`logical_step` または `epoch` を、そのキーを出さない Trainer で使うと、評価の時点で拒否する。6 つの公式 Trainer は両方を出す。

完了した token は再開後に再実行しない。`boundary_pending` で再開した境界は 1 回だけ実行する。評価は `torch.inference_mode()` で行い、終了時に train/eval、RNG、データカーソルを戻す。GAN の更新途中では実行しない。全 rank が同じ評価に参加する。rank 0 だけが `research_metrics.jsonl` と研究用の推論ファイルを書く。

Trainer は `ResearchSession.score` と `metrics` に、runtime、`output_directory`、`config_identity`、`kind`、`epoch`、progress、公式指標、停止状態を渡す。CheckpointManager は渡さない。ログに残る `config_sha256` はその同一性である。設定を持たない直接比較では null のままである。

DeepFilterNet3 の波形は `DeepFilterNet3Trainer.research_predict` が返す。MossFormerGAN の研究用波形は予測した複素スペクトルの ISTFT であり、公式の time 項が clean の実部と予測の虚部を混ぜる経路とは別である。研究用の評価は弁別器を呼ばない。

## モデル選択

`BestMetricSelection` の `metric` は `source.name`。`mode` は `min` または `max`。同値は既定で早い logical step を残す。`tie_break: later` で遅い方を残す。NaN、Inf、欠落、`incomplete` は拒否し、best にしない。`k` は順位メタデータの長さである。checkpoint の Archive は `TopKSelector` が持つ。

名前空間なしの `loss` は従来どおり境界だけで順位付ける。`validation.wer` のように `.` を含む名前は、そのキーが付いた step と final も `TopKSelector` が順位付ける。公式の有限な境界指標は `validation.{name}` としても保存点に載る。研究用の指標が同じキーを返すときは失敗する。

改善したとき、rank 0 が `research_selection/best.pt` と `best.json` を一時ファイルから `os.replace` する。形式は `audio-db-research-selection-v1`。ポリシー自身はファイルを書かない。このファイルは checkpoint ではない。

研究用のログと選択の来歴は、データが `lineage()` を返すときその claim、dataset、profile、protocol、`selection_uses_test`、snapshot hash を残す。`selection_uses_test` が true のとき independent は false。来歴が無いか不完全なときは UNKNOWN。true にはしない。

`official_faithful` またはデータの `official_reproduction` と、研究用の metric、evaluator、selection の併用は構築時に拒否する。`experiment_claim: official` という表記で任意の選択が公式再現になることはない。

## 例

公式の検証損失で研究用の best を選ぶ。追加の損失指標は置かない。

```yaml
selection:
  role: other
  class: src.experiments.selection.BestMetricSelection
  kwargs:
    metric: validation.loss
    mode: min
    tie_break: earlier
  state:
    save: true
trainer:
  role: trainer
  class: src.experiments.official.tf_gridnet.trainer.TFGridNetTrainer
  inject:
    model_selection: {ref: selection}
```

検証 WER で選ぶ。認識器と参照テキストはデータ側が供給する。

```yaml
wer:
  role: metric
  class: src.experiments.metrics.WerMetric
  kwargs:
    name: wer
    normalization: wer-norm-v1
  inject:
    recognizer: {ref: recognizer}
validation_eval:
  role: evaluator
  class: src.experiments.evaluation.AudioPairEvaluator
  kwargs:
    source: validation
    every: {unit: epoch, interval: 1}
  inject:
    model: {ref: model}
    data: {ref: data}
    metrics: [{ref: wer}]
selection:
  role: other
  class: src.experiments.selection.BestMetricSelection
  kwargs:
    metric: validation.wer
    mode: min
  state:
    save: true
checkpoint:
  selector:
    class: src.experiments.selectors.TopKSelector
    kwargs:
      metric: validation.wer
      mode: min
      k: 1
```

train_eval を監視し、選択は公式の検証損失で行う。

```yaml
train_loss:
  role: metric
  class: src.experiments.metrics.LossMetric
  kwargs:
    name: loss
train_eval:
  role: evaluator
  class: src.experiments.evaluation.AudioPairEvaluator
  kwargs:
    source: train_eval
    every: {unit: epoch, interval: 1}
  inject:
    model: {ref: model}
    data: {ref: data}
    objective: {ref: objective}
    metrics: [{ref: train_loss}]
selection:
  role: other
  class: src.experiments.selection.BestMetricSelection
  kwargs:
    metric: validation.loss
    mode: min
  state:
    save: true
```

`train_eval.loss` は objective を評価モードで再計算した値である。更新時に集計した訓練損失とは別の行になり、`research_metrics.jsonl` に `source: train_eval` と書く。
