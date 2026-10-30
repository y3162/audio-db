# 音声強調 API（未完成）

`src.se` は `available_enhancers()`, `build_model(name, config=None)`, `load_enhancer(name, checkpoint, device='cpu', config=None)` を公開する。登録済みは MP-SENet、TF-GridNet、DeepFilterNet3、MossFormerGAN、SEMamba++、UniPASE、StuPASE。`build_model` は勾配を維持する `nn.Module` を返す。学習処理は含まない。単一ファイルのモデルはパスを渡し、UniPASE は `dewavlm` / `adapter` / `vocoder` / `postnet`、StuPASE は `dewavlm` / `cfm` / `vocoder` の辞書を渡す。キーの過不足は拒否する。

`enhance` は一次元浮動小数点 Tensor と整数サンプルレートを受け、元の長さ、dtype、デバイスの Tensor を返す。内部で 16 kHz にリサンプリング、RMS 正規化、複素 STFT、振幅圧縮、Generator、逆変換を行う。無音は無音として返す。非常に短い音声は reflect padding の制約により拒否する。公式の iSTFT は長さを hop の倍数に切り詰めるが、API は元の長さを明示する。リサンプリングは torchaudio を使用し公式 librosa と一致検証していない。複数チャンネル、ストリーミング、チャンク処理は未対応。

新規モデルは独立した `nn.Module` と固有の前後処理を追加し、構築・ロード API へ登録する必要がある。構造の設定と重みキーの互換性を検証し、公式出力との比較および出典・ライセンスを記録すること。Registry とチェックポイント指定の仕組みは下記を参照。

## 追記: Registry と TF-GridNet

`src/se/core/registry.py` の `ModelSpec` にモデル構築・ローダー・別名・variant・基本メタデータを登録し、`model_metadata(name)` で参照できる。`build_model(..., variant=...)` と `load_enhancer(..., variant=...)` は指定 variant を検証する。任意の checkpoint パスに加え、必要な名前とパスの辞書を受け取り、キーの過不足と state dict の不一致を拒否する。ただし現時点のモデルはいずれも単一ファイルであり、複数重みのモデルでの統合検証は未実施。

設定は辞書・JSON・YAML を受け付ける。YAML は追加の PyYAML が必要で、既存 lab には未インストール。既存環境は変更していない。モデル設定は flat mapping であり公式の ESPnet YAML 全体を直接読み込む機能ではない。

`enhancer.enhance_batch(waveforms, sample_rate, lengths=None)` は `(batch, time)` と発話長を受け、各発話を独立推論し `(zero-padded output, lengths)` を返す。バッチ内の異なるサンプルレートは未対応。TF-GridNet の `urgent24` と `ms-snsd` は構造が異なるため `variant` を明示する。`wsj0-2mix` は未対応。URGENT24 は入力サンプルレートに応じて STFT を変更し、MS-SNSD は内部で 16 kHz に変換する。公式の resampling 実装との数値比較は未実施。

DeepFilterNet3 は 48 kHz。各 `enhance` の開始時に STFT の overlap と指数正規化状態を捨てる。MossFormerGAN の公開重みは 201 ビン（`n_fft=400`、窓 400、hop 100、periodic Hamming）。10 秒を超える入力は 10 秒窓で分割し、エネルギー正規化は分割前の発話全体で一度だけ計算する。SEMamba++ は 16 kHz、`n_fft=400`、hop 100、Hann、ピーク正規化のあと復元する。UniPASE は入力レートを保持し、16 kHz 超では PostNet を通す。出力ピークは入力ピークへ合わせる。StuPASE は内部を 16 kHz にし、公式CLIと同じ 8 step / CFG 0.5 / sway -1.0 の Euler 積分でサンプリングする。同じ乱数種を呼び出し前に固定すると短音声の結果は再現する。
