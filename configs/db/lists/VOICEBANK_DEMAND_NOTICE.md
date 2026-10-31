# VoiceBank-DEMAND test ids

This list is the 824 utterance ids from MP-SENet's VoiceBank-DEMAND test split.
It is stored here so `make db-init` does not read `src/se/official/`.

Source: <https://github.com/yxlu-0102/MP-SENet> `VoiceBank+DEMAND/test.txt`  
License: MIT  
Copyright (c) 2023 Yexin Lu

The MIT notice requires the copyright and permission text to stay with the list:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The wav files are not in this repository. `configs/db/init.yaml` points
`clean_dir` and `noisy_dir` at `data/corpora/VoiceBank-DEMAND/`. Registration
is skipped when either side is missing, and those directories are not created
here.
