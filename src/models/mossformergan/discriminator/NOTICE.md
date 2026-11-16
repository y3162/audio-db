This package adapts the discriminator and Conformer from ClearerVoice-Studio,
commit 6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61 (Apache-2.0;
LICENSE-APACHE). The upstream discriminator identifies AdaptCNN and PoolAttFF
as derived from NISQA, copyright 2021 Gabriel Mittag, Quality and Usability
Lab (MIT; LICENSE-NISQA). Its Conformer identifies lucidrains/conformer,
copyright 2020 Phil Wang (MIT; LICENSE-CONFORMER). The local Conformer replaces
einops rearrangements with equivalent PyTorch operations; imports point to this
package, not the external reference tree.
