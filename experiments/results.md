# Experiment results

Raw experiment provenance, hyperparameters, media, plots, and data artifacts. Sections are intentionally descriptive and contain no result interpretation.

<details>

<summary><strong>Audio metamers</strong></summary>

**Result layout:** `audio/<model>/metamers/<layer>/`

**Inputs and optimization:** AudioSet TFRecord examples `0-9`; 2-second clips; 4 rounds; 200 steps per round; Adam; learning rate `0.25`; learning-rate decay `0.7`; initial noise mean `0.0` and scale `0.01`; clamp range `[-1, 1]`; total-variation weight `0.0`; range weight `0.0`; range norm `p=6`; seed `42`. Model-aware low-pass filtering and effective cutoff are recorded per sample in `metadata.jsonl`.

**Provenance:** [`generate_audio_metamers.sh`](../generate_audio_metamers.sh) · [`scripts/generate_metamers.py`](../scripts/generate_metamers.py) · [`src/analysis/saving.py`](../src/analysis/saving.py) · [`notebooks/AudioMetamerSpectrograms.ipynb`](../notebooks/AudioMetamerSpectrograms.ipynb)

**Per-layer files:** ten original WAVs, ten metamer WAVs, ten metamer PT tensors, and `metadata.jsonl`. The representative below is sample `idx0000`; use the layer directory or metadata to access samples `idx0001-idx0009`.

<details>

<summary>&emsp;<strong>audiomae_as2m</strong> — 14 layers</summary>

[Open metamer folder](audio/audiomae_as2m/metamers/)

<details>

<summary>&emsp;&emsp;<code>vit_patch_embeddings</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/vit_patch_embeddings/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/vit_patch_embeddings/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_00</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_00/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_00/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_01</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_01/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_01/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_02</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_02/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_02/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_03</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_03/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_03/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_04</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_04/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_04/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_05</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_05/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_05/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_06</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_06/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_06/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_07</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_07/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_07/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_08</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_08/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_08/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_09</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_09/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_09/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_10</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_10/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_10/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_11</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/transformer_block_11/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/transformer_block_11/)

</details>

<details>

<summary>&emsp;&emsp;<code>final_pooled_embedding</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m/metamers/final_pooled_embedding/metadata.jsonl) · [Layer folder](audio/audiomae_as2m/metamers/final_pooled_embedding/)

</details>

</details>

<details>

<summary>&emsp;<strong>audiomae_as2m_ft_as20k</strong> — 14 layers</summary>

[Open metamer folder](audio/audiomae_as2m_ft_as20k/metamers/)

<details>

<summary>&emsp;&emsp;<code>vit_patch_embeddings</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/vit_patch_embeddings/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/vit_patch_embeddings/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/vit_patch_embeddings/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_00</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_00/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_00/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_01</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_01/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_01/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_02</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_02/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_02/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_03</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_03/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_03/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_04</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_04/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_04/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_05</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_05/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_05/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_06</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_06/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_06/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_07</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_07/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_07/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_08</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_08/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_08/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_09</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_09/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_09/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_10</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_10/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_10/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_11</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_11/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/transformer_block_11/)

</details>

<details>

<summary>&emsp;&emsp;<code>final_pooled_embedding</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/audiomae_as2m_ft_as20k/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/audiomae_as2m_ft_as20k/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/audiomae_as2m_ft_as20k/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/audiomae_as2m_ft_as20k/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/audiomae_as2m_ft_as20k/metamers/final_pooled_embedding/metadata.jsonl) · [Layer folder](audio/audiomae_as2m_ft_as20k/metamers/final_pooled_embedding/)

</details>

</details>

<details>

<summary>&emsp;<strong>beats_iter3</strong> — 14 layers</summary>

[Open metamer folder](audio/beats_iter3/metamers/)

<details>

<summary>&emsp;&emsp;<code>projected_features</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/projected_features/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/projected_features/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/projected_features/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/projected_features/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/projected_features/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/projected_features/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/projected_features/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_00</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_00/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_00/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_01</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_01/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_01/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_02</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_02/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_02/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_03</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_03/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_03/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_04</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_04/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_04/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_05</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_05/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_05/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_06</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_06/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_06/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_07</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_07/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_07/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_08</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_08/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_08/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_09</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_09/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_09/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_10</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_10/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_10/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_11</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/transformer_layer_11/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/transformer_layer_11/)

</details>

<details>

<summary>&emsp;&emsp;<code>final_pooled_embedding</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3/metamers/final_pooled_embedding/metadata.jsonl) · [Layer folder](audio/beats_iter3/metamers/final_pooled_embedding/)

</details>

</details>

<details>

<summary>&emsp;<strong>beats_iter3_plus_as2m</strong> — 14 layers</summary>

[Open metamer folder](audio/beats_iter3_plus_as2m/metamers/)

<details>

<summary>&emsp;&emsp;<code>projected_features</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/projected_features/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/projected_features/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/projected_features/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/projected_features/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/projected_features/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/projected_features/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/projected_features/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_00</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_00/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_00/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_00/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_01</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_01/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_01/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_01/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_02</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_02/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_02/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_02/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_03</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_03/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_03/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_03/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_04</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_04/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_04/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_04/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_05</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_05/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_05/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_05/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_06</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_06/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_06/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_06/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_07</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_07/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_07/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_07/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_08</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_08/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_08/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_08/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_09</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_09/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_09/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_09/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_10</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_10/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_10/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_10/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_layer_11</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/transformer_layer_11/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/transformer_layer_11/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/transformer_layer_11/)

</details>

<details>

<summary>&emsp;&emsp;<code>final_pooled_embedding</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/beats_iter3_plus_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/beats_iter3_plus_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/beats_iter3_plus_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/beats_iter3_plus_as2m/metamers/final_pooled_embedding/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/beats_iter3_plus_as2m/metamers/final_pooled_embedding/metadata.jsonl) · [Layer folder](audio/beats_iter3_plus_as2m/metamers/final_pooled_embedding/)

</details>

</details>

<details>

<summary>&emsp;<strong>clap</strong> — 14 layers</summary>

[Open metamer folder](audio/clap/metamers/)

<details>

<summary>&emsp;&emsp;<code>transformer_block_00</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_00/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_00/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_00/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_01</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_01/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_01/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_01/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_02</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_02/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_02/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_02/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_03</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_03/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_03/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_03/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_04</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_04/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_04/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_04/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_05</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_05/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_05/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_05/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_06</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_06/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_06/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_06/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_07</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_07/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_07/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_07/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_08</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_08/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_08/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_08/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_09</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_09/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_09/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_09/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_10</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_10/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_10/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_10/)

</details>

<details>

<summary>&emsp;&emsp;<code>transformer_block_11</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/transformer_block_11/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/transformer_block_11/metadata.jsonl) · [Layer folder](audio/clap/metamers/transformer_block_11/)

</details>

<details>

<summary>&emsp;&emsp;<code>pooler_output</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/pooler_output/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/pooler_output/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/pooler_output/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/pooler_output/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/pooler_output/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/pooler_output/metadata.jsonl) · [Layer folder](audio/clap/metamers/pooler_output/)

</details>

<details>

<summary>&emsp;&emsp;<code>final_audio_embedding</code> — representative sample <code>idx0000</code></summary>

Original: <audio controls preload="none" src="audio/clap/metamers/final_audio_embedding/idx0000_-jBKbbVBHzI_original.wav"></audio>

Metamer: <audio controls preload="none" src="audio/clap/metamers/final_audio_embedding/idx0000_-jBKbbVBHzI_metamer.wav"></audio>

[Original WAV](audio/clap/metamers/final_audio_embedding/idx0000_-jBKbbVBHzI_original.wav) · [Metamer WAV](audio/clap/metamers/final_audio_embedding/idx0000_-jBKbbVBHzI_metamer.wav) · [Metamer tensor](audio/clap/metamers/final_audio_embedding/idx0000_-jBKbbVBHzI_metamer.pt) · [Metadata](audio/clap/metamers/final_audio_embedding/metadata.jsonl) · [Layer folder](audio/clap/metamers/final_audio_embedding/)

</details>

</details>

</details>

<details>

<summary><strong>Audio adversarial attacks</strong></summary>

<details>

<summary>&emsp;<strong>Experiment: 2026-07-15 — targeted Speech</strong></summary>

`attack_mode=targeted`; one-hot BCE target `Speech` (AudioSet label `0`). The shared inputs are the first 10 examples without a Speech label: `0, 5, 10, 11, 12, 14, 19, 20, 22, 23`. Each model/epsilon cell contains 10 attacks.

**Attack settings:** models `audiomae_as2m_ft_as20k`, `beats_iter3_plus_as2m`, and `panns_cnn14`; L2 epsilon grid `{0, 0.8, 1.6, 2.4, 3.2, 4, 4.8}`; 100 PGD steps; step size `2 × epsilon / 100`; one random start. Perturbations were filtered at the model Nyquist cutoff (8 kHz for AudioMAE/BEATs, 16 kHz for PANNs), reprojected to the requested L2 radius, and independently validated from saved PT tensors.

**Main result:** every nonzero model/epsilon cell increased mean Speech probability and decreased mean non-target probability. Mean Speech probability rose from `0.0486` to `0.7959–0.8762` for AudioMAE, `0.1096` to `0.7121–0.7884` for BEATs, and `0.0403` to `0.7733–0.9200` for PANNs. Target top-5 hit rate was `0.9–1.0` for BEATs and `1.0` for every nonzero AudioMAE and PANNs cell. All 210 saved examples passed the independent budget, metadata-consistency, and Nyquist checks; every nonzero cell had mean budget utilization `1.0000`.

[Campaign folder](audio_adversarial_runs/2026_07_15_targeted_speech/) · [Budget validation](audio_adversarial_runs/2026_07_15_targeted_speech/budget_validation.csv) · [Combined curve data](audio_adversarial_runs/2026_07_15_targeted_speech/plots/performance_vs_epsilon.csv)

### Plots

#### Mean Speech probability vs epsilon

![Mean Speech probability vs epsilon](audio_adversarial_runs/2026_07_15_targeted_speech/plots/target_probability_vs_epsilon.png)

#### Speech top-1 hit rate vs epsilon

![Speech top-1 hit rate vs epsilon](audio_adversarial_runs/2026_07_15_targeted_speech/plots/target_top1_hit_rate_vs_epsilon.png)

#### Speech top-5 hit rate vs epsilon

![Speech top-5 hit rate vs epsilon](audio_adversarial_runs/2026_07_15_targeted_speech/plots/target_top5_hit_rate_vs_epsilon.png)

#### Saved perturbation budget utilization

![Saved perturbation budget utilization](audio_adversarial_runs/2026_07_15_targeted_speech/plots/budget_utilization_vs_epsilon.png)

### Aggregate results

| Model | L2 epsilon | Source Speech probability | Adversarial Speech probability | Speech top-1 hit rate | Speech top-5 hit rate | Mean budget utilization |
|---|---:|---:|---:|---:|---:|---:|
| AudioMAE fully fine-tuned | 0 | 0.0486 | 0.0486 | 0.0 | 0.3 | — |
| AudioMAE fully fine-tuned | 0.8 | 0.0486 | 0.7959 | 0.8 | 1.0 | 1.0000 |
| AudioMAE fully fine-tuned | 1.6 | 0.0486 | 0.8542 | 0.9 | 1.0 | 1.0000 |
| AudioMAE fully fine-tuned | 2.4 | 0.0486 | 0.8724 | 0.9 | 1.0 | 1.0000 |
| AudioMAE fully fine-tuned | 3.2 | 0.0486 | 0.8762 | 0.9 | 1.0 | 1.0000 |
| AudioMAE fully fine-tuned | 4 | 0.0486 | 0.8556 | 0.9 | 1.0 | 1.0000 |
| AudioMAE fully fine-tuned | 4.8 | 0.0486 | 0.8454 | 0.9 | 1.0 | 1.0000 |
| BEATs fully fine-tuned | 0 | 0.1096 | 0.1096 | 0.0 | 0.4 | — |
| BEATs fully fine-tuned | 0.8 | 0.1096 | 0.7715 | 1.0 | 1.0 | 1.0000 |
| BEATs fully fine-tuned | 1.6 | 0.1096 | 0.7884 | 1.0 | 1.0 | 1.0000 |
| BEATs fully fine-tuned | 2.4 | 0.1096 | 0.7121 | 0.9 | 1.0 | 1.0000 |
| BEATs fully fine-tuned | 3.2 | 0.1096 | 0.7446 | 0.9 | 0.9 | 1.0000 |
| BEATs fully fine-tuned | 4 | 0.1096 | 0.7505 | 0.8 | 1.0 | 1.0000 |
| BEATs fully fine-tuned | 4.8 | 0.1096 | 0.7389 | 0.8 | 1.0 | 1.0000 |
| PANNs CNN14 supervised | 0 | 0.0403 | 0.0403 | 0.0 | 0.1 | — |
| PANNs CNN14 supervised | 0.8 | 0.0403 | 0.7733 | 0.9 | 1.0 | 1.0000 |
| PANNs CNN14 supervised | 1.6 | 0.0403 | 0.8724 | 0.9 | 1.0 | 1.0000 |
| PANNs CNN14 supervised | 2.4 | 0.0403 | 0.8961 | 0.9 | 1.0 | 1.0000 |
| PANNs CNN14 supervised | 3.2 | 0.0403 | 0.9096 | 0.9 | 1.0 | 1.0000 |
| PANNs CNN14 supervised | 4 | 0.0403 | 0.9200 | 0.9 | 1.0 | 1.0000 |
| PANNs CNN14 supervised | 4.8 | 0.0403 | 0.9193 | 0.9 | 1.0 | 1.0000 |

### Data by model and epsilon

<details>

<summary>&emsp;&emsp;<code>audiomae_as2m_ft_as20k</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_targeted_label0000.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_targeted_label0000.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_targeted_label0000.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_targeted_label0000.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_targeted_label0000.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_targeted_label0000.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_targeted_label0000.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_15_targeted_speech/audiomae_as2m_ft_as20k/adversarial/)

</details>

<details>

<summary>&emsp;&emsp;<code>beats_iter3_plus_as2m</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_targeted_label0000.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_targeted_label0000.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_targeted_label0000.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_targeted_label0000.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_targeted_label0000.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_targeted_label0000.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_targeted_label0000.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_15_targeted_speech/beats_iter3_plus_as2m/adversarial/)

</details>

<details>

<summary>&emsp;&emsp;<code>panns_cnn14</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0_targeted_label0000.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0p8_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_0p8_targeted_label0000.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_1p6_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_1p6_targeted_label0000.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_2p4_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_2p4_targeted_label0000.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_3p2_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_3p2_targeted_label0000.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4_targeted_label0000.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4p8_targeted_label0000/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/adversarial_l2_eps_4p8_targeted_label0000.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_15_targeted_speech/panns_cnn14/adversarial/)

</details>

</details>

**Shared inputs and attack settings:** AudioSet TFRecord examples `0-399`; 400 2-second clips per model/epsilon cell; models `audiomae_as2m_ft_as20k`, `beats_iter3_plus_as2m`, and `panns_cnn14`; logits layer; L2 epsilon grid `{0, 0.8, 1.6, 2.4, 3.2, 4, 4.8}`; 100 PGD steps; step size `2 × epsilon / 100`; one random start; model-aware low-pass filtering before save; saved metrics recomputed after filtering; spectrogram export disabled.

**Provenance:** [`generate_audio_adversarials.sh`](../generate_audio_adversarials.sh) · [`scripts/generate_audio_adversarials.py`](../scripts/generate_audio_adversarials.py) · [`scripts/evaluate_audio_adversarial_map.py`](../scripts/evaluate_audio_adversarial_map.py) · [`scripts/validate_audio_adversarial_budgets.py`](../scripts/validate_audio_adversarial_budgets.py) · [`scripts/plot_audio_adversarial_epsilon_curves.py`](../scripts/plot_audio_adversarial_epsilon_curves.py)

**Per-cell files:** `sweep_summary.json` (aggregate metrics and attack settings), `summary.csv` (per-clip metrics), `metadata.jsonl` (full per-clip provenance/configuration and saved paths), `scores.npz` (527-class scores and targets), plus WAV/PT tensors for original, adversarial, delta, and perturbation outputs.

<details>

<summary>&emsp;<strong>Experiment: 2026-07-09 — untargeted BCE</strong></summary>

`attack_mode=untargeted`; loss `bce_multihot`; objective recorded by the run as untargeted multilabel BCE.

[Campaign folder](audio_adversarial_runs/2026_07_09_bce_untargeted/) · [Budget validation](audio_adversarial_runs/2026_07_09_bce_untargeted/budget_validation.csv) · [Combined curve data](audio_adversarial_runs/2026_07_09_bce_untargeted/plots/performance_vs_epsilon.csv)

### Plots

#### Adversarial mAP vs epsilon

![Adversarial mAP vs epsilon](audio_adversarial_runs/2026_07_09_bce_untargeted/plots/adversarial_map_vs_epsilon.png)

#### Adversarial top-1 vs epsilon

![Adversarial top-1 vs epsilon](audio_adversarial_runs/2026_07_09_bce_untargeted/plots/adversarial_top1_vs_epsilon.png)

#### Adversarial top-5 vs epsilon

![Adversarial top-5 vs epsilon](audio_adversarial_runs/2026_07_09_bce_untargeted/plots/adversarial_top5_vs_epsilon.png)

### Data by model and epsilon

<details>

<summary>&emsp;&emsp;<code>audiomae_as2m_ft_as20k</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0/logits/map_per_class.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8/logits/map_per_class.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6/logits/map_per_class.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4/logits/map_per_class.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2/logits/map_per_class.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4/logits/map_per_class.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8/logits/map_per_class.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_09_bce_untargeted/audiomae_as2m_ft_as20k/adversarial/)

</details>

<details>

<summary>&emsp;&emsp;<code>beats_iter3_plus_as2m</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0/logits/map_per_class.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8/logits/map_per_class.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6/logits/map_per_class.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4/logits/map_per_class.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2/logits/map_per_class.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4/logits/map_per_class.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8/logits/map_per_class.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_09_bce_untargeted/beats_iter3_plus_as2m/adversarial/)

</details>

<details>

<summary>&emsp;&emsp;<code>panns_cnn14</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0/logits/map_per_class.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0p8/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0p8/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0p8/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0p8/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0p8/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_0p8/logits/map_per_class.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_1p6/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_1p6/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_1p6/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_1p6/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_1p6/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_1p6/logits/map_per_class.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_2p4/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_2p4/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_2p4/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_2p4/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_2p4/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_2p4/logits/map_per_class.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_3p2/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_3p2/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_3p2/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_3p2/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_3p2/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_3p2/logits/map_per_class.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4/logits/map_per_class.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4p8/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4p8/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4p8/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4p8/logits/scores.npz) | [mAP summary](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4p8/logits/map_summary.csv) · [per-class mAP](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/adversarial_l2_eps_4p8/logits/map_per_class.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_09_bce_untargeted/panns_cnn14/adversarial/)

</details>

</details>

<details>

<summary>&emsp;<strong>Experiment: 2026-07-09 — true-label suppression</strong></summary>

`attack_mode=suppress`; loss `suppression_bce`; true labels jointly suppressed.

[Campaign folder](audio_adversarial_runs/2026_07_09_suppress/) · [Budget validation](audio_adversarial_runs/2026_07_09_suppress/budget_validation.csv) · [Combined curve data](audio_adversarial_runs/2026_07_09_suppress/plots/performance_vs_epsilon.csv)

### Plots

#### Adversarial mAP vs epsilon

![Adversarial mAP vs epsilon](audio_adversarial_runs/2026_07_09_suppress/plots/adversarial_map_vs_epsilon.png)

#### Mean true-label probability vs epsilon

![Mean true-label probability vs epsilon](audio_adversarial_runs/2026_07_09_suppress/plots/true_label_probability_vs_epsilon.png)

#### Fraction of true labels suppressed vs epsilon

![Fraction of true labels suppressed vs epsilon](audio_adversarial_runs/2026_07_09_suppress/plots/fraction_true_labels_suppressed_vs_epsilon.png)

#### Adversarial top-1 vs epsilon

![Adversarial top-1 vs epsilon](audio_adversarial_runs/2026_07_09_suppress/plots/adversarial_top1_vs_epsilon.png)

#### Adversarial top-5 vs epsilon

![Adversarial top-5 vs epsilon](audio_adversarial_runs/2026_07_09_suppress/plots/adversarial_top5_vs_epsilon.png)

### Data by model and epsilon

<details>

<summary>&emsp;&emsp;<code>audiomae_as2m_ft_as20k</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0_suppress.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_0p8_suppress.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_1p6_suppress.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_2p4_suppress.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_3p2_suppress.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4_suppress.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/adversarial_l2_eps_4p8_suppress.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_09_suppress/audiomae_as2m_ft_as20k/adversarial/)

</details>

<details>

<summary>&emsp;&emsp;<code>beats_iter3_plus_as2m</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0_suppress.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_0p8_suppress.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_1p6_suppress.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_2p4_suppress.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_3p2_suppress.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4_suppress.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/adversarial_l2_eps_4p8_suppress.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_09_suppress/beats_iter3_plus_as2m/adversarial/)

</details>

<details>

<summary>&emsp;&emsp;<code>panns_cnn14</code></summary>

| L2 epsilon | Aggregate | Per-clip CSV | Metadata | Scores | Additional |
|---:|---|---|---|---|---|
| 0 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0_suppress.csv) |
| 0.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0p8_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0p8_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0p8_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0p8_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_0p8_suppress.csv) |
| 1.6 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_1p6_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_1p6_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_1p6_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_1p6_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_1p6_suppress.csv) |
| 2.4 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_2p4_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_2p4_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_2p4_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_2p4_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_2p4_suppress.csv) |
| 3.2 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_3p2_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_3p2_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_3p2_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_3p2_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_3p2_suppress.csv) |
| 4 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4_suppress.csv) |
| 4.8 | [sweep JSON](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4p8_suppress/logits/sweep_summary.json) | [summary](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4p8_suppress/logits/summary.csv) | [JSONL](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4p8_suppress/logits/metadata.jsonl) | [NPZ](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4p8_suppress/logits/scores.npz) | [compact CSV](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/adversarial_l2_eps_4p8_suppress.csv) |

[Open model attack folder](audio_adversarial_runs/2026_07_09_suppress/panns_cnn14/adversarial/)

</details>

</details>

</details>

<details>

<summary><strong>Future experiment entry template</strong></summary>

### `YYYY-MM-DD — experiment name`

- **Purpose/type:**
- **Output root:**
- **Models/checkpoints:**
- **Dataset/samples:**
- **Layers/targets:**
- **Hyperparameters:**
- **Generation command or launcher:**
- **Metadata/config files:**
- **Aggregate data:**
- **Plots/media:**
- **Missing provenance:**

</details>
