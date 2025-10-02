# ARTI-6: Towards Six-dimensional Articulatory Speech Encoding
### Jihwan Lee<sup>1</sup>, Sean Foley<sup>1,2</sup>, Thanathai Lertpetchpun<sup>1</sup>, Kevin Huang<sup>1</sup>, Yoonjeong Lee<sup>1</sup>, Tiantian Feng<sup>1</sup>, Dani Byrd<sup>2</sup>, Louis Goldstein<sup>2</sup>, Shrikanth Narayanan<sup>1</sup>

#### <sup>1</sup>Signal Analysis and Interpretation Laboratory, University of Southern California, USA <br> <sup>2</sup>Department of Linguistics, University of Southern California, USA

### Official code implementation of [paper](https://www.arxiv.org/abs/2509.21447 "paper link") (Submitted to ICASSP 2026)

### Abstract
We propose ARTI-6, a compact six-dimensional articulatory speech encoding framework derived from real-time MRI data that captures crucial vocal tract regions including the velum, tongue root, and larynx. ARTI-6 consists of three components: (1) a six-dimensional articulatory feature set representing key regions of the vocal tract; (2) an articulatory inversion model, which predicts articulatory features from speech acoustics leveraging speech foundation models, achieving a prediction correlation of 0.87; and (3) an articulatory synthesis model, which reconstructs intelligible speech directly from articulatory features, showing that even a low-dimensional representation can generate natural-sounding speech. Together, ARTI-6 provides an interpretable, computationally efficient, and physiologically grounded framework for advancing articulatory inversion, synthesis, and broader speech technology applications. The source code and speech samples are publicly available.

![overall_architecture](docs/src/overview_architecture.png)

## Data
For the articulatory inversion task, we use the dataset from [A long-form single-speaker real-time MRI speech dataset and benchmark](https://arxiv.org/abs/2509.14479). It can be downloaded from [here](https://sail.usc.edu/span/single_spk). For the articulatory synthesis task, we utilize [LibriTTS-R](https://www.openslr.org/141) for data augmentation. 

## Sample Page
Speech samples are available [here](https://lee-jhwn.github.io/arti-6/ "speech samples").

## Example
An example code snippet is available in [`example.py`](example.py). You can load the model as below. The checkpoints will be automatically loaded from huggingface repository. Only `WavLM-large` version is currently supported.

```python
from arti6 import ARTI6

arti6_model = ARTI6(device=device) # device: cpu or cuda
arti6_model.load_model()
```

### Articulatory Inversion
Once the model is loaded, articulatory features can be inferred from speech acoustics as below. It will return a dictionary of `arti_feats` and `spk_emb`, which are articulatory features and speaker embedding, respectively. The 

```python
articulatory_feats = arti6_model.invert(wav_path=<path_to_wav_file>)
```

It will return the predicted articulatory features in the following format. The current version does not support batch processing, hence the batch size (B) is fixed to 1. The order of the articulatory features (Regions-of-Interest) are: `Lip Aperture (LA), Tongue Tip (TT), TB (Tongue Body), Velum (VL), Tongue Root (TR), and Larynx (LX)`. The speaker embedding is extracted by `ECAPA-TDNN`.

```python
{
    "arti_feats": (B,T,6), # LA, TT, TB, VL, TR, and LX
    "spk_emb": (B,192)
}
```

### Articulatory Synthesis
Speech acoustics can be synthesized from articulatory features as below.

```python
synthesized_audio = arti6_model.synthesize(articulatory_feats['arti_feats'], articulatory_feats['spk_emb'])
```

## Training
Training code will be available upon publication.

## Contact
Jihwan Lee (jihwan@usc.edu)

## Reference
Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae, “Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis,” Advances in neural information processing systems, 2020 [(https://github.com/jik876/hifi-gan)](https://github.com/jik876/hifi-gan)