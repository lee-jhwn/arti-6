# ARTI-6: Towards Six-dimensional Articulatory Speech Encoding
### Jihwan Lee<sup>1</sup>, Sean Foley<sup>1,2</sup>, Thanathai Lertpetchpun<sup>1</sup>, Kevin Huang<sup>1</sup>, Yoonjeong Lee<sup>1</sup>, Tiantian Feng<sup>1</sup>, Dani Byrd<sup>2</sup>, Louis Goldstein<sup>2</sup>, Shrikanth Narayanan<sup>1</sup>

#### <sup>1</sup>Signal Analysis and Interpretation Laboratory, University of Southern California, USA <br> <sup>2</sup>Department of Linguistics, University of Southern California, USA

### Official code implementation of [paper](https://www.arxiv.org/abs/2509.21447 "paper link") (Submitted to ICASSP 2026)

### Abstract
We propose ARTI-6, a compact six-dimensional articulatory speech encoding framework derived from real-time MRI data that captures crucial vocal tract regions including the velum, tongue root, and larynx. ARTI-6 consists of three components: (1) a six-dimensional articulatory feature set representing key regions of the vocal tract; (2) an articulatory inversion model, which predicts articulatory features from speech acoustics leveraging speech foundation models, achieving a prediction correlation of 0.87; and (3) an articulatory synthesis model, which reconstructs intelligible speech directly from articulatory features, showing that even a low-dimensional representation can generate natural-sounding speech. Together, ARTI-6 provides an interpretable, computationally efficient, and physiologically grounded framework for advancing articulatory inversion, synthesis, and broader speech technology applications. The source code and speech samples are publicly available.

![overall_architecture](docs/src/overview_architecture.png)

## Data
We use the dataset from [A long-form single-speaker real-time MRI speech dataset and benchmark](https://arxiv.org/abs/2509.14479). It can be downloaded from [here](https://sail.usc.edu/span/single_spk/).

## Sample Page
Speech samples are available [here](https://lee-jhwn.github.io/arti-6/ "speech samples").

## Training
Coming soon...

## Inference
Coming soon...

## Reference
