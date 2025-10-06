import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr threads

import sys

sys.path.append("arti6")

import soundfile as sf
import torch
from arti6 import ARTI6


device = "cuda" if torch.cuda.is_available() else "cpu"

arti6_model = ARTI6(device=device)
arti6_model.load_model()
articulatory_feats = arti6_model.invert(wav_path='./example_gt.wav')
synthesized_audio = arti6_model.synthesize(articulatory_feats['arti_feats'], articulatory_feats['spk_emb'])
sf.write('./example_arti6.wav', synthesized_audio, 16000)