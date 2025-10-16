import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr threads

import yaml
import torch
import torch.nn.functional as F
import librosa
from huggingface_hub import hf_hub_download
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier
from inversion.wavlm_articulatory import WavLMWrapper
from synthesis.articulatory_synthesis import Generator, load_checkpoint, AttrDict, MAX_WAV_VALUE

class ARTI6():
    def __init__(self, backbone='wavlm_large', spk_model='ecapa',
                 finetune_method='lora', lora_rank=16, freeze_params=True, use_conv_output=True, device='cpu'):
        self.backbone = backbone
        self.spk_model= spk_model
        self.finetune_method = finetune_method
        self.lora_rank = lora_rank
        self.freeze_params = freeze_params
        self.use_conv_output = use_conv_output
        self.device = device
        self.invert_model = None
        self.synthesis_model = None
        self.spk_encoder = None


    def load_model(self, mode='all', invert_ckpt='inversion_flt_ckpt.pt', synthesis_ckpt='generator.pt', from_huggingface=True):


        if mode in ['all','both','invert','inversion']: # load articulatory inversion model

            if invert_ckpt is None:
                AssertionError("No checkpoint provided")

            if self.backbone == "wavlm" or self.backbone == "wavlm_large":
                self.invert_model = WavLMWrapper(
                    pretrain_model=self.backbone, 
                    finetune_method=self.finetune_method,
                    lora_rank=self.lora_rank, 
                    freeze_params=self.freeze_params, 
                    use_conv_output=self.use_conv_output,
                ).to(self.device)

            if from_huggingface:
                invert_lora_ckpt = hf_hub_download(repo_id='lee-jhwn/arti-6', filename=invert_ckpt.replace('flt','lora'))
                invert_ckpt = hf_hub_download(repo_id='lee-jhwn/arti-6', filename=invert_ckpt)

            self.invert_model.load_state_dict(torch.load(invert_ckpt, weights_only=True, map_location=self.device), strict=False)
            if self.finetune_method == "lora": 
                self.invert_model.load_state_dict(torch.load(invert_ckpt.replace('filtered','lora'), map_location=self.device), strict=False)
            
            else: #TODO
                NotImplementedError()
            
            if self.spk_model == 'ecapa':
                self.spk_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":self.device})
            else: #TODO
                NotImplementedError()
        
            self.invert_model.eval()
            self.spk_encoder.eval()
        
        if mode in ['all','both','synthesis']: # load articulatory synthesis model
            
            if synthesis_ckpt is None:
                AssertionError("No checkpoint provided")
            with open('./arti6/synthesis/config_synthesis.yml') as f:
                config = yaml.safe_load(f)
            h = AttrDict(config['model_args'])
            torch.manual_seed(h.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(h.seed)
            self.synthesis_model = Generator(h)
            self.synthesis_model.to(self.device)
            if from_huggingface:
                synthesis_ckpt = hf_hub_download(repo_id='lee-jhwn/arti-6', filename=synthesis_ckpt)
            state_dict_g = load_checkpoint(synthesis_ckpt, self.device)
            self.synthesis_model.load_state_dict(state_dict_g['generator'])
            self.synthesis_model.eval()
            self.synthesis_model.remove_weight_norm()


    def extract_spkemb(self, wav):
        
        spk_emb = self.spk_encoder.encode_batch(wav).squeeze().squeeze() # (B, 192); B=1
        spk_emb = F.normalize(spk_emb, p=2, dim=0) # L2 norm
        spk_emb = spk_emb.unsqueeze(0) # (1, 192)
        
        return spk_emb

    def load_wav(self, wav_path, sr=16000):

        wav, _ = librosa.load(wav_path, sr=sr)
        wav = torch.tensor(wav).unsqueeze(0).to(self.device)
        return wav
    
    def invert(self, wav_path): # TODO: add batch processing
        
        wav = self.load_wav(wav_path)
        with torch.no_grad():
            arti_feats = self.invert_model(wav)
            spk_emb = self.extract_spkemb(wav)

        return {'arti_feats': arti_feats, # (B, T, D); B=1 and D=6
                'spk_emb': spk_emb # 192 dim
                }
        

    def synthesize(self, arti_feats, spk_emb): # TODO: add batch processing

        with torch.no_grad():

            arti_feats = torch.autograd.Variable(arti_feats.to(self.device, non_blocking=True))
            spk_emb = torch.autograd.Variable(spk_emb.to(self.device, non_blocking=True))

            y_g_hat = self.synthesis_model(arti_feats.transpose(1,2), g=spk_emb) # (B, 1, T)
            audio = y_g_hat.squeeze() # (B, T); B=1
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')    

        return audio 


if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    arti6_model = ARTI6(device=device)
    arti6_model.load_model()
    articulatory_feats = arti6_model.invert(wav_path='example_gt.wav'
)
    synthesized_audio = arti6_model.synthesize(articulatory_feats['arti_feats'], articulatory_feats['spk_emb'])
    sf.write('example_out.wav', synthesized_audio, 16000)

