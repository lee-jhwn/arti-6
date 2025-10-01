import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr threads

import yaml
import torch
import librosa
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


    def load_model(self, mode='all', invert_ckpt=None, synthesis_ckpt=None):

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

            self.invert_model.load_state_dict(torch.load(invert_ckpt, weights_only=True), strict=False)
            if self.finetune_method == "lora": 
                self.invert_model.load_state_dict(torch.load(invert_ckpt.replace('filtered_epoch', 'lora_epoch')), strict=False)
            
            else: #TODO
                NotImplementedError()
            
            if self.spk_model == 'ecapa':
                self.spk_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb").to(self.device)

            else: #TODO
                NotImplementedError()
        
            self.invert_model.eval()
            self.spk_encoder.eval()
        
        if mode in ['all','both','synthesis']: # load articulatory synthesis model
            
            if synthesis_ckpt is None:
                AssertionError("No checkpoint provided")
            with open('./synthesis/config_synthesis.yml') as f:
                config = yaml.safe_load(f)
            h = AttrDict(config['model_args'])
            self.synthesis_model = Generator(h).to(self.device)
            self.synthesis_model.load_state_dict(load_checkpoint(synthesis_ckpt, self.device)['generator'])
            self.synthesis_model.eval()
            self.synthesis_model.remove_weight_norm()

        


    def extract_spkemb(self, wav):
        
        spk_emb = self.spk_encoder.encode_batch(wav).squeeze(0) # (B, 192); B=1

        return spk_emb

    def load_wav(self, wav_path, sr=16000):

        wav, _ = librosa.load(wav_path, sr=sr)
        wav = torch.tensor(wav).float().unsqueeze(0).to(self.device)

        return wav
    
    def invert(self, wav_path): # TODO: add batch processing
        
        wav = self.load_wav(wav_path)
        with torch.no_grad():
            arti_feats = self.invert_model(wav)
            spk_emb = self.extract_spkemb(wav)

        print(arti_feats.shape)
        print(spk_emb.shape)

        return {'arti_feats': arti_feats, # (B, T, D); B=1 and D=6
                'spk_emb': spk_emb # 192 dim
                }
        

    def synthesize(self, arti_feats, spk_emb): # TODO: add batch processing

        with torch.no_grad():
            y_g_hat = self.synthesis_model(arti_feats.transpose(1,2), g=spk_emb) # (B, 1, T)
            audio = y_g_hat.squeeze(1) # (B, T); B=1
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')    
            print(audio.shape)

        return audio # TODO


if __name__=="__main__":

    arti6_model = ARTI6()

    arti6_model.load_model(mode='all', invert_ckpt='/home/jihwan/scratch_jhwn/jihwan/projects/2509_mriarti/code/inversion/vox-profile/log/articulatory_inversion_wavlm_jhwn_v1/wavlm_large/lr00005_ep50_lora_16_accumulation_frozen/best_filtered_epoch_40.pt',
                           synthesis_ckpt='/home/jihwan/scratch_jhwn/jihwan/projects/2509_mriarti/code/hifi-gan/logs/config_v2_libri_only/g_01200000')
    articulatory_feats = arti6_model.invert(wav_path='/home/jihwan/scratch_jhwn/jihwan/projects/2509_mriarti/code/hifi-gan/samples_mos/candidates/121_121726_000046_000003_gt.wav')
    synthesized_audio = arti6_model.synthesize(articulatory_feats['arti_feats'], articulatory_feats['spk_emb'])
    sf.write('test_synthesized.wav', synthesized_audio[0], 16000)

