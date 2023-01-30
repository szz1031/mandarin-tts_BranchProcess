import argparse
import os
import subprocess
import re

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from mtts.models.fs2_model import FastSpeech2
from mtts.models.vocoder import *
from mtts.text import TextProcessor
from mtts.utils.logging import get_logger
from mtts.text.gp2py import TextNormal
#from mtts.synthesize import *

logger = get_logger(__file__)


def check_ffmpeg():
    r, path = subprocess.getstatusoutput("which ffmpeg")
    return r == 0


with_ffmpeg = check_ffmpeg()


def build_vocoder(device, config):
    vocoder_name = config['vocoder']['type']
    VocoderClass = eval(vocoder_name)
    model = VocoderClass(**config['vocoder'][vocoder_name])
    return model


def normalize(wav):
    assert wav.dtype == np.float32
    eps = 1e-6
    sil = wav[1500:2000]
    #wav = wav - np.mean(sil)
    #wav = (wav - np.min(wav))/(np.max(wav)-np.min(wav)+eps)
    wav = wav / np.max(np.abs(wav))
    #wav = wav*2-1
    wav = wav * 32767
    return wav.astype('int16')


def to_int16(wav):
    wav = wav = wav * 32767
    wav = np.clamp(wav, -32767, 32768)
    return wav.astype('int16')


def CleanText(text:str)->str:
    text=text.replace(",","")
    text=text.replace("."," ")
    text=text.replace("，","")
    text=text.replace("。","")
    text=text.replace("、","")
    text=text.replace(" ","")
    text=text.replace("!","")
    text=text.replace("！","")
    text=text.replace("?","")
    text=text.replace("？","")
    

    s=re.sub('[a-zA-Z0-9]','',text)

    #print(s)
    return s

def TextGen(text:str,filename='test0',speakerId='0')->str:
    text=CleanText(text)
    
    tn = TextNormal('gp.vocab', 'py.vocab', add_sp1=True, fix_er=True)
    py_list, gp_list = tn.gp2py(text)
    for py, gp in zip(py_list, gp_list):
        with open("temp.txt","w+",encoding='utf-8') as f:
            f.write(filename+"|"+ py + '|' + gp+"|"+str(speakerId))
        
        return(filename+"|"+ py + '|' + gp+"|"+str(speakerId))
    
def TextToWave(text):

    checkpoint="./checkpoints/checkpoint_1350000.pth.tar"
    duration=1.0
    _config='./config.yaml'
    device=torch.device('cpu')
    output_dir='./outputs/'


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(_config) as f:
        config = yaml.safe_load(f)
        logger.info(f.read())

    sr = config['fbank']['sample_rate']

    vocoder = build_vocoder(device, config)
    text_processor = TextProcessor(config)
    model = FastSpeech2(config)

    if checkpoint != '':
        sd = torch.load(checkpoint, map_location=device)
        if 'model' in sd.keys():
            sd = sd['model']
    model.load_state_dict(sd)
    del sd  # to save mem
    model = model.to(device)
    torch.set_grad_enabled(False)

    
    line=text

    if len(line) == 0 or line.startswith('#'):
        pass
    logger.info(f'processing {line}')
    name, tokens = text_processor(line)
    tokens = tokens.to(device)
    seq_len = torch.tensor([tokens.shape[1]])
    tokens = tokens.unsqueeze(1)
    seq_len = seq_len.to(device)
    max_src_len = torch.max(seq_len)
    output = model(tokens, seq_len, max_src_len=max_src_len, d_control=duration)
    mel_pred, mel_postnet, d_pred, src_mask, mel_mask, mel_len = output

    # convert to waveform using vocoder
    mel_postnet = mel_postnet[0].transpose(0, 1).detach()
    mel_postnet += config['fbank']['mel_mean']
    wav = vocoder(mel_postnet)
    if config['synthesis']['normalize']:
        wav = normalize(wav)
    else:
        wav = to_int16(wav)
    dst_file = os.path.join(output_dir, f'{name}.wav')
    #np.save(dst_file+'.npy',mel_postnet.cpu().numpy())
    logger.info(f'writing file to {dst_file}')
    wavfile.write(dst_file, sr, wav)
        
def TextToWaveViaCmd():
    cmd='python D:\GitHub\mandarin-tts_BranchProcess\mtts/synthesize.py  -d cpu -c D:\GitHub\mandarin-tts_BranchProcess\examples\\aishell3\config.yaml --checkpoint D:\GitHub\mandarin-tts_BranchProcess\examples\\aishell3/checkpoints\checkpoint_1350000.pth.tar -i temp.txt'
    os.system(cmd)
    
    pass  
    
if __name__ == '__main__':
    text=TextGen("他们怎么只排了一个人来来我还需要援军",filename='test001',speakerId='0')
    TextToWave(text)
    # for i in range(0,100):
    #     text=TextGen("声线测试",filename=str(i),speakerId=str(i))
    #     TextToWave(text)