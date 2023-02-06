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

import csv
from pprint import pprint
import pandas as pd
#from mtts.synthesize import *

logger = get_logger(__file__)


def nameToSpeakerId(SpeakerName:str)->str:
    '''
    指定声线配置
    '''
    
    if SpeakerName=="泽姬":
        return "35"
    if SpeakerName=="扎娜":
        return "4"
    if SpeakerName=="路易莎":
        return "0"
    
    if SpeakerName=="卡泽尔":
        return "10"
    
    if SpeakerName=="阿瑞纳":
        return "40"
    
    return "10"


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
    '''
    清除所有字符和数字，并且将句中所有的符号转化为重复字来模拟停顿
    '''
    text=text.replace('...','.')
    text=text.replace('锢','固')
    text=text.replace('嘻','息')
    text=text.replace('……','.')
    text=text.replace('…','.')
    text=text.replace('）','')
    text=text.replace('（','')
    text=text.replace('(','')
    text=text.replace(')','')
    text=text.replace('祂','鱼')
    text=text.replace('臾','鱼')
    
    
    
    s=re.sub('[a-zA-Z0-9]','',text) # 去除字母数字
    
    punctuationList=[',','.','!','?',   '。','，','！','？','、',' ',   '"','“','”','[',']','(',')','~','（','）','嗯','儿','……','…'] # 所有标点符号 和 暂时无法处理的语气词
    
    if s=='' or s==None:
        return""
    
    _str=""
    for i in s[:-1]:
        if i not in punctuationList:
            _str=_str+i
        else:
            if _str!="" and _str[-1] not in punctuationList:
                _str=_str+_str[-1]

    if s[-1] not in punctuationList:
        _str=_str+s[-1]
        
        
    if _str=='' or _str==None:
        return""
    
    print(_str)
    return _str

def TextGen(text:str,filename='test0',speakerId='0')->str:
    '''
    将文本转化为拼音并且格式化
    '''
    text=CleanText(text)
    
    tn = TextNormal('gp.vocab', 'py.vocab', add_sp1=True, fix_er=True)
    py_list, gp_list = tn.gp2py(text)
    for py, gp in zip(py_list, gp_list):
        return(filename+"|"+ py + '|' + gp+"|"+str(speakerId))
    
def TextToWave(text,output_dir='./outputs/'):

    checkpoint="./checkpoints/checkpoint_1350000.pth.tar"
    duration=1.0
    _config='./config.yaml'
    device=torch.device('cpu')
    


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(_config) as f:
        config = yaml.safe_load(f)
        #logger.info(f.read())

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

    if not line:
        return
    
    if len(line) == 0 or line.startswith('#'):
        return
    #logger.info(f'processing {line}')
    name, tokens = text_processor(line)
    dst_file = os.path.join(output_dir, name.replace(".","_")+'.wav')
    if os.path.isfile(dst_file):
        logger.info(f'Skipping {dst_file}')
        return
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
    
    #np.save(dst_file+'.npy',mel_postnet.cpu().numpy())
    logger.info(f'writing file to {dst_file}')
    wavfile.write(dst_file, sr, wav)
        
def TextToWave_Branch(textList,output_dir='./outputs/'):

    checkpoint="./checkpoints/checkpoint_1350000.pth.tar"
    duration=1.0
    _config='./config.yaml'
    device=torch.device('cpu')
    


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(_config) as f:
        config = yaml.safe_load(f)
        #logger.info(f.read())

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


    if not textList:
        return
    
    for line in textList:
        if not line or len(line) == 0 or line.startswith('#'):
            continue
        #logger.info(f'processing {line}')
        name, tokens = text_processor(line)
        dst_file = os.path.join(output_dir, name.replace(".","_")+'.wav')
        if os.path.isfile(dst_file):
            logger.info(f'Skipping {dst_file}')
            continue
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
        
        #np.save(dst_file+'.npy',mel_postnet.cpu().numpy())
        logger.info(f'writing file to {dst_file}')
        wavfile.write(dst_file, sr, wav)

def ProcessCsv(filePath):
    rootPath=os.path.dirname(filePath)
    data=[]
    csv_reader = csv.reader(open(filePath,encoding='utf-8'))
    
    for line in csv_reader:
        if line[0]!='---' and line[0]!='' and line[5]!='' and line[0] and line[1]!='GroupName':
            data.append(line)
    
    lastname=''
    count=0
    _eventData=[]
    textList=[]
    for item in data:
        name=item[1]
        if name!=lastname:
            lastname=name
            count=1
            truename=lastname+'_'+str(count)
        else:
            count+=1
            truename=lastname+'_'+str(count)
            
        speaker=item[4]#[:-2].split('"')[-1]
        text=item[5]#[:-2].split('"')[-1]
        print (speaker + ":" +text)
        text2=TextGen(text,truename,nameToSpeakerId(speaker))
        textList.append(text2)
        _eventData.append([item[0],truename])
    
    TextToWave_Branch(textList,rootPath+"/outputs/")
    #print(_eventData)
    reWriteCsv(_eventData,filePath)
    
def reWriteCsv(datalist,filePath):
    '''
    把生成的wave的信息，编写成akevent路径，并回填CSV文件。
    每组信息格式为 [原csv行数，event命名]
    '''
    df=pd.read_csv(filePath,encoding='utf-8',index_col=0)
    #pprint(df)
    # df.reset_index(inplace=True)
    # pprint(df)
    # 莫名其妙会读出小数点
    _index_type=df.index.inferred_type
    print("index type="+str(_index_type))
    
    path,filename=os.path.split(filePath)
    for item in datalist:
        #print(type(item[0]))
        if str(_index_type)=="integer":
            df.loc[ int(item[0]),'SoundPath']=makeAkEventPath(item[1], filename[12:-4]) # 去掉CSV的前缀和后缀
        if str(_index_type)=="str":
            df.loc[ str(item[0]),'SoundPath']=makeAkEventPath(item[1], filename[12:-4]) # 去掉CSV的前缀和后缀
        
    df.to_csv(filePath,encoding='utf-8')
    logger.info("Processing Finished")
    pass

def makeAkEventPath(eventName:str,CsvName):
    
    if not eventName or eventName=="":
        return ""
    
    #参考
    #AkAudioEvent'/Game/WwiseAudio/Events/Dialog/FT/(CSV)/Event_FT_Relics.Event_FT_Relics'
    #AkAudioEvent'/Game/WwiseAudio/Events/Dialog/FT/FreeTalk_Prologue_TheImperialWall/Event.Event'
    
    _str="AkAudioEvent'/Game/WwiseAudio/Events/Dialog/FT/"+CsvName+"/Play_"+eventName.replace(".",'_')+".Play_"+eventName.replace(".",'_')+"'"
    
    return _str
    
if __name__ == '__main__':
    ProcessCsv(r"F:\kkxszz_DevMain\Projectlsa\Resources\DataTable\Quest\FreeTalk\DT_FreeTalk_Prologue_SlumPort.csv")
    
    
    #text=TextGen("...",filename='test001',speakerId=nameToSpeakerId("卡泽尔"))
    #TextToWave(text)
    # for i in range(0,100):
    #     text=TextGen("声线测试",filename=str(i),speakerId=str(i))
    #     TextToWave(text)
    
    
    
    