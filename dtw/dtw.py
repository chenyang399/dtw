
import numpy
import librosa
from basic_operator import *


yes1 = "yes1.wav"
no2  =  "no2.wav"
yes3 =  "yes3.wav"
def mfcc(path):
    data,fs=librosa.load(path)
    # print(data)
    # print(data.shape)
    # print(fs)
    step1   =   pre_emphasis(data) 
    # print(step1)
    # print(step1.shape)
    step2   =   framing(step1,fs) 
    # print(step2)
    # print(step2.shape)
    step3   =   add_window(step2,fs)
    # print(step3)
    # print(step3.shape)
    step4   =   stft(step3) 
    # print(step4)
    # print(step4.shape)
    step5   =   mel_filter(step4, fs) 
    # print(step5)
    # print(step5.shape)
    fbank   =   log_pow(step5) 
    # print(fbank)
    # print(fbank.shape)
    mfcc  = discrete_cosine_transform(fbank)
    return mfcc
    # print(mfcc)
    # print(mfcc.shape)

def fbank(path):
    data,fs=librosa.load(path)
    # print(data)
    # print(data.shape)
    # print(fs)
    step1   =   pre_emphasis(data) 
    # print(step1)
    # print(step1.shape)
    step2   =   framing(step1,fs) 
    # print(step2)
    # print(step2.shape)
    step3   =   add_window(step2,fs)
    # print(step3)
    # print(step3.shape)
    step4   =   stft(step3) 
    # print(step4)
    # print(step4.shape)
    step5   =   mel_filter(step4, fs) 
    # print(step5)
    # print(step5.shape)
    fbank   =   log_pow(step5) 
    # print(fbank)
    # print(fbank.shape)
    
    return fbank
    # print(mfcc)
    # print(mfcc.shape)

"""
DTWDistance(s1, s2) is copied from:
http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html
"""
 
def DTWDistance(s1, s2):
    DTW={}
    len1=s1.shape[0]
    len2=s2.shape[0]
    dist = np.zeros((len1,len2))
  
    for i in range(len1):
        for j in range(len2):
            dist[i][j]=(sum((s1[i][:]-s2[j][:])*(s1[i][:]-s2[j][:])))

 
    for i in range(len1):
        DTW[(i, -1)] = float('inf')
    for i in range(len2):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
 
    for i in range(len1):
        for j in range(len2):
            DTW[(i, j)] = dist[i][j] + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
 
    return np.sqrt(DTW[len1-1, len2-1])
 




mfcc_yes1 = fbank(yes1)
mfcc_no2 = fbank(no2)
mfcc_yes3 = fbank(yes3)
plot_spectrogram(mfcc_yes1,'fbank','yes1')
# plot_spectrogram(mfcc_no2)
# plot_spectrogram(mfcc_yes3)

print(DTWDistance(mfcc_yes1,mfcc_yes3))

print(DTWDistance(mfcc_no2,mfcc_yes3))
