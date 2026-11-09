import numpy as np, soundfile as sf, librosa, pedalboard as pb, sys
sys.path.insert(0, sys.argv[1]+'/analyse'); import x42tune
P=sys.argv[1]; SR=44100
v,_=sf.read(P+'/stems/htdemucs_ft/v3/vocals.wav',dtype='float32'); x=v[int(67*SR):int(87*SR)].T.copy()
x=pb.Pedalboard([pb.HighpassFilter(95)])(x,SR).mean(0)
def meas(y):
    f0,vf,_=librosa.pyin(y,fmin=70,fmax=700,sr=SR,frame_length=2048,hop_length=512)
    mid=librosa.hz_to_midi(f0[vf]); dev=np.abs(mid-np.round(mid))*100
    return 'dev %3.0f ct  <30ct %3.0f%%  onscale %3.0f%%'%(np.median(dev),100*np.mean(dev<30),100*np.mean(np.isin(np.round(mid)%12,[1,3,4,6,8,9,11])))
for f in (0.1,0.02):
    y,lat=x42tune.tune(x,SR,filt=f,fast=1 if f<0.05 else 0)
    print('x42 filter',f,'lat',lat,meas(y), 'corr w/ input %.2f'%np.corrcoef(x,y)[0,1])
    if f==0.02: sf.write(P+'/analyse/x42test.wav',y,SR)
