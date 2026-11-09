import numpy as np, soundfile as sf, librosa, pedalboard as pb, sys
sys.path.insert(0, sys.argv[1]+'/analyse'); import x42tune
P=sys.argv[1]; SR=44100
v,_=sf.read(P+'/stems/htdemucs_ft/v3/vocals.wav',dtype='float32'); x=v[int(67*SR):int(87*SR)].T.copy()
x=pb.Pedalboard([pb.HighpassFilter(95)])(x,SR)
def meas(y):
    f0,vf,vp=librosa.pyin(y,fmin=70,fmax=700,sr=SR,frame_length=2048,hop_length=512)
    sel=vf&(vp>0.8); mid=librosa.hz_to_midi(f0[sel]); dev=np.abs(mid-np.round(mid))*100
    return 'n %4d dev %3.0f ct  <30ct %3.0f%%  onscale %3.0f%%'%(sel.sum(),np.median(dev),100*np.mean(dev<30),100*np.mean(np.isin(np.round(mid)%12,[1,3,4,6,8,9,11])))
print('raw     ',meas(x.mean(0)))
p=pb.load_plugin('/usr/lib/vst3/Auburn Sounds Graillon 3.vst3')
for k,o in {"c":0,"c_sharp":1,"d":0,"d_sharp":1,"e":1,"f":0,"f_sharp":1,"g":0,"g_sharp":1,"a":1,"a_sharp":0,"b":1}.items(): setattr(p,'allow_'+k,bool(o))
p.correction=True; p.corr_amount=100.0; p.dry_mix_db=-np.inf; p.smooth=1.0; p.inertia=0.0
print('graillon',meas(p(x,SR).mean(0)))
y,_=x42tune.tune(x.mean(0),SR,filt=0.02,fast=1); print('x42     ',meas(y))
