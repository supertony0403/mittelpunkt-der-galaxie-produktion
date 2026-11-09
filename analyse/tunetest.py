import numpy as np, soundfile as sf, librosa, pedalboard as pb, sys
P=sys.argv[1]; SR=44100
v,_=sf.read(P+'/stems/htdemucs_ft/v3/vocals.wav',dtype='float32'); x=v[int(67*SR):int(87*SR)].T.copy()
x=pb.Pedalboard([pb.HighpassFilter(95)])(x,SR)
SC={"c":0,"c_sharp":1,"d":0,"d_sharp":1,"e":1,"f":0,"f_sharp":1,"g":0,"g_sharp":1,"a":1,"a_sharp":0,"b":1}
def meas(y):
    f0,vf,_=librosa.pyin(y.mean(0),fmin=70,fmax=700,sr=SR,frame_length=2048,hop_length=512)
    mid=librosa.hz_to_midi(f0[vf]); dev=np.abs(mid-np.round(mid))*100
    return 'dev %3.0f ct  <30ct %3.0f%%  onscale %3.0f%%  voiced %d'%(np.median(dev),100*np.mean(dev<30),100*np.mean(np.isin(np.round(mid)%12,[1,3,4,6,8,9,11])),len(mid))
print('raw     ',meas(x))
for cfg in [dict(smooth=6.0,inertia=10.0),dict(smooth=1.0,inertia=0.0),dict(smooth=1.0,inertia=0.0,snap_min_st=0.0,snap_max_st=8.0,reduce=0.0),dict(smooth=1.0,inertia=0.0,pitch_engine='Graillon2')]:
    p=pb.load_plugin('/usr/lib/vst3/Auburn Sounds Graillon 3.vst3')
    for k,o in SC.items(): setattr(p,'allow_'+k,bool(o))
    p.correction=True; p.corr_amount=100.0; p.dry_mix_db=-np.inf
    try:
        for k,val in cfg.items(): setattr(p,k,val)
    except Exception as e: print('ERR',e); continue
    y=p(x,SR); print(cfg, meas(y))
print('engines', pb.load_plugin('/usr/lib/vst3/Auburn Sounds Graillon 3.vst3').parameters['pitch_engine'].valid_values)
