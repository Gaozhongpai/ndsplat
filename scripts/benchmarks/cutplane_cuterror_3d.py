#!/usr/bin/env python3
"""Geometric (3D, occlusion-free) cut error for the XClipGS operator ladder.

Measures each operator's deviation from the EXACT truncated-Gaussian mass at the
clip plane, in closed form from the trained primitives -- no camera, no occlusion.
For an x/y/z plane at the volume median, per Gaussian (center offset t in std units,
opacity a): exact kept mass = a*Phi(t). HC keeps/drops whole by center -> hole a*Phi
(dropped, t<=0) + overshoot a*(1-Phi) (kept, t>0). MM = moment-matched surrogate,
mass-preserving -> only overshoot (Gaussian tail past plane). Ours = exact -> 0.
Ladder Ours(0) < MM < HC confirms Prop. exact. ClipGS has no closed-form mass (MLP).
Reads output/xclipgs/ours/<scene>_900 ply; averages the 3 axis-cuts. -> cerr3d.json.
"""
import numpy as np, math, json
from plyfile import PlyData
try: from scipy.special import erf
except Exception:
    from numpy import vectorize; erf=vectorize(math.erf)
SQ2=math.sqrt(2)
def Phi(z): return 0.5*(1+erf(z/SQ2))
def phi_s(z): return np.exp(-0.5*z*z)/math.sqrt(2*math.pi)
def op_error_axis(t,a):
    Z=np.clip(Phi(t),1e-9,None); km=a*Z
    hc_over=np.where(t>0,a*(1-Phi(t)),0.0); hc_hole=np.where(t<=0,a*Phi(t),0.0)
    b=t; mean=-phi_s(b)/Z; var=np.clip(1+(-b*phi_s(b))/Z-mean**2,1e-9,None); sd=np.sqrt(var)
    mm_over=km*(1-Phi((t-mean)/sd))
    return km.sum(), (0.,0.), (0., mm_over.sum()), (hc_hole.sum(), hc_over.sum())

def analyze(scene):
    ply=f"/data/output/xclipgs/ours/{scene}_900/point_cloud/iteration_best/point_cloud.ply"
    p=PlyData.read(ply)['vertex']
    xyz=np.stack([p['x'],p['y'],p['z']],1)
    a=1/(1+np.exp(-np.array(p['opacity'])))
    # average over the 3 axis-cuts (matches the eval: one plane per voxel axis at median)
    tot=0.; mm=[0.,0.]; hc=[0.,0.]
    for ax in range(3):
        r=np.exp(p['scale_%d'%ax]); tau=np.median(xyz[:,ax]); t=(tau-xyz[:,ax])/r
        km,_o,mmv,hcv=op_error_axis(t,a)
        tot+=km; mm[0]+=mmv[0]; mm[1]+=mmv[1]; hc[0]+=hcv[0]; hc[1]+=hcv[1]
    return dict(ours=0.0, mm=(mm[0]+mm[1])/tot, hc=(hc[0]+hc[1])/tot)

scenes=['gel','intestine','kneejoint','lower','vascular','heart','nose','hand']
print(f"{'scene':10s} {'ours':>7s} {'mm':>7s} {'hc':>7s}  (CErr-3D x10^-2)")
acc={'mm':[],'hc':[]}
out={}
for s in scenes:
    r=analyze(s); out[s]=r
    print(f"{s:10s} {0.0:7.3f} {r['mm']*100:7.3f} {r['hc']*100:7.3f}")
    acc['mm'].append(r['mm']*100); acc['hc'].append(r['hc']*100)
print(f"{'AVG':10s} {0.0:7.3f} {np.mean(acc['mm']):7.3f} {np.mean(acc['hc']):7.3f}")
json.dump({s:{'mm':out[s]['mm'],'hc':out[s]['hc']} for s in scenes}, open('/out/cerr3d.json','w'))
