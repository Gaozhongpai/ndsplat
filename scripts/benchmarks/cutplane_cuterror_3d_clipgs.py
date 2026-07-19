import json,math,numpy as np,torch
from torch import nn
from plyfile import PlyData
try: from scipy.special import erf
except Exception:
    from numpy import vectorize; erf=vectorize(math.erf)
SQ2=math.sqrt(2)
def Phi(z): return 0.5*(1+erf(z/SQ2))
class MLP(nn.Module):
    def __init__(s,ds=False,osc=1e-3):
        super().__init__(); s.osc=osc; out=6 if ds else 3
        s.net=nn.Sequential(nn.Linear(4,64),nn.ReLU(True),nn.Linear(64,64),nn.ReLU(True),nn.Linear(64,out))
    def forward(s,x,sd): o=s.net(torch.cat([x,sd],1))*s.osc; return o[:,:3],None
def load(model,scene,d="/data/output/xclipgs"):
    p=PlyData.read(f"{d}/{model}/{scene}_900/point_cloud/iteration_best/point_cloud.ply")['vertex']
    xyz=np.stack([p['x'],p['y'],p['z']],1).astype(np.float64)
    sc=np.exp(np.stack([p['scale_0'],p['scale_1'],p['scale_2']],1)).astype(np.float64)
    a=(1/(1+np.exp(-np.array(p['opacity'])))).astype(np.float64); return xyz,sc,a
def clip_op(scene,xyz,sc,a,d="/data/output/xclipgs"):
    ck=torch.load(f"{d}/clipgs/{scene}_900/point_cloud/iteration_best/deform_mlp.pt",map_location="cpu")
    m=MLP(ds=ck.get("deform_scale",False)); m.load_state_dict(ck["state_dict"]); m.eval()
    xt=torch.tensor(xyz,dtype=torch.float32); H=O=T=0.0
    for ax in range(3):
        nt=torch.tensor(np.eye(3)[ax],dtype=torch.float32); tau=np.median(xyz[:,ax])
        ek=a*Phi((tau-xyz[:,ax])/sc[:,ax])
        with torch.no_grad(): dxyz,_=m(xt,(xt@nt-float(tau)).unsqueeze(1))
        defc=((xt+dxyz)@nt).numpy(); kept=np.where(defc<=tau,a,0.0)
        H+=np.clip(ek-kept,0,None).sum();O+=np.clip(kept-ek,0,None).sum();T+=ek.sum()
    return (H+O)/T
scenes=['gel','intestine','kneejoint','lower','vascular','heart','nose','hand']
print("ClipGS operator (MLP trained on clipgs-cloud) applied to the OURS cloud:")
print(f"{'scene':10s} {'ClipGS/ours-geom':>16s}")
vals=[]
for s in scenes:
    xyz,sc,a=load("ours",s)
    c=clip_op(s,xyz,sc,a); vals.append(c*100)
    print(f"{s:10s} {c*100:16.3f}")
print(f"{'AVG':10s} {np.mean(vals):16.3f}")
print("(now directly comparable to HC=1.19 on the same ours cloud)")
