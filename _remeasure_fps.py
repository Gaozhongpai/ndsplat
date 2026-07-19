# Robust FPS re-measurement for the XClipGS operators. Loads each trained
# checkpoint (iteration_best), renders ALL 90 held-out test views multiple
# passes, DISCARDS warmup, times with cuda.synchronize, reports the median
# per-frame FPS. Run SERIALLY on ONE idle GPU so no cross-job contention skews
# it (the original first-20-frames measurement caught transient contention, e.g.
# gel MM=235). Overwrites fps.txt + patches "FPS" in results.json.
import os, sys, json, time, glob
import numpy as np, torch
sys.argv=['x']
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, get_gaussian_model

DATA='/data'
OPS=['ours','clipgs','mm','hc','ours_nomip']
SCENES=['gel','intestine','kneejoint','lower','vascular','heart','nose','hand']
WARMUP=30      # frames to discard
PASSES=3       # passes over the 90 test views (270 timed frames after warmup)

def measure(op, scene):
    mp=f'{DATA}/output/xclipgs/{op}/{scene}_900'
    if not os.path.isfile(f'{mp}/results.json'): return None
    # read cfg for mode + params
    cfg=open(f'{mp}/cfg_args').read()
    mode='clipgs' if op=='clipgs' else ('dgs' if op in('ours','mm','hc','ours_nomip') else '3dgs')
    ap=ArgumentParser(); lp=ModelParams(ap); pp=PipelineParams(ap)
    args=ap.parse_args(['-s',f'{DATA}/nerf_dataset/{scene}_900','--eval'])
    ds=lp.extract(args); ds.mode=mode; ds.model_path=mp
    # model params from cfg (input_dim etc.) — pull a few
    for kv in ['input_dim','use_view_dependent_pos','l_22_inv_init_scale','lambda_init','lambda_opc','use_opacity_pos_decouple']:
        import re
        m=re.search(rf"{kv}=([^,\)]+)", cfg)
        if m:
            val=m.group(1).strip()
            try: val=eval(val)
            except: pass
            setattr(ds,kv,val)
    G=get_gaussian_model(mode)
    if mode=='clipgs': g=G(ds.sh_degree, deform_scale=getattr(ds,'clipgs_deform_scale',False))
    elif mode=='dgs': g=G(ds.sh_degree, input_dim=getattr(ds,'input_dim',6), use_view_dependent_pos=getattr(ds,'use_view_dependent_pos',False), use_opacity_pos_decouple=getattr(ds,'use_opacity_pos_decouple',False), l_22_inv_init_scale=getattr(ds,'l_22_inv_init_scale',2.0), lambda_init=getattr(ds,'lambda_init',-1.2), lambda_opc=getattr(ds,'lambda_opc',0.35))
    else: g=G(ds.sh_degree)
    scene_obj=Scene(ds, g, load_iteration=-1, shuffle=False, load_train_cameras=False)  # best
    if op=='clipgs': g.clip_operator='clipgs'
    else: g.clip_operator={'ours':'analytic','mm':'moment','hc':'hardcull','ours_nomip':'analytic'}[op]
    cams=scene_obj.getTestCameras()
    bg=torch.zeros(3,device='cuda')
    from render import render_wrapper
    class P: convert_SHs_python=False; compute_cov3D_python=False; debug=False
    times=[]
    idx=0
    for p in range(PASSES):
        for cam in cams:
            torch.cuda.synchronize(); t=time.time()
            _=render_wrapper(cam, g, P(), bg, mode, is_test=False)
            torch.cuda.synchronize(); dt=time.time()-t
            if idx>=WARMUP: times.append(dt)
            idx+=1
    fps=1.0/np.median(times)
    return fps

results={}
for op in OPS:
    for s in SCENES:
        try:
            fps=measure(op,s)
            if fps is None: continue
            results[(op,s)]=fps
            # write fps.txt (best) + patch results.json
            mp=f'{DATA}/output/xclipgs/{op}/{s}_900'
            bestdir=glob.glob(f'{mp}/point_cloud/iteration_best')
            fp=f'{mp}/test/ours_best/fps.txt'
            if os.path.isdir(os.path.dirname(fp)): open(fp,'w').write(f'{fps:.2f}')
            rj=json.load(open(f'{mp}/results.json'))
            def patch(d):
                for k,v in d.items():
                    if isinstance(v,dict):
                        if 'PSNR' in v: v['FPS']=round(fps,2)
                        else: patch(v)
            patch(rj)
            json.dump(rj, open(f'{mp}/results.json','w'), indent=2)
            print(f'{op:11s} {s:10s} FPS={fps:.1f}', flush=True)
        except Exception as e:
            print(f'{op:11s} {s:10s} ERR {e}', flush=True)
print('REMEASURE_DONE')
