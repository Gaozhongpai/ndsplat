

import json
import os

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
methods = ["ndgs", "opacity_only", "opacity_pos"]
method_names = ["N-DGS", "Opacity-Only", "Opacity+Pos"]

base_path = "/home/zhongpai/code/gaussian/workspace/6dgs-iclr/output"

results = {m: {s: {} for s in scenes} for m in methods}

for method in methods:
    for scene in scenes:
        path = os.path.join(base_path, method, "nerf_synthetic", scene, "results.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
                # Use ours_30000
                metrics = data.get("ours_30000", {})
                results[method][scene] = {
                    "PSNR": metrics.get("PSNR", 0.0),
                    "SSIM": metrics.get("SSIM", 0.0),
                    "LPIPS": metrics.get("LPIPS", 0.0),
                }
        except FileNotFoundError:
            print(f"Warning: Results not found for {method}/{scene}")
            results[method][scene] = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0}

# Calculate Means
means = {m: {"PSNR": [], "SSIM": [], "LPIPS": []} for m in methods}
for method in methods:
    for scene in scenes:
        means[method]["PSNR"].append(results[method][scene]["PSNR"])
        means[method]["SSIM"].append(results[method][scene]["SSIM"])
        means[method]["LPIPS"].append(results[method][scene]["LPIPS"])

mean_results = {m: {} for m in methods}
for method in methods:
    psnr_vals = means[method]["PSNR"]
    ssim_vals = means[method]["SSIM"]
    lpips_vals = means[method]["LPIPS"]
    
    mean_results[method]["PSNR"] = sum(psnr_vals) / len(psnr_vals) if psnr_vals else 0
    mean_results[method]["SSIM"] = sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0
    mean_results[method]["LPIPS"] = sum(lpips_vals) / len(lpips_vals) if lpips_vals else 0

# Print Table Rows
print(r"\toprule")
print(r"& \multicolumn{3}{c|}{\textbf{N-DGS}} & \multicolumn{3}{c|}{\textbf{Opacity-Only}} & \multicolumn{3}{c}{\textbf{Opacity+Pos}} \\")
print(r"Scene & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\")
print(r"\midrule")

for scene in scenes:
    row = f"\\texttt{{{scene}}}"
    
    psnrs = [results[m][scene]["PSNR"] for m in methods]
    ssims = [results[m][scene]["SSIM"] for m in methods]
    lpipss = [results[m][scene]["LPIPS"] for m in methods]
    
    max_psnr = max(psnrs)
    max_ssim = max(ssims)
    min_lpips = min(lpipss)

    for method in methods:
        val = results[method][scene]
        p = val["PSNR"]
        s = val["SSIM"]
        l = val["LPIPS"]
        
        p_str = f"{p:.2f}"
        s_str = f"{s:.3f}"
        l_str = f"{l:.3f}"
        
        # Bolding logic
        # Using a small epsilon for float comparison if needed, but strict equality usually works for exact values read from same source or if distinct enough.
        # Actually exact match might fail if floats are slightly different due to internal representation? No, they come from json.
        # But max() returns one of them. So equality should hold.
        
        if p == max_psnr: p_str = f"\\textbf{{{p_str}}}"
        if s == max_ssim: s_str = f"\\textbf{{{s_str}}}"
        if l == min_lpips: l_str = f"\\textbf{{{l_str}}}"
        
        row += f" & {p_str} & {s_str} & {l_str}"
    
    print(row + " \\\\")

print(r"\midrule")

# Print Mean Row
row = r"\textbf{\texttt{Mean}}"
psnrs = [mean_results[m]["PSNR"] for m in methods]
ssims = [mean_results[m]["SSIM"] for m in methods]
lpipss = [mean_results[m]["LPIPS"] for m in methods]

max_psnr = max(psnrs)
max_ssim = max(ssims)
min_lpips = min(lpipss)

for method in methods:
    p = mean_results[method]["PSNR"]
    s = mean_results[method]["SSIM"]
    l = mean_results[method]["LPIPS"]
    
    p_str = f"{p:.2f}"
    s_str = f"{s:.3f}"
    l_str = f"{l:.3f}"
    
    if p == max_psnr: p_str = f"\\textbf{{{p_str}}}"
    if s == max_ssim: s_str = f"\\textbf{{{s_str}}}"
    if l == min_lpips: l_str = f"\\textbf{{{l_str}}}"
    
    row += f" & {p_str} & {s_str} & {l_str}"

print(row + " \\\\")
print(r"\bottomrule")

