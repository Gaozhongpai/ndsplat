import json
import os
from pathlib import Path
from collections import defaultdict

def parse_metrics_file(file_path):
    """Parse a metrics_summary.txt file and extract scene metrics and means."""
    scenes = {}
    means = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_scene = None
    reading_means = False

    for line in lines:
        line = line.strip()

        if line.startswith("Scene:"):
            current_scene = line.split("Scene:")[1].strip()
            scenes[current_scene] = {}
        elif line.startswith("MEAN VALUES"):
            reading_means = True
            current_scene = None
        elif current_scene and ":" in line:
            # Parse metric for current scene
            parts = line.split(":")
            if len(parts) == 2:
                metric = parts[0].strip()
                value = parts[1].strip()
                try:
                    scenes[current_scene][metric] = float(value)
                except:
                    pass
        elif reading_means and ":" in line and not line.startswith("Total"):
            # Parse mean value
            parts = line.split(":")
            if len(parts) == 2:
                metric = parts[0].strip()
                value = parts[1].strip()
                try:
                    means[metric] = float(value)
                except:
                    pass

    return scenes, means

def load_all_metrics(output_root):
    """Load all metrics from all method-dataset combinations."""
    data = defaultdict(lambda: defaultdict(dict))

    methods = ['ndgs', 'opacity_only', 'opacity_pos', 'opacity_pos_decouple', '3dgs']
    datasets = ['nerf_synthetic', '7dgs_pbr', 'dnerf', 'medical_pbr', 'tandt_pbr']

    for method in methods:
        for dataset in datasets:
            metrics_file = Path(output_root) / method / dataset / "metrics_summary.txt"
            if metrics_file.exists():
                scenes, means = parse_metrics_file(metrics_file)
                data[dataset][method] = {
                    'scenes': scenes,
                    'means': means
                }

    return data

def format_metric(value, metric_type, is_best=False):
    """Format a metric value for LaTeX, with bold formatting if best."""
    if metric_type == 'PSNR':
        formatted = f"{value:.2f}"
    elif metric_type in ['SSIM']:
        formatted = f"{value:.3f}"
    elif metric_type == 'LPIPS':
        formatted = f"{value:.3f}"
    elif metric_type == 'Number':
        formatted = f"{int(value)}"
    elif metric_type == 'Training_time':
        formatted = f"{value:.2f}"
    elif metric_type == 'FPS':
        formatted = f"{value:.2f}"
    else:
        formatted = f"{value:.2f}"

    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted

def find_best_value(values, metric_type):
    """Find the best value for a metric (higher for PSNR/SSIM/FPS, lower for LPIPS/Time)."""
    if not values:
        return None

    if metric_type in ['LPIPS', 'Training_time']:
        return min(values)
    else:  # PSNR, SSIM, FPS, Number
        return max(values)

def generate_quality_table(dataset_name, dataset_data, method_order):
    """Generate a quality comparison table (PSNR, SSIM, LPIPS)."""
    # Map method names to display names
    method_display = {
        'ndgs': 'N-DGS',
        'opacity_only': 'Opacity-Only',
        'opacity_pos': 'Opacity+Pos',
        'opacity_pos_decouple': 'Opacity+Pos (Decoupled)',
        '3dgs': '3DGS'
    }

    # Get all scene names (from first available method)
    scene_names = None
    for method in method_order:
        if method in dataset_data and 'scenes' in dataset_data[method]:
            scene_names = sorted(dataset_data[method]['scenes'].keys())
            break

    if not scene_names:
        return None

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Quantitative comparison on {dataset_name.replace('_', ' ').title()} dataset at 30K iterations. Best results in \\textbf{{bold}}.}}")
    lines.append(f"\\label{{tab:{dataset_name}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")

    # Header
    num_methods = len([m for m in method_order if m in dataset_data])
    col_spec = "l|" + "ccc|" * (num_methods - 1) + "ccc"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Method headers
    header_parts = [""]
    methods_in_data = [m for m in method_order if m in dataset_data]
    for idx, method in enumerate(methods_in_data):
        col_sep = '|' if idx < len(methods_in_data) - 1 else ''
        header_parts.append(f"\\multicolumn{{3}}{{c{col_sep}}}{{\\textbf{{{method_display[method]}}}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Metric headers
    metric_header = ["Scene"]
    for method in method_order:
        if method in dataset_data:
            metric_header.extend(["PSNR $\\uparrow$", "SSIM $\\uparrow$", "LPIPS $\\downarrow$"])
    lines.append(" & ".join(metric_header) + " \\\\")
    lines.append("\\midrule")

    # Scene rows
    for scene in scene_names:
        # Escape underscores in scene names for LaTeX
        scene_escaped = scene.replace('_', '\\_')
        row = [f"\\texttt{{{scene_escaped}}}"]

        # For each method, add PSNR, SSIM, LPIPS
        for method in method_order:
            if method in dataset_data:
                for metric in ['PSNR', 'SSIM', 'LPIPS']:
                    # Collect values from all methods for this metric to find best
                    values = []
                    for m in method_order:
                        if m in dataset_data and scene in dataset_data[m]['scenes']:
                            v = dataset_data[m]['scenes'][scene].get(metric)
                            if v is not None:
                                values.append(v)

                    best_val = find_best_value(values, metric)

                    # Add current method's value
                    if scene in dataset_data[method]['scenes']:
                        val = dataset_data[method]['scenes'][scene].get(metric)
                        if val is not None:
                            is_best = (val == best_val) if best_val is not None else False
                            row.append(format_metric(val, metric, is_best))
                        else:
                            row.append("N/A")
                    else:
                        row.append("N/A")

        lines.append(" & ".join(row) + " \\\\")

    # Mean row
    lines.append("\\midrule")
    row = ["\\textbf{\\texttt{Mean}}"]

    # For each method, add PSNR, SSIM, LPIPS
    for method in method_order:
        if method in dataset_data:
            for metric in ['PSNR', 'SSIM', 'LPIPS']:
                # Collect values from all methods for this metric to find best
                values = []
                for m in method_order:
                    if m in dataset_data and 'means' in dataset_data[m]:
                        v = dataset_data[m]['means'].get(metric)
                        if v is not None:
                            values.append(v)

                best_val = find_best_value(values, metric)

                # Add current method's value
                if 'means' in dataset_data[method]:
                    val = dataset_data[method]['means'].get(metric)
                    if val is not None:
                        is_best = (val == best_val) if best_val is not None else False
                        row.append(format_metric(val, metric, is_best))
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")

    lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)

def generate_efficiency_table(dataset_name, dataset_data, method_order):
    """Generate an efficiency comparison table (Number, Training_time, FPS)."""
    # Map method names to display names
    method_display = {
        'ndgs': 'N-DGS',
        'opacity_only': 'Opacity-Only',
        'opacity_pos': 'Opacity+Pos',
        'opacity_pos_decouple': 'Opacity+Pos (Decoupled)',
        '3dgs': '3DGS'
    }

    # Get all scene names
    scene_names = None
    for method in method_order:
        if method in dataset_data and 'scenes' in dataset_data[method]:
            scene_names = sorted(dataset_data[method]['scenes'].keys())
            break

    if not scene_names:
        return None

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Efficiency comparison on {dataset_name.replace('_', ' ').title()} dataset.}}")
    lines.append(f"\\label{{tab:{dataset_name}_efficiency}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")

    # Header
    num_methods = len([m for m in method_order if m in dataset_data])
    col_spec = "l|" + "rrr|" * (num_methods - 1) + "rrr"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Method headers
    header_parts = [""]
    methods_in_data = [m for m in method_order if m in dataset_data]
    for idx, method in enumerate(methods_in_data):
        col_sep = '|' if idx < len(methods_in_data) - 1 else ''
        header_parts.append(f"\\multicolumn{{3}}{{c{col_sep}}}{{\\textbf{{{method_display[method]}}}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Metric headers
    metric_header = ["Scene"]
    for method in method_order:
        if method in dataset_data:
            metric_header.extend(["$\\#$Gauss (K)", "Time (min)", "FPS"])
    lines.append(" & ".join(metric_header) + " \\\\")
    lines.append("\\midrule")

    # Scene rows
    for scene in scene_names:
        # Escape underscores in scene names for LaTeX
        scene_escaped = scene.replace('_', '\\_')
        row = [f"\\texttt{{{scene_escaped}}}"]

        # For each method, add Number, Training_time, FPS
        for method in method_order:
            if method in dataset_data:
                for metric in ['Number', 'Training_time', 'FPS']:
                    # Collect values from all methods for this metric to find best
                    values = []
                    for m in method_order:
                        if m in dataset_data and scene in dataset_data[m]['scenes']:
                            v = dataset_data[m]['scenes'][scene].get(metric)
                            if v is not None:
                                values.append(v)

                    best_val = find_best_value(values, metric)

                    # Add current method's value
                    if scene in dataset_data[method]['scenes']:
                        val = dataset_data[method]['scenes'][scene].get(metric)
                        if val is not None:
                            is_best = (val == best_val) if best_val is not None else False
                            # Convert Number to K (thousands)
                            if metric == 'Number':
                                val_display = val / 1000.0
                                formatted = format_metric(val_display, metric, is_best)
                            elif metric == 'Training_time':
                                # Convert seconds to minutes
                                val_display = val / 60.0
                                formatted = format_metric(val_display, metric, is_best)
                            else:
                                formatted = format_metric(val, metric, is_best)
                            row.append(formatted)
                        else:
                            row.append("N/A")
                    else:
                        row.append("N/A")

        lines.append(" & ".join(row) + " \\\\")

    # Mean row
    lines.append("\\midrule")
    row = ["\\textbf{\\texttt{Mean}}"]

    # For each method, add Number, Training_time, FPS
    for method in method_order:
        if method in dataset_data:
            for metric in ['Number', 'Training_time', 'FPS']:
                # Collect values from all methods for this metric to find best
                values = []
                for m in method_order:
                    if m in dataset_data and 'means' in dataset_data[m]:
                        v = dataset_data[m]['means'].get(metric)
                        if v is not None:
                            values.append(v)

                best_val = find_best_value(values, metric)

                # Add current method's value
                if 'means' in dataset_data[method]:
                    val = dataset_data[method]['means'].get(metric)
                    if val is not None:
                        is_best = (val == best_val) if best_val is not None else False
                        # Convert Number to K (thousands)
                        if metric == 'Number':
                            val_display = val / 1000.0
                            formatted = format_metric(val_display, metric, is_best)
                        elif metric == 'Training_time':
                            # Convert seconds to minutes
                            val_display = val / 60.0
                            formatted = format_metric(val_display, metric, is_best)
                        else:
                            formatted = format_metric(val, metric, is_best)
                        row.append(formatted)
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")

    lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)

if __name__ == "__main__":
    output_root = "/code/workspace/6dgs-iclr/output"

    # Load all metrics
    print("Loading metrics from all datasets...")
    all_data = load_all_metrics(output_root)

    # Define method order
    method_order = ['ndgs', 'opacity_only', 'opacity_pos', 'opacity_pos_decouple', '3dgs']

    # Generate tables for each dataset
    output_file = "/code/workspace/generated_latex_tables.tex"

    with open(output_file, 'w') as f:
        for dataset in sorted(all_data.keys()):
            print(f"\nGenerating tables for {dataset}...")

            # Quality table
            quality_table = generate_quality_table(dataset, all_data[dataset], method_order)
            if quality_table:
                f.write(f"\n% {dataset.upper()} - Quality Metrics\n")
                f.write(quality_table)
                f.write("\n\n")

            # Efficiency table
            efficiency_table = generate_efficiency_table(dataset, all_data[dataset], method_order)
            if efficiency_table:
                f.write(f"\n% {dataset.upper()} - Efficiency Metrics\n")
                f.write(efficiency_table)
                f.write("\n\n")

    print(f"\n✓ LaTeX tables generated successfully!")
    print(f"✓ Saved to: {output_file}")
