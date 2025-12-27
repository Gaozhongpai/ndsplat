import json
import os
from pathlib import Path

def extract_metrics(base_path):
    """Extract ours_best metrics from all scenes in a dataset."""
    base_path = Path(base_path)

    # Get all scene directories
    scenes = [d for d in base_path.iterdir() if d.is_dir()]

    results = []
    output_lines = []

    # Header
    output_lines.append("=" * 80)
    output_lines.append(f"Dataset: {base_path.name}")
    output_lines.append("=" * 80)
    output_lines.append("")

    for scene_dir in sorted(scenes):
        results_file = scene_dir / "results.json"

        if not results_file.exists():
            continue

        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            if "ours_best" not in data:
                continue

            best_data = data["ours_best"]

            # Add scene name
            scene_results = {"Scene": scene_dir.name}
            scene_results.update(best_data)
            results.append(scene_results)

            # Format output
            output_lines.append(f"Scene: {scene_dir.name}")
            output_lines.append("-" * 80)
            output_lines.append(f"  Number:        {best_data.get('Number', 'N/A')}")
            output_lines.append(f"  Training_time: {best_data.get('Training_time', 'N/A')}")
            output_lines.append(f"  FPS:           {best_data.get('FPS', 'N/A')}")
            output_lines.append(f"  SSIM:          {best_data.get('SSIM', 'N/A')}")
            output_lines.append(f"  PSNR:          {best_data.get('PSNR', 'N/A')}")
            output_lines.append(f"  LPIPS:         {best_data.get('LPIPS', 'N/A')}")
            output_lines.append("")

        except Exception as e:
            print(f"Error processing {results_file}: {e}")
            continue

    # Calculate means
    if results:
        output_lines.append("=" * 80)
        output_lines.append("MEAN VALUES")
        output_lines.append("=" * 80)

        metrics = ["Number", "Training_time", "FPS", "SSIM", "PSNR", "LPIPS"]
        means = {}

        for metric in metrics:
            values = [r[metric] for r in results if metric in r and r[metric] is not None]
            if values:
                mean_val = sum(values) / len(values)
                means[metric] = mean_val
                output_lines.append(f"  {metric:15s}: {mean_val}")
            else:
                output_lines.append(f"  {metric:15s}: N/A")

        output_lines.append("")
        output_lines.append(f"Total scenes: {len(results)}")
        output_lines.append("=" * 80)
    else:
        output_lines.append("No valid results found.")
        output_lines.append("=" * 80)

    # Save to file
    output_file = base_path / "metrics_summary.txt"
    with open(output_file, 'w') as f:
        f.write("\n".join(output_lines))

    return results, means, str(output_file)

def process_all_methods_datasets(root_path):
    """Process all methods and datasets in the output directory."""
    root_path = Path(root_path)

    all_summaries = []

    # Get all method directories (3dgs, ndgs, opacity_only, opacity_pos)
    methods = [d for d in root_path.iterdir() if d.is_dir()]

    for method_dir in sorted(methods):
        method_name = method_dir.name
        print(f"\n{'=' * 80}")
        print(f"Processing Method: {method_name}")
        print(f"{'=' * 80}")

        # Get all dataset directories under this method
        datasets = [d for d in method_dir.iterdir() if d.is_dir()]

        for dataset_dir in sorted(datasets):
            dataset_name = dataset_dir.name
            print(f"\n  Processing Dataset: {dataset_name}")

            results, means, output_file = extract_metrics(dataset_dir)

            if results:
                print(f"    ✓ Saved summary to: {output_file}")
                print(f"    ✓ Processed {len(results)} scenes")

                # Store summary info
                summary_info = {
                    'method': method_name,
                    'dataset': dataset_name,
                    'num_scenes': len(results),
                    'means': means,
                    'output_file': output_file
                }
                all_summaries.append(summary_info)

                # Print quick stats
                if means:
                    print(f"    ✓ Mean PSNR: {means.get('PSNR', 0):.4f} | SSIM: {means.get('SSIM', 0):.6f} | LPIPS: {means.get('LPIPS', 0):.6f}")
            else:
                print(f"    ✗ No valid results found in {dataset_dir}")

    # Print overall summary
    print(f"\n\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nProcessed {len(all_summaries)} method-dataset combinations:\n")

    # Group by method
    methods_data = {}
    for summary in all_summaries:
        method = summary['method']
        if method not in methods_data:
            methods_data[method] = []
        methods_data[method].append(summary)

    for method, datasets in sorted(methods_data.items()):
        print(f"\n{method}:")
        for ds in datasets:
            print(f"  - {ds['dataset']:20s} ({ds['num_scenes']:2d} scenes) | PSNR: {ds['means'].get('PSNR', 0):7.4f} | SSIM: {ds['means'].get('SSIM', 0):.6f} | LPIPS: {ds['means'].get('LPIPS', 0):.6f}")

    print(f"\n{'=' * 80}")
    print(f"Total summaries created: {len(all_summaries)}")
    print(f"{'=' * 80}\n")

    return all_summaries

if __name__ == "__main__":
    output_root = "/code/workspace/6dgs-iclr/output"

    print(f"Starting metrics extraction from: {output_root}\n")
    summaries = process_all_methods_datasets(output_root)

    print("\n✓ All metrics extracted successfully!")
