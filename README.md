# Running `pointcloud_metrics.py` with uv

`pointcloud_metrics.py` compares two point clouds (Chamfer Distance, Hausdorff, Precision/Recall/F-score,
RMSE, and - with `--advanced` - PSNR D2, PSNR YUV, PointSSIM, PCQM, MS-GraphSIM). This doc covers how to
set up the environment with [uv](https://docs.astral.sh/uv/) and shows it running end-to-end against a
small generated sample point cloud pair.

## 1. Install uv

```bash
brew install uv        # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Sync the environment

The repo root has a `pyproject.toml` pinned to the versions in `requirements.txt` (numpy, scipy, open3d).
`open3d==0.19.0` only ships wheels for Python 3.11/3.12, so the project pins `requires-python = ">=3.11,<3.13"`
- uv will download a matching interpreter automatically, you don't need one installed system-wide.

```bash
uv sync
```

This creates `.venv/` with everything `pointcloud_metrics.py` needs.

## 3. Generate sample data

There's no real scan data checked into the repo, so `sample_data/generate_sample_data.py` synthesizes a
small pair of colored point clouds to exercise the script against:

- `sample_data/reference.ply` — 20,000 points sampled on a unit sphere, colored by position, normals
  estimated by Open3D.
- `sample_data/distorted.ply` — the same sphere with Gaussian noise added to the geometry (σ=0.01) and
  colors (σ=0.02), simulating a reconstruction that's close but not identical to the reference.

```bash
uv run python sample_data/generate_sample_data.py
```

```
Wrote sample_data/reference.ply (20000 points)
Wrote sample_data/distorted.ply (20000 points)
```

## 4. Run the metrics

Basic metrics:

```bash
uv run python pointcloud_metrics.py sample_data/reference.ply sample_data/distorted.ply --threshold 0.05
```

```
[BASIC GEOMETRIC METRICS]
Chamfer Distance (CD):
  Overall CD:        0.012411
  PC1 → PC2:         0.011869
  PC2 → PC1:         0.012952

Hausdorff Distance:  0.042495
RMSE:                0.017354

Precision/Recall Metrics:
  Precision:         1.0000 (100.00%)
  Recall:            1.0000 (100.00%)
  F-score:           1.0000
```

Full run with advanced metrics, normals, and a JSON dump:

```bash
uv run python pointcloud_metrics.py sample_data/reference.ply sample_data/distorted.ply \
  --threshold 0.05 --advanced --normals --output sample_data/metrics_output.json
```

```
[ADVANCED QUALITY METRICS]

PSNR MSE D2 (Point-to-Plane):
  PSNR D2:           104.9564 dB

PSNR MSE YUV (Point-to-Attribute):
  PSNR Y:            37.4004 dB
  PSNR Cb:           38.0222 dB
  PSNR Cr:           37.4453 dB
  PSNR Color:        37.4838 dB

PointSSIM (Structural Similarity):
  Geometry Features: 0.310559
  Luminance Features:0.766555
  (Lower is better, 0 = identical)

PCQM (Point Cloud Quality Metric):
  PCQM Score:        0.455303
  (Higher is better, range [0, 1])

MS-GraphSIM (Multi-Scale Graph Similarity):
  MS-GraphSIM:       0.000030
  (Higher is better, range [0, 1])
```

`sample_data/metrics_output.json` then contains all metrics as plain JSON:

```json
{
  "chamfer_distance": 0.01241067045547799,
  "chamfer_pc1_to_pc2": 0.011869486885894492,
  "chamfer_pc2_to_pc1": 0.01295185402506149,
  "hausdorff_distance": 0.042494766609338,
  "precision": 1.0,
  "recall": 1.0,
  "fscore": 1.0,
  "rmse": 0.017353945657899918,
  "psnr_d2": 104.9564334790141,
  "psnr_y": 37.400447845458984,
  "psnr_cb": 38.02219772338867,
  "psnr_cr": 37.445289611816406,
  "psnr_color": 37.483768463134766,
  "pointssim_geometry": 0.3105591735134726,
  "pointssim_luminance": 0.7665554035964598,
  "pcqm": 0.45530303844996034,
  "ms_graphsim": 3.0332887739348426e-05
}
```

`--threshold 0.05` is loose on purpose given the unit-sphere scale of the sample data (radius 1, noise
σ=0.01) — tune it to your own point clouds' scale and expected error. Add `--visualize` to open both
clouds side by side in the Open3D viewer, or `--max-points` / `--voxel-size` to downsample large scans
before computing metrics.

## Shared nearest-neighbor searches

`compute_all_metrics` shares two KD-trees and the exact whole-cloud `k=1` query
results across metrics. Metric formulas, neighborhood queries, and output keys
are unchanged. Individual metric functions still work without a shared cache.
The cache lasts only for one comparison and assumes its point arrays are unchanged.
Caching is enabled by default. Add `--no-cache` to disable sharing trees and query
results between metrics; each metric then creates its own search. For Python
callers, use `compute_all_metrics(..., use_cache=False)`. Metric formulas and
output keys are unchanged in either mode.

Shared search work is charged to the first metric that needs it (Chamfer Distance);
later per-metric timings exclude that reused work. Timing keys remain unchanged.
The timing table is hidden by default; add `--timings` to display it. Metrics are
computed and saved in the same way with or without this flag.

## Note

Running with `--output` uncovered a bug where PSNR/YUV values (`numpy.float32`) failed `json.dump`
with `TypeError: Object of type float32 is not JSON serializable`. Fixed in `pointcloud_metrics.py` by
casting every metric to a native Python `float` before serialization.


```
@misc{zielinski2026comparativeevaluationgeometricaccuracy,
      title={A Comparative Evaluation of Geometric Accuracy in NeRF and Gaussian Splatting}, 
      author={Mikolaj Zielinski and Eryk Vykysaly and Bartlomiej Biesiada and Jan Baturo and Mateusz Capala and Dominik Belter},
      year={2026},
      eprint={2604.18205},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.18205}, 
}
```
