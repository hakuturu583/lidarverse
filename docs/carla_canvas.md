<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LidarVerse in CARLA (carla_canvas)

[carla_canvas](https://github.com/hakuturu583/carla_canvas) swaps how CARLA
renders cameras and LiDARs behind the CARLA API: a typesafe_carla program stays
as it is and a YAML file named by `CARLA_CANVAS_CONFIG` picks the renderer.
`lidarverse_canvas.LidarverseRenderer` is such a renderer: CARLA's
`sensor.lidar.ray_cast` measurements come from LidarVerse's HDMap-conditioned
world generation instead of the server's ray casts.

## Running

1. Install and download the checkpoints ([README](../README.md#setup)):
   `uv sync`, `python scripts/download_checkpoints.py`.
2. Start the renderer service in this repository's environment, on the GPU
   machine:

   ```bash
   uv run carla-canvas serve --config examples/carla_canvas/lidarverse.yaml
   ```

3. Run the CARLA program (in its own environment, with `carla-canvas[client]`)
   with the same file, or a copy whose `renderer.endpoint` points at the service:

   ```bash
   CARLA_CANVAS_CONFIG=examples/carla_canvas/lidarverse.yaml typesafe-codon run main.py
   ```

`LoadWorld` reads the map, loads the model and runs one throwaway block so the
attention kernels are compiled before the first tick; give it time with
`renderer.startup_timeout` (the example allows 600 s). ROS 2 output
(`enable_for_ros()`) works as for any carla_canvas renderer.

## What it does

| step | |
|---|---|
| map | The OpenDRIVE CARLA sends with `LoadWorld` is read by [roadgen](https://pypi.org/project/roadgen/) and written as the vector layers the models were conditioned on: lanes, lane lines, road boundaries, crosswalks, wait lines (polylines), traffic lights and signs (box wireframes), in OpenDRIVE's frame, which is carla_canvas's `right_handed` frame (CARLA's with y negated). `options.hdmap: {rdshq, clip}` uses an RDS-HQ clip instead. |
| actors | Bounding boxes of actors matching `options.actors` (default vehicles and pedestrians), not the LiDAR's own vehicle. |
| sweeps | The model runs at 10 Hz: a 360-degree sweep is due every 0.1 s of simulation time. Frame 0 is generated alone, then blocks of four, each from its HDMap range condition (map points and box edges rasterized with the LiDAR's poses at the sweeps' start and end, rolling shutter, `scripts/diffusion/infer_pai.py --raw-hdmap-condition-rasterize`) and the caches of the previous blocks: the rolling self-attention K/V cache with its sink frame, the tokenizer caches and the spatial cache of generated geometry (`diffusion/inference/causal.py`, made open-ended in `lidarverse_canvas/stream.py`). |
| lookahead | A block is generated when its first sweep is due, so its later sweeps' poses (and the boxes) are extrapolated from the last ticks at constant velocity. |
| measurements | Each CARLA measurement reads its rays from the sweep covering its time, at the nearest cell of the 128x1024 grid. carla_canvas applies CARLA's LiDAR model around it: partial sweeps per tick, `dropoff_general_rate`, noise, intensity `exp(-atmosphere_attenuation_rate * d)`. |

Several LiDARs each get their own stream (caches) on the shared model.

## Limits

* The released packages are range-only NVIDIA PAI models: a Hesai Pandar128
  (128 lasers from +15 to -25 degrees) with returns from 2.5 to 100 m. CARLA
  rays outside that field of view or range return nothing, whatever the
  blueprint's `channels`, `upper_fov`, `lower_fov` or `range`.
* `sensor.lidar.ray_cast_semantic` and cameras are not rendered; give them to
  another backend or leave them to CARLA.
* What is generated is conditioned on the map's vector layers and the boxes, not
  on CARLA's meshes: buildings, vegetation and props are the model's own.
* A time jump of more than `options.reset_after_seconds` (default 1 s), or
  backwards, restarts a sensor's stream; so does `load_world`.

## Options

| option | default | |
|---|---|---|
| `checkpoints` | `<repo>/checkpoints` | download_checkpoints.py's directory, relative to the configuration file |
| `stage` | 3 | causal package: 3 (SF-DMD) or 2 |
| `num_steps` | 4 (stage 3), 35 (stage 2) | SDE steps per block |
| `sde_eta` | 1.0 | |
| `seed` | 0 | block `i` uses `seed + i` |
| `kv_cache_frames` | 9 (stage 3), 21 (stage 2) | self-attention cache, 1 + 4N frames including the sink |
| `actors` | `["vehicle.*", "walker.pedestrian.*"]` | blueprint globs of the boxes in the condition |
| `bbox_dilation` | 3 | box edges widened to k x k range pixels |
| `segment_spacing` | 0.1 | meters between rasterized map / box points |
| `hdmap` | none | `{rdshq: <root>, clip: <id>}`: an RDS-HQ clip's 3d_* layers instead of the OpenDRIVE |
| `map_offset` | `[0, 0, 0]` | added to the vector map (right-handed frame) |
| `warmup` | true | compile kernels during `LoadWorld` |
| `device` | `cuda` | where the HDMap condition is rasterized |
