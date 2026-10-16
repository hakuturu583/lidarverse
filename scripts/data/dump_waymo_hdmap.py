# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dump Waymo HDMap (static map features + dynamic boxes) to a LidarVerse
NPZ that the NCore dataloader can rasterize to a range image on the fly.

Reads the RAW Waymo Open Dataset ``.tfrecord`` (NOT the NCore store) and writes,
per clip, ``<out_root>/<clip_id>.npz`` with:

  Static map (Waymo WORLD frame, pose-independent), pre-sampled to dense points:
    map_points : (M, 3) float32   -- xyz in world frame
    map_type   : (M,)   int8      -- 0 lane / 1 road_line / 2 road_edge / 3 crosswalk

  Dynamic boxes (per native 10 Hz frame; flat layout):
    box_frame    : (K,)      int32  -- frame index this box belongs to
    box_o2w      : (K, 4, 4) float32-- object_to_world (Waymo world frame)
    box_lwh      : (K, 3)    float32-- length(x) width(y) height(z)
    box_type     : (K,)      int8   -- 0 Car / 1 Pedestrian / 2 Cyclist
    box_moving   : (K,)      bool
    box_track_id : (K,)      int32  -- per-clip integer track id (stable across
                                      frames; from WOD label.id). -1 if missing.
    box_velocity : (K, 2)    float32-- WORLD-frame (speed_x, speed_y) m/s, i.e.
                                      the raw WOD metadata speed OpenPCDet rotates
                                      into the ego frame for the vel head target.

  Misc:
    num_frames           : int    -- raw frame count (== NCore lidar.frames_count)
    vehicle_to_world_f0  : (4,4)   -- frame-0 ego pose (debug / frame-alignment)

``clip_id`` == NCore stem == the bit between ``segment-`` and
``_with_camera_labels`` in the tfrecord name, so the NCore lidar store and this
NPZ key off the same id.

NEVER writes anywhere except ``<out_root>`` (default ``data/waymo_hdmap``); the
raw Waymo dataset is opened read-only.

Run inside the training container (needs tensorflow + waymo_open_dataset):
  python scripts/data/dump_waymo_hdmap.py \
      --waymo_root data/waymo/raw/training \
      --clip 10017090168044687777_6380_000_6400_000 \
      --out_root data/waymo_hdmap
"""

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Map-feature name -> type id (kept; speed_bump/driveway dropped, as in cosmos).
_MAP_TYPE_ID = {"lane": 0, "road_line": 1, "road_edge": 2, "crosswalk": 3}
_MAP_POLYLINE = {"lane", "road_line", "road_edge"}
_MAP_POLYGON = {"crosswalk"}


def _hump_to_underline(s: str) -> str:
    import re

    return re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()


def _resample_polyline(verts: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    """Densify a polyline/polygon (V,3) to ~``spacing`` m point spacing."""
    if verts.shape[0] < 2:
        return verts.reshape(-1, 3)
    if closed:
        verts = np.concatenate([verts, verts[:1]], axis=0)
    out = []
    for a, b in zip(verts[:-1], verts[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        n = max(int(np.ceil(L / spacing)), 1)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append(a[None, :] + ts * seg[None, :])
    out.append(verts[-1:][None, 0])
    return np.concatenate(out, axis=0).astype(np.float32)


def _dump_clip(waymo_root: str, clip_id: str, out_root: str, spacing: float, overwrite: bool = False):
    out_path = os.path.join(out_root, f"{clip_id}.npz")
    if os.path.isfile(out_path) and not overwrite:
        print(f"[{clip_id}] exists, skip ({out_path})")
        return
    import tensorflow as tf
    from google.protobuf import json_format
    from waymo_open_dataset import dataset_pb2, label_pb2

    def _yaw_matrix(heading: float) -> np.ndarray:
        c, s = np.cos(heading), np.sin(heading)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    tfr = os.path.join(waymo_root, f"segment-{clip_id}_with_camera_labels.tfrecord")
    if not os.path.isfile(tfr):
        raise FileNotFoundError(f"tfrecord not found: {tfr}")

    proto2type = {
        label_pb2.Label.Type.TYPE_VEHICLE: 0,  # Car
        label_pb2.Label.Type.TYPE_PEDESTRIAN: 1,
        label_pb2.Label.Type.TYPE_CYCLIST: 2,
    }
    min_moving_speed = 0.2

    dataset = tf.data.TFRecordDataset(tfr, compression_type="")

    map_points_list, map_type_list = [], []
    box_frame, box_o2w, box_lwh, box_type, box_moving = [], [], [], [], []
    box_track_id, box_velocity = [], []
    track_id_map: dict = {}  # WOD label.id (str) -> per-clip int, stable over frames
    num_frames = 0
    vehicle_to_world_f0 = np.eye(4, dtype=np.float32)

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        vehicle_to_world = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        if frame_idx == 0:
            vehicle_to_world_f0 = vehicle_to_world.astype(np.float32)
            # Static map lives only on the first frame's map_features (world frame).
            for feat in json_format.MessageToDict(frame).get("mapFeatures", []):
                keys = [k for k in feat.keys() if k != "id"]
                if not keys:
                    continue
                name = _hump_to_underline(keys[0])
                if name not in _MAP_TYPE_ID:
                    continue
                node = feat[keys[0]]
                if name in _MAP_POLYLINE:
                    raw = node.get("polyline", [])
                    closed = False
                elif name in _MAP_POLYGON:
                    raw = node.get("polygon", [])
                    closed = True
                else:
                    continue
                verts = np.array(
                    [[p["x"], p["y"], p.get("z", 0.0)] for p in raw], dtype=np.float32
                )
                if verts.shape[0] < 2:
                    continue
                pts = _resample_polyline(verts, spacing, closed)
                map_points_list.append(pts)
                map_type_list.append(
                    np.full(pts.shape[0], _MAP_TYPE_ID[name], dtype=np.int8)
                )

        for label in frame.laser_labels:
            if label.type not in proto2type:
                continue
            box = label.box
            if not box.ByteSize():
                continue
            center = np.array([box.center_x, box.center_y, box.center_z], dtype=np.float64)
            b2v = np.eye(4)
            b2v[:3, :3] = _yaw_matrix(box.heading)
            b2v[:3, 3] = center
            o2w = vehicle_to_world @ b2v
            vx, vy = float(label.metadata.speed_x), float(label.metadata.speed_y)
            speed = np.sqrt(vx ** 2 + vy ** 2 + label.metadata.speed_z ** 2)
            tid = track_id_map.setdefault(label.id, len(track_id_map)) if label.id else -1
            box_frame.append(frame_idx)
            box_o2w.append(o2w.astype(np.float32))
            box_lwh.append(
                np.array([box.length, box.width, box.height], dtype=np.float32)
            )
            box_type.append(proto2type[label.type])
            box_moving.append(bool(speed > min_moving_speed))
            box_track_id.append(int(tid))
            box_velocity.append(np.array([vx, vy], dtype=np.float32))

        num_frames = frame_idx + 1

    map_points = (
        np.concatenate(map_points_list, axis=0)
        if map_points_list
        else np.zeros((0, 3), np.float32)
    )
    map_type = (
        np.concatenate(map_type_list, axis=0)
        if map_type_list
        else np.zeros((0,), np.int8)
    )

    os.makedirs(out_root, exist_ok=True)
    np.savez_compressed(
        out_path,
        map_points=map_points,
        map_type=map_type,
        box_frame=np.array(box_frame, dtype=np.int32),
        box_o2w=(
            np.stack(box_o2w, 0) if box_o2w else np.zeros((0, 4, 4), np.float32)
        ),
        box_lwh=(np.stack(box_lwh, 0) if box_lwh else np.zeros((0, 3), np.float32)),
        box_type=np.array(box_type, dtype=np.int8),
        box_moving=np.array(box_moving, dtype=bool),
        box_track_id=np.array(box_track_id, dtype=np.int32),
        box_velocity=(
            np.stack(box_velocity, 0) if box_velocity else np.zeros((0, 2), np.float32)
        ),
        num_frames=np.int32(num_frames),
        vehicle_to_world_f0=vehicle_to_world_f0,
    )
    n_moving = int(np.sum(box_moving)) if box_moving else 0
    moving_frac = (n_moving / max(len(box_frame), 1)) * 100.0
    print(
        f"[{clip_id}] frames={num_frames}  map_pts={map_points.shape[0]}  "
        f"boxes={len(box_frame)}  moving={n_moving} ({moving_frac:.0f}%)  -> {out_path}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--waymo_root",
        required=True,
        help="directory containing raw Waymo segment TFRecords",
    )
    ap.add_argument(
        "--clip",
        action="append",
        default=None,
        help="clip id (NCore stem). Repeatable. If omitted with --filter_list, "
        "uses the list; else dumps nothing.",
    )
    ap.add_argument("--filter_list", default=None, help="optional .lst of clip ids")
    ap.add_argument("--limit", type=int, default=-1, help="cap clips from filter_list (-1 = all)")
    ap.add_argument("--out_root", default="data/waymo_hdmap")
    ap.add_argument("--spacing", type=float, default=0.08, help="map point spacing (m)")
    ap.add_argument("--num_workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--overwrite", action="store_true", help="re-dump existing NPZs")
    args = ap.parse_args()

    clips = list(args.clip) if args.clip else []
    if not clips and args.filter_list and os.path.isfile(args.filter_list):
        with open(args.filter_list) as f:
            clips = [l.strip() for l in f if l.strip()]
    if not clips:
        # Fallback: every tfrecord in --waymo_root (clip id = NCore stem).
        import glob
        import re

        pat = os.path.join(args.waymo_root, "segment-*_with_camera_labels.tfrecord")
        clips = sorted(
            re.sub(r"^segment-(.*)_with_camera_labels\.tfrecord$", r"\1", os.path.basename(p))
            for p in glob.glob(pat)
        )
    if args.limit >= 0:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(
            "No clips. Pass --clip <id>, --filter_list, or point --waymo_root at a dir of tfrecords."
        )

    if not os.path.isabs(args.out_root):
        args.out_root = os.path.join(_REPO_ROOT, args.out_root)
    os.makedirs(args.out_root, exist_ok=True)
    print(f"Dumping {len(clips)} clip(s) -> {args.out_root}  (workers={args.num_workers})")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    if args.num_workers <= 1:
        for cid in tqdm(clips, desc="dump", unit="clip"):
            try:
                _dump_clip(args.waymo_root, cid, args.out_root, args.spacing, args.overwrite)
            except Exception as e:
                print(f"[{cid}] FAILED: {e}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {
                ex.submit(
                    _dump_clip, args.waymo_root, cid, args.out_root, args.spacing, args.overwrite
                ): cid
                for cid in clips
            }
            for fut in tqdm(
                as_completed(futs), total=len(futs), desc="dump", unit="clip"
            ):
                cid = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[{cid}] FAILED: {e}")
    print("DONE.")


if __name__ == "__main__":
    main()
