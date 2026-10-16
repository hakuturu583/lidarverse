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

"""Dump nuScenes HDMap (static map features + dynamic boxes) to a LidarVerse NPZ
that the nuScenes NCore dataloader can rasterize to a range image on the fly.

Reads the RAW nuScenes dataset (devkit + map expansion; NOT the NCore store) and
writes, per scene, ``<out_root>/<scene_name>.npz``. The scene_name == the NCore
stem (e.g. ``scene-0001``), so the NCore lidar store and this NPZ key off the
same id.

KEY DIFFERENCE vs the Waymo dumper: NCore's re-derived pose graph DRIFTS from
nuScenes' ``ego_pose`` trajectory (measured ~20 m by mid-scene), so map/box
points CANNOT be placed into the NCore world frame with one rigid transform.
Instead we keep everything in the nuScenes GLOBAL (map) frame and ALSO store the
per-sweep nuScenes ``lidar_to_global`` poses; the composer projects the map into
the sensor via THESE poses (nuScenes map + ego_pose are self-consistent -> the
map lands on the actual returns), fully decoupled from the NCore trajectory.

Schema (all in nuScenes GLOBAL frame unless noted):
  Static map, pre-sampled to dense points:
    map_points : (M, 3) float32  -- xyz (z=0, nuScenes maps are 2D)
    map_type   : (M,)   int8     -- 0 lane / 1 road_line / 2 road_edge / 3 crosswalk
  Dynamic boxes, INTERPOLATED to every sweep (flat layout):
    box_frame  : (K,)      int32 -- sweep index this box belongs to (0..num_frames-1)
    box_o2w    : (K, 4, 4) float32-- object_to_global
    box_lwh    : (K, 3)    float32-- length(x) width(y) height(z)  [nuScenes wlh -> lwh]
    box_type   : (K,)      int8   -- 0 vehicle / 1 pedestrian / 2 cyclist
    box_moving : (K,)      bool
    box_track_id : (K,)    int64  -- stable per-scene instance index
  Per-sweep pose (the projection frame the composer uses):
    lidar_to_global : (num_frames, 4, 4) float32
    frame_timestamps_us : (num_frames,) int64  -- for a load-time alignment guard
  Misc:
    num_frames          : int
    vehicle_to_world_f0 : (4,4) -- frame-0 lidar_to_global (debug)

NEVER writes anywhere except ``<out_root>`` (default
``<lidarverse>/data/nuscenes_hdmap``, resolved from this script's repo root);
the raw nuScenes dataset is opened read-only.

Run from the LidarVerse repo (needs nuscenes-devkit):
  python scripts/data/dump_nuscenes_hdmap.py \
      --nusc_root data/nuscenes/raw --scene scene-0001
  # -> data/nuscenes_hdmap/scene-0001.npz
  python scripts/data/dump_nuscenes_hdmap.py \
      --nusc_root data/nuscenes/raw --num_workers 8
"""

import argparse
import os
import sys
import types

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# nuscenes-devkit imports cv2 for optional camera/video rendering, but this
# metadata/map dumper never calls those paths. The project container carries
# conflicting OpenCV binary/Python packages, so provide the unused module as a
# stub rather than mutating the environment or failing during devkit import.
sys.modules["cv2"] = types.ModuleType("cv2")

# Always under LidarVerse data/ (independent of shell cwd).
_DEFAULT_HDMAP_OUT = os.path.join(_REPO_ROOT, "data", "nuscenes_hdmap")

_MAP_TYPE_ID = {"lane": 0, "road_line": 1, "road_edge": 2, "crosswalk": 3}

# nuScenes category prefix -> box_type. Everything else is dropped.
_PED_PREFIX = "human.pedestrian"
_CYCLIST = {"vehicle.bicycle", "vehicle.motorcycle"}
_VEHICLE_PREFIX = "vehicle."
_MIN_MOVING_SPEED = 0.2  # m/s (matches the Waymo dumper)
_REQUIRED_OUTPUT_KEYS = {
    "map_points",
    "map_type",
    "box_frame",
    "box_o2w",
    "box_lwh",
    "box_type",
    "box_moving",
    "box_track_id",
    "lidar_to_global",
    "frame_timestamps_us",
    "num_frames",
}


def _valid_output(path):
    """Return whether a completed, readable scene NPZ is present."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if not _REQUIRED_OUTPUT_KEYS.issubset(data.files):
                return False
            num_frames = int(data["num_frames"])
            return (
                num_frames > 0
                and data["lidar_to_global"].shape == (num_frames, 4, 4)
                and data["frame_timestamps_us"].shape == (num_frames,)
                and data["box_track_id"].shape == data["box_frame"].shape
            )
    except Exception:
        return False


def _category_to_type(name: str):
    if name.startswith(_PED_PREFIX):
        return 1
    if name in _CYCLIST:
        return 2
    if name.startswith(_VEHICLE_PREFIX):
        return 0
    return None


def _resample_polyline(verts: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    """Densify a polyline/polygon (V,>=2) to ~``spacing`` m spacing -> (P,3) z=0."""
    verts = np.asarray(verts, dtype=np.float64)
    if verts.shape[0] < 2:
        return np.zeros((0, 3), np.float32)
    xy = verts[:, :2]
    if closed:
        xy = np.concatenate([xy, xy[:1]], axis=0)
    out = []
    for a, b in zip(xy[:-1], xy[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        n = max(int(np.ceil(L / spacing)), 1)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append(a[None, :] + ts * seg[None, :])
    out.append(xy[-1:][None, 0])
    pts2d = np.concatenate(out, axis=0)
    return np.concatenate([pts2d, np.zeros((pts2d.shape[0], 1))], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- #
# Static map extraction (nuScenes map API)
# ----------------------------------------------------------------------------- #
def _extract_map(nusc_map, patch, spacing, centerline_res):
    """Return (map_points (M,3), map_type (M,)) in the nuScenes global frame.

    patch = (x_min, y_min, x_max, y_max). Classes:
      lane(0)      = lane + lane_connector centerlines
      road_line(1) = lane_divider + road_divider lines
      road_edge(2) = road_segment polygon exteriors
      crosswalk(3) = ped_crossing polygons
    """
    from nuscenes.map_expansion.arcline_path_utils import discretize_lane

    pts_list, type_list = [], []

    def _add(points3d, tid):
        if points3d.shape[0] > 0:
            pts_list.append(points3d)
            type_list.append(np.full(points3d.shape[0], tid, dtype=np.int8))

    recs = nusc_map.get_records_in_patch(
        patch,
        ["lane", "lane_connector", "lane_divider", "road_divider",
         "road_segment", "ped_crossing"],
        mode="intersect",
    )

    # lane / lane_connector centerlines
    for layer in ("lane", "lane_connector"):
        for tok in recs.get(layer, []):
            try:
                arc = nusc_map.arcline_path_3.get(tok)
                if not arc:
                    continue
                disc = np.asarray(discretize_lane(arc, centerline_res), dtype=np.float64)
                if disc.shape[0] < 2:
                    continue
                _add(_resample_polyline(disc[:, :3], spacing, closed=False), 0)
            except Exception:
                continue

    # lane_divider / road_divider lines
    for layer in ("lane_divider", "road_divider"):
        for tok in recs.get(layer, []):
            try:
                rec = nusc_map.get(layer, tok)
                line = nusc_map.extract_line(rec["line_token"])
                if line.is_empty:
                    continue
                xy = np.asarray(line.xy).T  # (V,2)
                _add(_resample_polyline(xy, spacing, closed=False), 1)
            except Exception:
                continue

    # road_segment polygon exteriors -> road_edge
    for tok in recs.get("road_segment", []):
        try:
            rec = nusc_map.get("road_segment", tok)
            poly = nusc_map.extract_polygon(rec["polygon_token"])
            if poly.is_empty:
                continue
            xy = np.asarray(poly.exterior.xy).T
            _add(_resample_polyline(xy, spacing, closed=True), 2)
        except Exception:
            continue

    # ped_crossing polygons -> crosswalk
    for tok in recs.get("ped_crossing", []):
        try:
            rec = nusc_map.get("ped_crossing", tok)
            poly = nusc_map.extract_polygon(rec["polygon_token"])
            if poly.is_empty:
                continue
            xy = np.asarray(poly.exterior.xy).T
            _add(_resample_polyline(xy, spacing, closed=True), 3)
        except Exception:
            continue

    if not pts_list:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int8)
    return np.concatenate(pts_list, 0), np.concatenate(type_list, 0)


# ----------------------------------------------------------------------------- #
# Frame chain + boxes
# ----------------------------------------------------------------------------- #
def _frame_chain(nusc, scene):
    """Ordered LIDAR_TOP sample_data for the whole scene (all sweeps).

    Returns (timestamps (F,), ego_from_lidar (4,4), lidar_to_global (F,4,4))."""
    from pyquaternion import Quaternion

    samp = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", samp["data"]["LIDAR_TOP"])
    while sd["prev"] != "":
        sd = nusc.get("sample_data", sd["prev"])

    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego_from_lidar = np.eye(4)
    ego_from_lidar[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    ego_from_lidar[:3, 3] = cs["translation"]

    ts, l2g = [], []
    while True:
        ts.append(sd["timestamp"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        g = np.eye(4)
        g[:3, :3] = Quaternion(ego["rotation"]).rotation_matrix
        g[:3, 3] = ego["translation"]
        l2g.append(g @ ego_from_lidar)
        if sd["next"] == "":
            break
        sd = nusc.get("sample_data", sd["next"])
    return (np.asarray(ts, np.int64),
            ego_from_lidar.astype(np.float32),
            np.stack(l2g, 0).astype(np.float64))


def _instance_tracks(nusc, scene):
    """Per-instance keyframe tracks: token -> dict(t, trans, quat, lwh, type)."""
    from pyquaternion import Quaternion

    tracks = {}
    stok = scene["first_sample_token"]
    while stok != "":
        samp = nusc.get("sample", stok)
        t = samp["timestamp"]
        for ann_tok in samp["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            btype = _category_to_type(ann["category_name"])
            if btype is None:
                continue
            inst = ann["instance_token"]
            w, l, h = ann["size"]  # nuScenes size = [w, l, h]
            tr = tracks.setdefault(
                inst, {"t": [], "trans": [], "quat": [], "lwh": [l, w, h], "type": btype}
            )
            tr["t"].append(t)
            tr["trans"].append(np.asarray(ann["translation"], np.float64))
            tr["quat"].append(Quaternion(ann["rotation"]))
        stok = samp["next"]
    return tracks


def _interp_boxes(tracks, frame_ts):
    """Interpolate each instance track to every sweep timestamp in frame_ts.

    Returns flat box arrays plus a stable per-scene integer track ID."""
    from pyquaternion import Quaternion

    bf, bo2w, blwh, btype, bmov, btrack = [], [], [], [], [], []
    frame_ts = np.asarray(frame_ts, np.int64)
    for track_id, (_, tr) in enumerate(sorted(tracks.items())):
        tk = np.asarray(tr["t"], np.int64)
        if tk.shape[0] == 0:
            continue
        order = np.argsort(tk)
        tk = tk[order]
        trans = [tr["trans"][i] for i in order]
        quat = [tr["quat"][i] for i in order]
        lwh = np.asarray(tr["lwh"], np.float32)
        t_lo, t_hi = tk[0], tk[-1]
        # sweeps covered by this track
        fsel = np.nonzero((frame_ts >= t_lo) & (frame_ts <= t_hi))[0]
        for fi in fsel:
            ft = frame_ts[fi]
            j = int(np.searchsorted(tk, ft))
            if j == 0:
                pos, q, speed = trans[0], quat[0], 0.0
            elif j >= tk.shape[0]:
                pos, q, speed = trans[-1], quat[-1], 0.0
            else:
                t0, t1 = tk[j - 1], tk[j]
                a = 0.0 if t1 == t0 else float((ft - t0) / (t1 - t0))
                pos = (1 - a) * trans[j - 1] + a * trans[j]
                q = Quaternion.slerp(quat[j - 1], quat[j], a)
                dt = (t1 - t0) / 1e6
                speed = (float(np.linalg.norm(trans[j] - trans[j - 1])) / dt
                         if dt > 0 else 0.0)
            o2w = np.eye(4)
            o2w[:3, :3] = q.rotation_matrix
            o2w[:3, 3] = pos
            bf.append(int(fi))
            bo2w.append(o2w.astype(np.float32))
            blwh.append(lwh)
            btype.append(tr["type"])
            bmov.append(bool(speed > _MIN_MOVING_SPEED))
            btrack.append(track_id)
    if not bf:
        return (np.zeros((0,), np.int32), np.zeros((0, 4, 4), np.float32),
                np.zeros((0, 3), np.float32), np.zeros((0,), np.int8),
                np.zeros((0,), bool), np.zeros((0,), np.int64))
    return (np.asarray(bf, np.int32), np.stack(bo2w, 0), np.stack(blwh, 0),
            np.asarray(btype, np.int8), np.asarray(bmov, bool),
            np.asarray(btrack, np.int64))


# ----------------------------------------------------------------------------- #
# Per-scene dump
# ----------------------------------------------------------------------------- #
def _dump_scene(nusc, maps, scene, out_root, spacing, centerline_res,
                map_margin, overwrite):
    name = scene["name"]
    out_path = os.path.join(out_root, f"{name}.npz")
    if os.path.isfile(out_path) and not overwrite:
        if _valid_output(out_path):
            print(f"[{name}] exists, skip")
            return
        print(f"[{name}] existing output is invalid; rebuilding")

    ts, ego_from_lidar, l2g = _frame_chain(nusc, scene)
    num_frames = ts.shape[0]

    location = nusc.get("log", scene["log_token"])["location"]
    nusc_map = maps[location]

    # crop map to the trajectory bbox + margin
    xy = l2g[:, :2, 3]
    x_min, y_min = xy.min(0) - map_margin
    x_max, y_max = xy.max(0) + map_margin
    map_points, map_type = _extract_map(
        nusc_map, (x_min, y_min, x_max, y_max), spacing, centerline_res
    )

    tracks = _instance_tracks(nusc, scene)
    (
        box_frame,
        box_o2w,
        box_lwh,
        box_type,
        box_moving,
        box_track_id,
    ) = _interp_boxes(tracks, ts)

    os.makedirs(out_root, exist_ok=True)
    tmp_path = f"{out_path}.tmp{os.getpid()}.npz"
    np.savez_compressed(
        tmp_path,
        map_points=map_points,
        map_type=map_type,
        box_frame=box_frame,
        box_o2w=box_o2w,
        box_lwh=box_lwh,
        box_type=box_type,
        box_moving=box_moving,
        box_track_id=box_track_id,
        lidar_to_global=l2g.astype(np.float32),
        frame_timestamps_us=ts,
        num_frames=np.int32(num_frames),
        vehicle_to_world_f0=l2g[0].astype(np.float32),
    )
    os.replace(tmp_path, out_path)
    n_mov = int(box_moving.sum())
    print(f"[{name}] loc={location} frames={num_frames} map_pts={map_points.shape[0]} "
          f"boxes={box_frame.shape[0]} moving={n_mov} -> {out_path}")


# Per-process NuScenes handle (parallel workers load devkit once each).
_MP = {}


def _init_worker(nusc_root, version):
    from nuscenes.nuscenes import NuScenes

    _MP["nusc_root"] = nusc_root
    _MP["nusc"] = NuScenes(version=version, dataroot=nusc_root, verbose=False)
    _MP["maps"] = {}
    _MP["scene_by_name"] = {s["name"]: s for s in _MP["nusc"].scene}


def _dump_scene_worker(name, out_root, spacing, centerline_res, map_margin, overwrite):
    scene = _MP["scene_by_name"].get(name)
    if scene is None:
        raise KeyError(f"unknown scene {name!r}")
    location = _MP["nusc"].get("log", scene["log_token"])["location"]
    if location not in _MP["maps"]:
        from nuscenes.map_expansion.map_api import NuScenesMap

        _MP["maps"][location] = NuScenesMap(
            dataroot=_MP["nusc_root"], map_name=location
        )
    _dump_scene(
        _MP["nusc"], _MP["maps"], scene, out_root,
        spacing, centerline_res, map_margin, overwrite,
    )


def _resolve_scene_names(args, filter_list):
    """Scene stems to dump from --scene / --filter_list (no devkit load)."""
    if args.scene:
        names = list(args.scene)
    elif filter_list and os.path.isfile(filter_list):
        with open(filter_list) as f:
            names = [ln.strip() for ln in f if ln.strip()]
    else:
        return None
    if args.limit >= 0:
        names = names[: args.limit]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nusc_root",
        required=True,
        help="raw nuScenes root containing maps, samples, and metadata",
    )
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out_root", default=_DEFAULT_HDMAP_OUT,
                    help=f"output dir (default: {_DEFAULT_HDMAP_OUT})")
    ap.add_argument("--scene", action="append", default=None,
                    help="scene name (NCore stem). Repeatable.")
    ap.add_argument("--filter_list", default=None,
                    help="optional .lst of scene names")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--spacing", type=float, default=0.15, help="map point spacing (m)")
    ap.add_argument("--centerline_res", type=float, default=0.5,
                    help="lane centerline discretization (m)")
    ap.add_argument("--map_margin", type=float, default=120.0,
                    help="crop map to trajectory bbox + this margin (m)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="parallel processes (each loads devkit once)")
    ap.add_argument(
        "--completion_marker",
        default=None,
        help="write this marker only after every requested scene has a valid NPZ",
    )
    args = ap.parse_args()

    if not os.path.isabs(args.out_root):
        args.out_root = os.path.join(_REPO_ROOT, args.out_root)
    os.makedirs(args.out_root, exist_ok=True)

    filter_list = args.filter_list
    if filter_list and not os.path.isabs(filter_list) and not os.path.isfile(filter_list):
        alt = os.path.join(_REPO_ROOT, filter_list)
        if os.path.isfile(alt):
            filter_list = alt

    scene_names = _resolve_scene_names(args, filter_list)
    if scene_names is None:
        from nuscenes.nuscenes import NuScenes

        print(f"loading {args.version} (slow) ...")
        nusc_enum = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
        scene_names = [s["name"] for s in nusc_enum.scene]
        if args.limit >= 0:
            scene_names = scene_names[: args.limit]
        del nusc_enum
    if not scene_names:
        raise SystemExit("No scenes matched (check --scene / --filter_list / --version).")
    print(f"dumping {len(scene_names)} scene(s) -> {args.out_root} "
          f"(workers={args.num_workers})")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    dump_kw = dict(
        out_root=args.out_root,
        spacing=args.spacing,
        centerline_res=args.centerline_res,
        map_margin=args.map_margin,
        overwrite=args.overwrite,
    )

    failures = []
    if args.num_workers <= 1:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.map_expansion.map_api import NuScenesMap

        print(f"loading {args.version} (slow) ...")
        nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
        scene_by_name = {s["name"]: s for s in nusc.scene}
        maps = {}
        for name in scene_names:
            scene = scene_by_name.get(name)
            if scene is None:
                print(f"[{name}] FAILED: unknown scene")
                failures.append(name)
                continue
            loc = nusc.get("log", scene["log_token"])["location"]
            if loc not in maps:
                print(f"  loading map: {loc}")
                maps[loc] = NuScenesMap(dataroot=args.nusc_root, map_name=loc)
        for name in tqdm(scene_names, desc="dump", unit="scene"):
            scene = scene_by_name.get(name)
            if scene is None:
                continue
            try:
                _dump_scene(
                    nusc, maps, scene, args.out_root, args.spacing,
                    args.centerline_res, args.map_margin, args.overwrite,
                )
            except Exception as e:
                print(f"[{name}] FAILED: {e}")
                failures.append(name)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(args.nusc_root, args.version),
        ) as ex:
            futs = {
                ex.submit(_dump_scene_worker, name, **dump_kw): name
                for name in scene_names
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="dump", unit="scene"):
                name = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[{name}] FAILED: {e}")
                    failures.append(name)

    invalid = [
        name for name in scene_names
        if not _valid_output(os.path.join(args.out_root, f"{name}.npz"))
    ]
    failures = sorted(set(failures + invalid))
    if failures:
        preview = ", ".join(failures[:10])
        raise SystemExit(
            f"{len(failures)} scene(s) failed or have invalid outputs: {preview}"
        )

    if args.completion_marker:
        marker = args.completion_marker
        if not os.path.isabs(marker):
            marker = os.path.join(_REPO_ROOT, marker)
        marker_dir = os.path.dirname(marker)
        if marker_dir:
            os.makedirs(marker_dir, exist_ok=True)
        marker_tmp = f"{marker}.tmp{os.getpid()}"
        with open(marker_tmp, "w") as handle:
            handle.write(f"scenes={len(scene_names)}\n")
        os.replace(marker_tmp, marker)
        print(f"completion marker -> {marker}")
    print(f"DONE: {len(scene_names)} valid scene NPZ(s).")


if __name__ == "__main__":
    main()
