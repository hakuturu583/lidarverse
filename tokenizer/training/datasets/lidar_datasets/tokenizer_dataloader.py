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


import omegaconf
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from utils.lazy_config import LazyCall as L
from tokenizer.training.datasets.lidar_datasets.configs import COMMONDATA_CONFIG
from tokenizer.training.datasets.lidar_datasets.rangemap_sampler import LidarRangeMapSampler
import numpy as np

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, ignore_keys=None):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """

    batched = {key: [] for key in samples[0]}
    # assert isinstance(samples[0][first_key], (list, tuple)), type(samples[first_key])

    for s in samples:
        [batched[key].append(s[key]) for key in batched if not (ignore_keys is not None and key in ignore_keys)]

    result = {}
    for key in batched:
        if ignore_keys and key in ignore_keys:
            continue
        try:
            if isinstance(batched[key][0], bool):
                assert key in {"is_preprocessed", "loss_mask_is_validity"}
                assert all(value == batched[key][0] for value in batched[key])
                result[key] = batched[key][0]  # this is a hack to align with cosmos data
            elif isinstance(batched[key][0], (int, float)):
                if combine_scalars:
                    result[key] = torch.from_numpy(np.array(list(batched[key])))
            elif isinstance(batched[key][0], torch.Tensor):
                if combine_tensors:
                    result[key] = torch.stack(list(batched[key]))
            elif isinstance(batched[key][0], np.ndarray):
                if combine_tensors:
                    result[key] = np.array(list(batched[key]))
            elif isinstance(batched[key][0], list) and isinstance(batched[key][0][0], int):
                result[key] = [torch.Tensor(elems).long() for elems in zip(*batched[key])]
            else:
                result[key] = list(batched[key])
        except Exception as e:
            print(key)
            raise e
        # result.append(b)
    del batched
    return result

def get_lidar_range_map_dataloader(
    dataset_name: str,
    shuffle=True,
    num_workers=4,
    prefetch_factor=4,
    **kwargs,
) -> omegaconf.dictconfig.DictConfig:
    return L(MyDataLoader)(
        dataset=L(LidarRangeMapDataset)(
            dataset_name=dataset_name,
            **kwargs,
        ),
        batch_size=1,
        num_workers=num_workers,
        shuffle=shuffle,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )


def get_cross_dataset_lidar_range_map_dataloader(
    dataset_names,
    validation=False,
    mix_datasets_across_ranks=False,
    dataset_sampling_weights=None,
    seed=0,
    num_workers=4,
    prefetch_factor=4,
    **kwargs,
) -> omegaconf.dictconfig.DictConfig:
    """Build opt-in native-grid homogeneous batches from multiple sensors."""
    return L(HomogeneousMultiDataLoader)(
        dataset=L(CrossDatasetRangeMapDataset)(
            dataset_names=dataset_names,
            shared_overrides=kwargs,
            uses_sensor_metadata=True,
        ),
        batch_size=1,
        validation=validation,
        mix_datasets_across_ranks=mix_datasets_across_ranks,
        dataset_sampling_weights=dataset_sampling_weights,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: int = 1, *args, **kw):
        dataset_obj = dataset.build_dataset()
        print("dl", kw.get("dataloaders"))
        if "dataloaders" in kw:
            kw.pop("dataloaders")  # HACK?
        print("dl2", kw.get("dataloaders"))
        super().__init__(dataset_obj, batch_size, collate_fn=dict_collation_fn, *args, **kw)


class CrossDatasetRangeMapDataset:
    def __init__(
        self,
        dataset_names,
        shared_overrides=None,
        per_dataset_overrides=None,
        uses_sensor_metadata=True,
    ):
        self.dataset_names = list(dataset_names)
        self.shared_overrides = dict(shared_overrides or {})
        self.per_dataset_overrides = dict(per_dataset_overrides or {})
        self.uses_sensor_metadata = bool(uses_sensor_metadata)

    def build_dataset(self):
        children = []
        for dataset_id, dataset_name in enumerate(self.dataset_names):
            dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
            overrides = OmegaConf.create(
                {
                    **self.shared_overrides,
                    **self.per_dataset_overrides.get(dataset_name, {}),
                    "emit_sensor_metadata": True,
                    "sensor_dataset_id": dataset_id,
                    "sensor_dataset_name": dataset_name,
                }
            )
            config = OmegaConf.to_container(
                OmegaConf.merge(dataset_config, overrides), resolve=True
            )
            children.append(LidarRangeMapSamplerTokenizer(**config))
        return HomogeneousMultiDataset(children)


class HomogeneousMultiDataset(Dataset):
    """Route a batch of tuple indices to exactly one native-grid child."""

    def __init__(self, children):
        self.children = list(children)
        if not self.children:
            raise ValueError("At least one child dataset is required")

    def __len__(self):
        return sum(len(child) for child in self.children)

    def __getitem__(self, index):
        dataset_id, local_index = index
        return self.children[dataset_id][local_index]


class HomogeneousDatasetBatchSampler(Sampler):
    """Keep local batches homogeneous, with optional rank-balanced mixing."""

    def __init__(
        self,
        dataset,
        batch_size,
        validation=False,
        mix_datasets_across_ranks=False,
        dataset_sampling_weights=None,
        seed=0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.validation = bool(validation)
        self.mix_datasets_across_ranks = bool(mix_datasets_across_ranks)
        self.dataset_sampling_weights = None
        if dataset_sampling_weights is not None:
            weights = torch.as_tensor(
                dataset_sampling_weights, dtype=torch.float64
            )
            if weights.numel() != len(dataset.children):
                raise ValueError(
                    "dataset_sampling_weights must have one value per dataset; "
                    f"got {weights.numel()} values for {len(dataset.children)} datasets"
                )
            if (
                not torch.isfinite(weights).all()
                or (weights < 0).any()
                or weights.sum() <= 0
            ):
                raise ValueError(
                    "dataset_sampling_weights must be finite, non-negative, and sum positive"
                )
            if self.mix_datasets_across_ranks and not torch.allclose(
                weights, weights.new_full(weights.shape, weights.mean())
            ):
                raise ValueError(
                    "Non-uniform dataset sampling is incompatible with "
                    "mix_datasets_across_ranks=True"
                )
            self.dataset_sampling_weights = weights / weights.sum()
        self.seed = int(seed)
        self.num_batches = (
            sum(len(child) for child in dataset.children) // self.batch_size
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        cursors = [0] * len(self.dataset.children)
        for step in range(self.num_batches):
            if self.validation:
                dataset_id = step % len(self.dataset.children)
            elif self.mix_datasets_across_ranks:
                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )
                dataset_id = (rank + step) % len(self.dataset.children)
            else:
                if self.dataset_sampling_weights is None:
                    dataset_id = int(
                        torch.randint(
                            len(self.dataset.children), (1,), generator=generator
                        ).item()
                    )
                else:
                    dataset_id = int(
                        torch.multinomial(
                            self.dataset_sampling_weights,
                            1,
                            generator=generator,
                        ).item()
                    )
            child_len = len(self.dataset.children[dataset_id])
            start = cursors[dataset_id]
            batch = [
                (dataset_id, (start + offset) % child_len)
                for offset in range(self.batch_size)
            ]
            cursors[dataset_id] = (start + self.batch_size) % child_len
            yield batch


class HomogeneousMultiDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        validation=False,
        mix_datasets_across_ranks=False,
        dataset_sampling_weights=None,
        seed=0,
        *args,
        **kwargs,
    ):
        dataset_obj = dataset.build_dataset()
        batch_sampler = HomogeneousDatasetBatchSampler(
            dataset_obj,
            batch_size=batch_size,
            validation=validation,
            mix_datasets_across_ranks=mix_datasets_across_ranks,
            dataset_sampling_weights=dataset_sampling_weights,
            seed=seed,
        )
        super().__init__(
            dataset_obj,
            batch_sampler=batch_sampler,
            collate_fn=dict_collation_fn,
            *args,
            **kwargs,
        )


class LidarRangeMapDataset:
    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
        self.extra_config = OmegaConf.create(kwargs)

    def build_dataset(self):
        # Merge extra_config (from kwargs/overrides) into dataset_config
        if self.extra_config:
            merged_config = OmegaConf.merge(self.dataset_config, self.extra_config)
        else:
            merged_config = self.dataset_config
        config_dict = OmegaConf.to_container(merged_config, resolve=True)
        return LidarRangeMapSamplerTokenizer(**config_dict)


class LidarRangeMapSamplerTokenizer(LidarRangeMapSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if self.load_rgb_only:
            raise NotImplementedError
        if self.load_lidar_or_rgb:
            raise NotImplementedError
        if self.load_lidar_and_rgb:
            raise NotImplementedError

        # sample the video index
        lidar_data_idx = (idx % self.n_data_per_node) + self.data_start_idx  # frame idx
        lidar_frame_indices = self.sample_frame_indices(load_lidar=True)

        # load the data
        example = self.dataset._read_rgb_and_lidar(lidar_data_idx, lidar_frame_indices, ["custom"])

        lidar_rangemap, recon_target, lidar_valid_mask = self.get_lidar(example)

        if self.is_video_tokenizer:
            key = "video"
        else:
            key = "images"

        sample = {}
        sample[key] = lidar_rangemap
        sample["recon_target"] = recon_target
        if lidar_valid_mask is not None:
            # Explicit per-pixel return mask (pre per-row) -> MASK_KEY +
            # mask_as_input concat. See LidarRangeMapSampler.process_lidar.
            sample["loss_mask"] = lidar_valid_mask
            sample["loss_mask_is_validity"] = True
        sample["__key__"] = example["__key__"]
        clip_name = "%s-%s-%d" % (self.data_name, example["__key__"], idx)
        sample["clip_name"] = clip_name
        sample["data_type"] = "lidar"
        sample.update(self.get_sensor_metadata(*lidar_rangemap.shape[-2:]))
        # Ego poses (lidar->world, sweep-start), aligned to the video temporal
        # layout: mirror the repeat_temporal + frame-0 prepend applied to the
        # range maps in `_apply_repeat_crop_temporal`, so pose[t] indexes the
        # same frame as video[:, t]. Shape: (T, 4, 4).
        lidar_to_world = np.asarray(example["lidar_to_world"], dtype=np.float32)  # (N, 4, 4)
        if self.is_video_tokenizer and self.frame_sampling_method in (
            "sequential",
            "sequential_from_zero",
        ):
            lidar_to_world = np.concatenate(
                [lidar_to_world[:1], np.repeat(lidar_to_world, self.repeat_temporal, axis=0)],
                axis=0,
            )
        sample["lidar_to_world"] = torch.from_numpy(lidar_to_world).float()

        del example

        return sample   


if __name__ == "__main__":
    pass
