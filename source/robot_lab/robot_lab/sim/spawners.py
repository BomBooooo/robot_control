# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Custom spawners for heterogeneous assets."""

from __future__ import annotations

import math
import random
import re
from dataclasses import MISSING

import carb
from pxr import Sdf, Usd

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils import configclass


def _compute_asset_indices(
    num_envs: int,
    ratios: list[float],
    seed: int | None,
    randomize: bool,
) -> list[int]:
    if num_envs <= 0:
        return []
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Asset ratios must sum to a positive value.")
    scaled = [num_envs * r / total for r in ratios]
    counts = [int(math.floor(x)) for x in scaled]
    remainder = num_envs - sum(counts)
    if remainder > 0:
        order = sorted(range(len(ratios)), key=lambda i: scaled[i] - counts[i], reverse=True)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    asset_indices: list[int] = []
    for idx, count in enumerate(counts):
        asset_indices.extend([idx] * count)
    if randomize and len(asset_indices) > 1:
        rng = random.Random(seed)
        rng.shuffle(asset_indices)
    return asset_indices


def spawn_multi_asset_fixed_ratio(
    prim_path: str,
    cfg: "FixedRatioMultiAssetSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple assets based on a fixed ratio across environments."""
    stage = sim_utils.get_current_stage()

    # resolve: {SPAWN_NS}/AssetName
    root_path, asset_path = prim_path.rsplit("/", 1)
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        if len(source_prim_paths) == 0:
            raise RuntimeError(f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning.")
    else:
        source_prim_paths = [root_path]

    template_prim_path = sim_utils.get_next_free_prim_path("/World/Template", stage=stage)
    sim_utils.create_prim(template_prim_path, "Scope", stage=stage)

    proto_prim_paths: list[str] = []
    for index, asset_cfg in enumerate(cfg.assets_cfg):
        if cfg.semantic_tags is not None:
            if asset_cfg.semantic_tags is None:
                asset_cfg.semantic_tags = cfg.semantic_tags
            else:
                asset_cfg.semantic_tags += cfg.semantic_tags
        for attr_name in ["mass_props", "rigid_props", "collision_props", "activate_contact_sensors", "deformable_props"]:
            attr_value = getattr(cfg, attr_name)
            if hasattr(asset_cfg, attr_name) and attr_value is not None:
                setattr(asset_cfg, attr_name, attr_value)
        proto_prim_path = f"{template_prim_path}/Asset_{index:04d}"
        asset_cfg.func(
            proto_prim_path,
            asset_cfg,
            translation=translation,
            orientation=orientation,
            clone_in_fabric=clone_in_fabric,
            replicate_physics=replicate_physics,
        )
        proto_prim_paths.append(proto_prim_path)

    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    num_envs = len(prim_paths)

    if cfg.asset_indices is not None:
        if len(cfg.asset_indices) != num_envs:
            raise ValueError(
                "Length of asset_indices does not match number of environments: "
                f"{len(cfg.asset_indices)} != {num_envs}."
            )
        asset_indices = list(cfg.asset_indices)
    else:
        ratios = cfg.asset_ratios or [1.0] * len(cfg.assets_cfg)
        if len(ratios) != len(cfg.assets_cfg):
            raise ValueError("asset_ratios length must match assets_cfg length.")
        asset_indices = _compute_asset_indices(num_envs, ratios, cfg.seed, cfg.randomize)
        cfg.asset_indices = list(asset_indices)

    with Sdf.ChangeBlock():
        for index, prim_path in enumerate(prim_paths):
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            proto_path = proto_prim_paths[asset_indices[index] % len(proto_prim_paths)]
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))

    sim_utils.delete_prim(template_prim_path, stage=stage)

    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/isaaclab/spawn/multi_assets", True)

    return stage.GetPrimAtPath(prim_paths[0])

@configclass
class FixedRatioMultiAssetSpawnerCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Spawn multiple assets with a fixed ratio and randomized assignment."""

    func = spawn_multi_asset_fixed_ratio

    assets_cfg: list[SpawnerCfg] = MISSING
    """List of asset configurations to spawn."""

    asset_ratios: list[float] | None = None
    """Relative ratios for each asset. Defaults to equal ratios."""

    randomize: bool = True
    """Whether to shuffle asset assignment across environments."""

    seed: int | None = None
    """Random seed for asset assignment. Defaults to None."""

    asset_indices: list[int] | None = None
    """Resolved asset index for each environment instance (written during spawning)."""
