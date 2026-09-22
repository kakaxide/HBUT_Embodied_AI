# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

import os
import pathlib

import torch
from omegaconf import DictConfig


def _load_sft_state_dict(model, model_state_dict):
    """Merge PEFT LoRA deltas into base weights, then load with strict=False.

    RLinf SFT with ``is_lora: True`` saves the full PEFT wrapper state dict
    (``*.base_layer.weight/bias`` + ``*.lora_A/B.default.weight``) whose keys
    also carry the extra ``.base_model.model.model.`` segment introduced by the
    HF PeftModel wrapper. A plain model built with ``is_lora: False`` expects
    bare ``<module>.proj.weight`` keys, so we (1) normalize the wrapper prefix,
    (2) fold each LoRA delta into its base weight, and (3) drop any key that
    still cannot be mapped onto a real model parameter.
    """
    sd = model_state_dict
    if isinstance(sd, dict):
        for wrapper_key in ("model", "state_dict", "module"):
            if wrapper_key in sd and isinstance(sd[wrapper_key], dict):
                sd = sd[wrapper_key]
                break

    expected = set(model.state_dict().keys())

    def _norm(key: str) -> str:
        # Most params sit under HF PeftModel's ``.base_model.model.model.``
        # expansion; a few (lm_head) only under ``.base_model.model.``. Replace
        # the wrapper segment with a single "." so name parts stay separated.
        return key.replace(".base_model.model.model.", ".model.").replace(
            ".base_model.model.", "."
        )

    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        nk = _norm(key)
        if nk.endswith(".lora_A.default.weight"):
            lora_a[nk[: -len(".lora_A.default.weight")]] = tensor
        elif nk.endswith(".lora_B.default.weight"):
            lora_b[nk[: -len(".lora_B.default.weight")]] = tensor

    merged: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        nk = _norm(key)
        if nk.endswith((".lora_A.default.weight", ".lora_B.default.weight")):
            continue
        if ".base_layer." in nk:
            head, _, tail = nk.rpartition(".base_layer.")
            target_key = f"{head}.{tail}"  # ...k_proj.weight / ...k_proj.bias
            if target_key not in expected:
                continue
            w = tensor
            if tail == "weight" and head in lora_a and head in lora_b:
                # PEFT stores A as [r, in] and B as [out, r]; delta = B @ A.
                delta = torch.matmul(
                    lora_b[head].float(), lora_a[head].float()
                ).to(tensor.dtype)
                w = tensor + delta
            merged[target_key] = w.contiguous()
            continue
        if nk in expected:
            merged[nk] = tensor

    result = model.load_state_dict(merged, strict=False)
    missing = result.missing_keys
    unexpected = result.unexpected_keys
    n_lora_pairs = len(set(lora_a) & set(lora_b))
    print(
        f"[openpi load] mapped {len(merged)} tensors "
        f"(lora pairs merged: {n_lora_pairs}); "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[openpi load] missing sample: {missing[:5]}", flush=True)
    if unexpected:
        print(f"[openpi load] unexpected sample: {unexpected[:5]}", flush=True)
    return result


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(
            full_weights_path, map_location="cpu", weights_only=False, mmap=True
        )
        _load_sft_state_dict(model, model_state_dict)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(
            actor_full_weights_path, map_location="cpu", weights_only=False, mmap=True
        )
        _load_sft_state_dict(model, model_state_dict)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        # Exclude lerobot policy_*_processor/*normalizer_processor artifacts, which
        # carry normalization stats, not model weights.
        weight_paths = [
            p
            for p in weight_paths
            if not any(s in os.path.basename(p) for s in ("processor", "normalizer"))
        ]
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            all_state_dict.update(state_dict)
        # Some checkpoints (e.g. lerobot-format pi05) wrap every key under a
        # leading "model." namespace; strip it so the bare params match.
        normalized_sd = {}
        for key, tensor in all_state_dict.items():
            nk = key[6:] if key.startswith("model.") else key
            normalized_sd[nk] = tensor
        res = model.load_state_dict(normalized_sd, strict=False)
        missing = res.missing_keys
        unexpected = res.unexpected_keys
        print(
            f"[openpi load] safetensors mapped {len(normalized_sd)} tensors; "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        if missing:
            print(f"[openpi load] missing sample: {missing[:5]}", flush=True)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
