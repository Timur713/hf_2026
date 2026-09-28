"""Модель-эмбеддер и формат артефакта (контракт C3 в docs/contracts.md).

Артефакт — каталог с weights.safetensors и model_config.yaml. Трек B загружает любую модель
через load(model_dir, device) и получает forward: [B, 3, H, W] → [B, D], L2-нормированный.
"""
from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

# arch из model_config.yaml → имя модели в timm и тег публичных весов.
# Третий элемент: принимает ли модель произвольный размер входа (у свёрточных сетей его не задают).
REGISTRY = {
    "dinov2_vits14": ("vit_small_patch14_dinov2", "lvd142m", True),
    "dinov2_vitb14": ("vit_base_patch14_dinov2", "lvd142m", True),
    "dinov2_vitl14": ("vit_large_patch14_dinov2", "lvd142m", True),
    "dinov2reg_vitb14": ("vit_base_patch14_reg4_dinov2", "lvd142m", True),
    "clip_vitb16": ("vit_base_patch16_clip_224", "laion2b", True),
    "siglip2_vitb16": ("vit_base_patch16_siglip_224", "v2_webli", True),
    "eva02_vitb14": ("eva02_base_patch14_224", "mim_in22k", True),
    "convnext_base": ("convnext_base", "fb_in22k", False),
}
POOLS = ("cls", "avg", "cls_avg", "cls_gem", "cls_max", "cls_top16",
         "gem", "max", "top16")


def build_backbone(arch: str, input_size, pretrained: bool = False) -> nn.Module:
    name, tag, takes_img_size = REGISTRY[arch]
    kwargs = {"img_size": tuple(input_size)} if takes_img_size else {}
    return timm.create_model(f"{name}.{tag}", pretrained=pretrained, num_classes=0, **kwargs)


def pool_tokens(tokens: torch.Tensor, n_prefix: int, pool: str) -> torch.Tensor:
    """Токены [B, T, D] → вектор. cls_avg — склейка нормированных CLS и среднего по патчам.

    У свёрточных сетей префиксных токенов нет (n_prefix = 0), поэтому им доступен только avg.

    Варианты с пиками (cls_gem, cls_max, cls_top16) отвечают на измеренную проблему: различать два
    одинаковых по модели автомобиля можно только по мелким деталям — наклейке, вмятине, диску. Такая
    деталь занимает два патча из четырёхсот, и среднее ослабляет её в двести раз. GeM и максимум
    оставляют вес сильным откликам (A-049 показал, что на уровне формы экземпляры неразличимы).

    Важно про `top16`: топ берётся **по каждому каналу отдельно** (`topk` по оси токенов), то есть
    это сглаженный максимум по признаку, а не «средний вектор шестнадцати сильнейших патчей». Так
    оно и измерялось в A-051; отбор самих патчей — другая операция и отдельная гипотеза.
    """
    if n_prefix == 0 and pool.startswith("cls"):
        raise ValueError(f"pool={pool} требует CLS-токена; без него доступны avg, gem, max, top16")
    cls, patches = tokens[:, 0], tokens[:, n_prefix:]
    if pool == "cls":
        return cls
    if pool == "avg":
        return patches.mean(1)
    # Варианты без CLS — для SigLIP и свёрточных сетей, у которых префиксного токена нет.
    if pool == "gem":
        return patches.clamp(min=1e-6).pow(3.0).mean(1).pow(1.0 / 3.0)
    if pool == "max":
        return patches.max(1).values
    if pool == "top16":
        return patches.topk(min(16, patches.shape[1]), dim=1).values.mean(1)
    if pool == "cls_avg":
        second = patches.mean(1)
    elif pool == "cls_gem":
        # обобщённое среднее, p=3: между средним (p=1) и максимумом (p=∞)
        second = patches.clamp(min=1e-6).pow(3.0).mean(1).pow(1.0 / 3.0)
    elif pool == "cls_max":
        second = patches.max(1).values
    elif pool == "cls_top16":
        k = min(16, patches.shape[1])
        second = patches.topk(k, dim=1).values.mean(1)
    else:
        raise ValueError(f"unknown pool: {pool}")
    return torch.cat([F.normalize(cls, dim=1), F.normalize(second, dim=1)], dim=1)


class Embedder(nn.Module):
    """Backbone + пулинг + (опционально) BNNeck + L2-нормализация.

    Вход уже нормализован по mean/std (reid.data.preprocess). При обучении нужны оба выхода:
    features() — до BNNeck, для triplet; forward() — после BNNeck, это и есть эмбеддинг.
    """

    def __init__(self, backbone: nn.Module, pool: str = "cls", neck: str = "none",
                 parts: int = 0):
        """parts — число горизонтальных полос патч-сетки со своей веткой (схема MGN).

        Зачем. Все наши модели дают один вектор на кадр, а различать нужно экземпляры одной модели
        и цвета — то есть мелкие локальные детали. Глобальный пулинг их усредняет. Полосы
        заставляют кодировать области отдельно: у машины это крыша со стёклами, борт, колёсная
        зона. В ReID это стандартный приём и обычно он даёт больше, чем любая настройка рецепта.
        """
        super().__init__()
        assert pool in POOLS and neck in ("none", "bnneck")
        self.backbone, self.pool, self.neck_kind, self.parts = backbone, pool, neck, parts
        dim = backbone.num_features * (1 if not pool.startswith("cls_") else 2)
        self.global_dim = dim
        if neck == "bnneck":
            bn = nn.BatchNorm1d(dim)
            bn.bias.requires_grad_(False)  # BNNeck: сдвиг не учится (Bag-of-Tricks)
            self.neck = bn
            self.part_necks = nn.ModuleList()
            for _ in range(parts):
                pb = nn.BatchNorm1d(backbone.num_features)
                pb.bias.requires_grad_(False)
                self.part_necks.append(pb)
        else:
            self.neck = nn.Identity()
            self.part_necks = nn.ModuleList(nn.Identity() for _ in range(parts))

    def part_dims(self) -> list[int]:
        return [self.backbone.num_features] * self.parts

    def part_features(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Патчи режутся на self.parts горизонтальных полос, в каждой — top-k по каналу."""
        patches = tokens[:, self.n_prefix:]
        n = patches.shape[1]
        side = int(round(n ** 0.5))
        if side * side != n:                       # несквадратная сетка: режем по длине
            side = n
        grid = patches.view(patches.shape[0], side, -1, patches.shape[-1]) if side != n \
            else patches.unsqueeze(2)
        rows = grid.shape[1]
        out = []
        for i in range(self.parts):
            lo, hi = rows * i // self.parts, max(rows * (i + 1) // self.parts, rows * i // self.parts + 1)
            block = grid[:, lo:hi].reshape(patches.shape[0], -1, patches.shape[-1])
            k = max(1, block.shape[1] // 8)
            out.append(block.topk(k, dim=1).values.mean(1))
        return out

    @property
    def n_prefix(self) -> int:
        return getattr(self.backbone, "num_prefix_tokens", 0)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Всегда [B, T, D]: у свёрточных сетей карта признаков разворачивается в последовательность."""
        t = self.backbone.forward_features(x)
        return t.flatten(2).transpose(1, 2) if t.ndim == 4 else t

    def branches(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Глобальный вектор и векторы полос из одного прохода backbone."""
        tok = self.tokens(x)
        g = pool_tokens(tok, self.n_prefix, self.pool)
        return g, (self.part_features(tok) if self.parts else [])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        g, ps = self.branches(x)
        if not ps:
            return g
        return torch.cat([F.normalize(g, dim=1)] + [F.normalize(p, dim=1) for p in ps], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, ps = self.branches(x)
        if not ps:
            return F.normalize(self.neck(g), dim=1)
        vecs = [F.normalize(self.neck(g), dim=1)]
        vecs += [F.normalize(bn(p), dim=1) for bn, p in zip(self.part_necks, ps)]
        return F.normalize(torch.cat(vecs, dim=1), dim=1)


def save(model: Embedder, cfg: dict, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(model_dir, "weights.safetensors"))
    with open(os.path.join(model_dir, "model_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "model_config.yaml")) as f:
        return yaml.safe_load(f)


def _resample_pos_embed(state: dict, backbone, old_size, new_size) -> dict:
    """Позиционные эмбеддинги под другой размер входа.

    Нужно для второй ступени (D-056): переранжирование не входит в замер скорости, поэтому кандидатов
    можно прогонять на большем входе, чем тот, на котором модель обучалась. Интерполяция —
    стандартная timm-овская, та же, что применяется при загрузке предобученных весов.
    """
    from timm.layers import resample_abs_pos_embed

    patch = backbone.patch_embed.patch_size[0]
    grid = [int(new_size[0]) // patch, int(new_size[1]) // patch]
    prefix = getattr(backbone, "num_prefix_tokens", 1)
    for key in ("pos_embed", "backbone.pos_embed"):
        if key in state:
            state[key] = resample_abs_pos_embed(state[key], new_size=grid, num_prefix_tokens=prefix)
    return state


def load(model_dir: str, device: str = "cpu", input_size=None) -> Embedder:
    """Загружает артефакт; модель в режиме eval на указанном устройстве.

    input_size — прогнать модель на другом входе, чем тот, на котором она обучалась: позиционные
    эмбеддинги интерполируются. Масштаб объекта при этом меняется, поэтому результат всегда
    проверяется замером, а не принимается на веру.
    """
    cfg = load_config(model_dir)
    size = list(input_size) if input_size else cfg["input_size"]
    backbone = build_backbone(cfg["arch"], size, pretrained=False)
    model = Embedder(backbone, cfg.get("pool", "cls"), cfg.get("neck", "none"),
                     parts=int(cfg.get("parts", 0)))
    state = load_file(os.path.join(model_dir, "weights.safetensors"))
    if list(size) != list(cfg["input_size"]):
        state = _resample_pos_embed(state, backbone, cfg["input_size"], size)
    if not any(k.startswith("backbone.") for k in state):
        model.backbone.load_state_dict(state, strict=True)  # артефакт старого формата: только backbone
    else:
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()
