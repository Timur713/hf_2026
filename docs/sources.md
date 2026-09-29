# Источники моделей и зависимостей


Автор дообученных весов — DeepTech AI. Финальные файлы распространяются вместе
в [публичном pair bundle](https://github.com/Timur713/hf_2026/releases/tag/fusion-pair-d065-20260927-rc1).
SHA отдельных файлов:

| Файл | SHA-256 |
|---|---|
| `hires392_all.safetensors` | `f69eb2283c03a2748ac44cb4e1a4902bbc6f3129440cf3799edafa0f0ea0aaeb` |
| `div_clip_all.safetensors` | `6157e1eb862a0c88f637ed2a163bf6bbbe52459a88d41f3b623e83e69180548f` |
| `v_fus_all_frz_input392.pt` | `aa254e396ac0be78d29f32641e83f8d18e165d76eb3345c18c1054d105b9abf5` |

Исходные pretrained (Apache-2.0, timm conversion):

| Автор / модель | Immutable revision URL | Исходный SHA-256 |
|---|---|---|
| Meta AI, DINOv2-B | [4685c99dabffe5affac90bd99dbffd25801ae58d](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m/tree/4685c99dabffe5affac90bd99dbffd25801ae58d) | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` |
| LAION/OpenCLIP, CLIP-B | [fbadfdd1c0e903c58855b70ae04740b22da9d305](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b/tree/fbadfdd1c0e903c58855b70ae04740b22da9d305) | `35fcf20089626ce1534aa3d4d8d97a02ca79e171a1735d0c8d58e150eded9ad5` |
| Meta AI, DINOv2-S verifier | [4610ca143709d58a633b6397a74412c2c3842454](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/tree/4610ca143709d58a633b6397a74412c2c3842454) | `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` |

Прямые ссылки на байты, размеры, авторы — [source_models.json](evidence/source_models.json). `scripts/analysis/verify_model_provenance.py` сверяет HF revision/LFS metadata; `--download` хеширует байты. Исходный pretrain SHA отличается от обученного checkpoint SHA. Verifier sidecar входит в манифест пакета.

Прямые и транзитивные runtime-зависимости точно закреплены в [requirements-lock-cu121.txt](../requirements-lock-cu121.txt), CUDA base — digest в Dockerfile. Основные: PyTorch2.5.1+cu121, torchvision0.20.1, timm1.0.15, numpy1.26.4, pandas2.2.3, Pillow11.3.0, safetensors0.5.3, huggingface_hub0.30.2, PyYAML6.0.2. FAISS1.9.0 — только ANN-анализ, не runtime.

Основной dataset — организаторы ЛЦТ2026, «Фалькон Тех». [VeRi-776](https://github.com/VehicleReId/VeRidataset) (Xinchen Liu и соавторы) и [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html) (PKU,CVPR2016) исследованы, но не вошли в финальное обучение пары. Данные имеют отдельные условия доступа и не распространяются в образе. VERI-Wild,DINOv3,fast-reid,SigLIP не входят в финальный инференс.
