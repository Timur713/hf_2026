# Источники моделей и зависимостей


Автор дообученных весов и graph — DeepTech AI. Текущий кандидат D082:
[фиксированный Release](https://github.com/Timur713/hf_2026/releases/tag/d082-candidate-20261019-r2).
Архив идентичен A-пакету; новая версия выпуска обозначает перенос в этот репозиторий,
а не новое обучение. До повторной GPU-приёмки статус NOT_ACCEPTED.

Manifest SHA-256: `94125e2665558df43fb6701f7f0ca6b5c1cf5289cbfd48c83c03d1ee0180305c`.
Общий размер весов: 788873516 байт.

| Файл | SHA-256 |
|---|---|
| `components/0/weights.safetensors` | `c96604025d1e5a8a4f0b2e7b8f05ea1039b466203eacb8d20f6851f895283463` |
| `components/1/weights.safetensors` | `a694bd4509c91d13ee444d91e8ab5d734fed241c541b8bca3d329c99aad57c46` |
| `head/weights.safetensors` | `895663d191822ab9ca0c17ff82e89e6d48d9b957c160597ec0fe50913cb34536` |
| `verifier.pt` | `2017f6ce493800bfb60e5847a4b861f4b5f6d0e288e242c941e68f8c8345c659` |

Архив содержит training receipts, точные конфиги и фактические folds.
Исторический D065 сохранён в Release `fusion-pair-d065-20260927-rc1`.

Исходные pretrained (Apache-2.0, timm conversion):

| Автор / модель | Immutable revision URL | Исходный SHA-256 |
|---|---|---|
| Meta AI, DINOv2-B | [4685c99dabffe5affac90bd99dbffd25801ae58d](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m/tree/4685c99dabffe5affac90bd99dbffd25801ae58d) | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` |
| LAION/OpenCLIP, CLIP-B | [fbadfdd1c0e903c58855b70ae04740b22da9d305](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b/tree/fbadfdd1c0e903c58855b70ae04740b22da9d305) | `35fcf20089626ce1534aa3d4d8d97a02ca79e171a1735d0c8d58e150eded9ad5` |
| Meta AI, DINOv2-S verifier | [4610ca143709d58a633b6397a74412c2c3842454](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/tree/4610ca143709d58a633b6397a74412c2c3842454) | `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` |

Прямые ссылки на байты, размеры, авторы — [source_models.json](evidence/source_models.json). `scripts/analysis/verify_model_provenance.py` сверяет HF revision/LFS metadata; `--download` хеширует байты. Исходный pretrain SHA отличается от обученного checkpoint SHA. Verifier sidecar входит в манифест пакета.

Прямые и транзитивные runtime-зависимости точно закреплены в [requirements-lock-cu121.txt](../requirements-lock-cu121.txt), CUDA base — digest в Dockerfile. Основные: PyTorch2.5.1+cu121, torchvision0.20.1, timm1.0.15, numpy1.26.4, pandas2.2.3, Pillow11.3.0, safetensors0.5.3, huggingface_hub0.30.2, PyYAML6.0.2. FAISS1.9.0 — только ANN-анализ, не runtime.

Основной dataset — организаторы ЛЦТ2026, «Фалькон Тех». [VeRi-776](https://github.com/VehicleReId/VeRidataset) (Xinchen Liu и соавторы) и [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html) (PKU,CVPR2016) исследованы, но не вошли в финальное обучение пары. Данные имеют отдельные условия доступа и не распространяются в образе. VERI-Wild,DINOv3,fast-reid,SigLIP не входят в финальный инференс.
