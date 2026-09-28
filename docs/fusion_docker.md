# Финальная пара D-065: Docker и проверяемый пакет

`docker build .` и `docker compose run --build --rm reid` выбирают **пару DINO392 + CLIP288**, verifier input392, `rerank_weight=0.15`, `candidate_threshold=6.6`, D=3072. SigLIP отсутствует. Исторический одиночный режим доступен явно: `docker build --target legacy …`.

## Зафиксированные байты

| Поле | Значение |
|---|---|
| Release | [fusion-pair-d065-20260927-rc1](https://github.com/Timur713/hf_2026/releases/tag/fusion-pair-d065-20260927-rc1) |
| Архив | [fusion-pair-d065-bundle.tar.gz](https://github.com/Timur713/hf_2026/releases/download/fusion-pair-d065-20260927-rc1/fusion-pair-d065-bundle.tar.gz) |
| SHA-256 архива | `9eeb99fae8bdcb228265ee83ac51da60da9896199d4746d843b30f3976fa49a8` |
| SHA-256 manifest | `4e3eacd2431ae634369b8e34689cb48d579f034207d1fd72bdff29661ad71e22` |
| Веса | **788647632 bytes**, меньше2000000000 |
| Verifier SHA | `aa254e396ac0be78d29f32641e83f8d18e165d76eb3345c18c1054d105b9abf5` |
| Статистика компонентов | `0.05045982709670803`, `0.06659743137636251` |

Статистика получена на636 полных кадрах **train fold0 gallery**, без query, локальным `prepare_fusion_artifact.py` (CPU, PyTorch2.2.2, batch16). Runtime Linux отдельно закреплён lock cu121/PyTorch2.5.1; разница окружений подготовки записана в manifest. Это упаковка фиксированных train-only масштабов, не новая оценка качества или калибровка порога.

Артефакт — immutable RC финального состава. Фактические результаты Docker/GPU-приёмки фиксируются отдельно в [чек-листе](final_delivery_checklist.md); сам pin не означает PASS.

## Публичная поставка и необязательный кэш

Состояние зеркала проверяйте в[материалах сдачи](delivery/README.md): до загрузки
assets владельцем новыйURL может возвращать404. После публикации clean build
скачивает архив автоматически,без`gh` и авторизации. Для необязательного кэша:

```bash
mkdir -p artifacts
test ! -e artifacts/fusion_bundle
test ! -e artifacts/fusion_bundle.tar.gz
curl --fail --location --proto '=https' --tlsv1.2 \
  https://github.com/Timur713/hf_2026/releases/download/fusion-pair-d065-20260927-rc1/fusion-pair-d065-bundle.tar.gz \
  --output artifacts/fusion_bundle.tar.gz
printf '%s  %s\n' \
  9eeb99fae8bdcb228265ee83ac51da60da9896199d4746d843b30f3976fa49a8 \
  artifacts/fusion_bundle.tar.gz | sha256sum --check --strict
```

На macOS замените `sha256sum` на `shasum -a 256`. При ошибке остановитесь, не продолжайте с неверным архивом. После открытия репозитория чистый clone/build загрузит тот же фиксированный URL автоматически, без `gh`.

```bash
mkdir -p outputs
docker compose run --build --rm reid
```

Или без Compose:

```bash
docker build -t falcon-reid:pair-d065 .
docker run --rm --gpus all --network none \
  --mount "type=bind,src=$PWD/dataset,dst=/data,readonly" \
  --mount "type=bind,src=$PWD/outputs,dst=/out" \
  falcon-reid:pair-d065
```

Во время runtime доступны только dataset read-only и output. Код, конфиги и все веса уже внутри image; внешние model/source mounts запрещены при приёмке. `--network none` относится к запуску. Образ собирается с разрешённой сетью для base/pip/release, зависимости фиксированы.

## Целостность и происхождение

Приоритет: каталог `artifacts/fusion_bundle` → архив `artifacts/fusion_bundle.tar.gz` → фиксированный HTTPS URL. **Неверный существующий кэш не переключается в сеть.** Archive SHA и manifest SHA независимы; каждый файл сверяется по размеру/SHA. Допустим только ограниченный USTAR без ссылок, traversal, GNU/PAX, лишних файлов и секретов. Все веса проверяются на≤2GB, каждый компонент и verifier — на all-data metadata.

```text
manifest.json
fusion_config.json
components/0/{model_config.yaml,weights.safetensors}
components/1/{model_config.yaml,weights.safetensors}
verifier.pt
verifier.provenance.yaml
```

Metadata-only исправленные YAML — из [D-066](benchmarks/final_pair_provenance_20260927/). Sidecar verifier привязан к SHA checkpoint и сам включён в manifest. Источники конечных весов — фиксированные URLs `fusion-all-v1`; исходные pretrained revision/автор/SHA — в YAML/sidecar и README. Installation receipt создаётся внутри образа, не добавляется в исходный архив.

Повторение подготовки (нужны проверенные исходные веса и новый каталог результата):

```bash
PYTHONPATH=. python scripts/prepare_fusion_artifact.py \
  --component artifacts/pair_sources/dino --component artifacts/pair_sources/clip \
  --images dataset/images --gallery splits/fold0_gallery.csv \
  --out artifacts/fusion_bundle --name fusion_pair_d065_all \
  --device cpu --batch-size 16 --require-all-data \
  --verifier artifacts/pair_sources/verifier.pt \
  --verifier-provenance docs/benchmarks/final_pair_provenance_20260927/v_fus_all_frz_input392.provenance.yaml \
  --rerank-weight 0.15 --candidate-threshold 6.6 \
  --threshold-source 'D-065 / A-104 matched OOF folds0/1'
```

Для точного восстановления распространённых байтов используйте pinned archive: разные вычислительные платформы могут дать слегка разные gallery scales. Не перепаковывайте пакет под тем же URL/SHA. Состав/порог не подбираются сборщиком.

## Приёмка готового image

27.09 финальная пара реально прошла offline build/tests/C6/all-data budget,
четыре полных запуска с одинаковымиSHA,trainmasked-control и fixed-shape
query content-control. [Image identity и отчёты](benchmarks/final_pair_acceptance_20260927/README.md).
Отдельный singleton/batched strict numeric-check сохранилFAIL поconfidence
до0.000393,без смены IDs/отказов; wrapper не выдаёт безусловныйPASS.

Запускать на Linux/GPU из чистого tracked-source checkout и проверенного кэша:

```bash
BUILD_CONTEXT="$PWD" DATA_ROOT=/absolute/dataset \
  OUTPUT_ROOT=/absolute/new-pair-acceptance \
  FUSION_IMAGE=falcon-reid:pair-d065 \
  bash scripts/fusion_rc_acceptance_session.sh
```

Очередь: build → offline tests/smoke50 → два полных test-прогона/C6/SHA → train-only masked-control → all-data weight gate → полный C5 → query-independence. Ненулевой код query-проверки нельзя скрывать; raw отчёт отделяет изменения IDs/отказов от FP16 confidence drift. Геометрическая маска не доказывает покрытие настоящих номеров.

В итоговом отчёте записываются source commit, Docker image ID (`sha256`), registry digest если образ опубликован, archive/manifest SHA и SHA каждого из трёх выходных файлов. Локальный image ID не выдаётся за registry manifest digest. Канонический `submission/` копируется только из проверенного запуска, без ручного редактирования результатов. GPU после сохранения результатов — Shelve/освобождение, не обычный Stop с продолжающейся тарификацией.
