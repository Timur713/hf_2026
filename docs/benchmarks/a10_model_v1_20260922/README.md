# A10 acceptance: model-v1, 22.09.2026

Воспроизводимая команда:

```bash
scripts/a10_acceptance.sh \
  --data dataset \
  --model artifacts/models/final \
  --output outputs/a10_day3 \
  --python .venv/bin/python \
  --label a10-model-v1-20260922
```

Стенд: NVIDIA A10 24 ГБ, драйвер 615.71.09, PyTorch 2.5.1+cu121,
8 vCPU, 32 ГБ RAM, HDD. Исходный код соответствовал `95c9701`; fresh run
directory не содержал `.git`, поэтому поле `commit` в `extract.json` равно
`null`.

Результат: 54 теста и real-image smoke прошли; Docker build проверил sha256
обоих артефактов model-v1; Docker run с `network_mode: none` обработал полный
test query/gallery и встроенный валидатор принял все три выходных файла.

Контрольные суммы выходов offline Docker-run:

| файл | sha256 |
|---|---|
| `submission.csv` | `4186b6424eec7a06eab796da51f71b88488dcb366a0a35a50b0c891ae0205ef0` |
| `embeddings.npy` | `49d562036b5185196e503be6cb6878d39429729ad87234e05a43ccb72f2fb152` |
| `candidates.csv` | `809e6a7ac0e9431d596a6e96e48d02b9ab41782cdec9dba6814cfd936faea223` |

Сами выходы и полный лог не коммитятся: это воспроизводимые игнорируемые
артефакты. Машиночитаемые результаты скорости сохранены рядом.
