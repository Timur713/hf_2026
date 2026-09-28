# Тимуру: безопасная публикация без приглашения аккаунта B

Цель — `Timur713/hf_2026`. При проверке 28.09 он PUBLIC, пустой, анонимно HTTP 200.
У B только READ; его аккаунт не находится в приглашениях и снаружи возвращает 404.
GitHub Support подтвердил flagged-состояние; пересмотр ещё не завершён.
Открывать старый приватный репозиторий
или переносить его Git-историю **нельзя**: там были credentials внешних ресурсов.

## Что передано и что остаётся

- Приватная ветка `b/final-publication`: скрипты экспорта и демонстрации,
  90-секундное видео, transcript, проверки, инструкции, новые публичные URL.
- Финальные веса уже лежат в приватном Release `fusion-pair-d065-20260927-rc1`.
  Их не переобучать и не перепаковывать: переносим точные байты и SHA.
- Отдельный приватный Release `delivery-handoff-20260928` предназначен для
  готового чистого snapshot-архива, manifest, видео и контрольных сумм.
  Если используете архив, сначала проверьте `HANDOFF-SHA256SUMS.txt`, затем
  распакуйте его в новый каталог. В нём намеренно нет `.git`: создайте новый
  root commit командами ниже, начиная с `git init` и указав каталог распаковки.
- Модель, порог, канонический `submission/` и уже измеренные метрики не изменены.
- Не сливать эту ветку в приватный `main` до публикации assets и чистой проверки:
  новые Docker URL пока указывают на ещё не заполненный публичный репозиторий.
- Нужны действия владельца: push чистого snapshot, публикация точных assets,
  презентация A, проверка публичной сборки и заполнение формы капитаном.
  Передача комплекта **не означает отправку решения**.

## 1. Чистая копия

Из существующего приватного checkout (не переключая грязную рабочую ветку):

```bash
git fetch origin b/final-publication
git worktree add ../falcon-publication-source origin/b/final-publication
cd ../falcon-publication-source
delivery_stage=$(mktemp -d)
delivery_stage=$(cd "$delivery_stage" && pwd -P)
python3 scripts/export_public_snapshot.py --source "$PWD" \
  --destination "$delivery_stage/public"
git -C "$delivery_stage/public" init -b main
git -C "$delivery_stage/public" add .
git -C "$delivery_stage/public" \
  -c user.name='DeepTech AI' \
  -c user.email='295956991+makdar2004-boop@users.noreply.github.com' \
  commit -m 'Clean public delivery snapshot — final pair D-065'
git -C "$delivery_stage/public" remote add origin https://github.com/Timur713/hf_2026.git
git -C "$delivery_stage/public" push -u origin main
```

Если публичный репозиторий уже непустой — остановиться и сверить изменения,
**не force-push**. Snapshot содержит only tracked файлы, новый root commit,
export_manifest с SHA/исключениями. Персональные документы, история, исходные
изображения, приватные логи не копируются. Raw FAIL не переписывается какPASS.
Публичная подпись команды намеренно не содержит локальный hostname/email.
Можно использовать собственную GitHub noreply-почту; не подставляйте личную
почту или данные другого человека без его согласия.

## 2. Точные финальные веса и видео

Использовать авторизованный `gh` Тимура **на хосте**, никогда не в Docker.

```bash
mkdir "$delivery_stage/assets"
gh release download fusion-pair-d065-20260927-rc1 \
  --repo Timur713/hack_falcon_2026 --dir "$delivery_stage/assets" \
  --pattern fusion-pair-d065-bundle.tar.gz --pattern manifest.json \
  --pattern fusion_config.json --pattern SHA256SUMS.txt
shasum -a 256 "$delivery_stage/assets/fusion-pair-d065-bundle.tar.gz"
shasum -a 256 "$delivery_stage/assets/manifest.json"
```

Требуются SHA:

- archive:`9eeb99fae8bdcb228265ee83ac51da60da9896199d4746d843b30f3976fa49a8`;
- manifest:`4e3eacd2431ae634369b8e34689cb48d579f034207d1fd72bdff29661ad71e22`.

Только после совпадения:

```bash
gh release create fusion-pair-d065-20260927-rc1 \
  --repo Timur713/hf_2026 --target main \
  --title 'Final D-065 pair — immutable weights' \
  --notes 'DINO392+CLIP288, verifier392, D3072, w0.15, threshold6.6. Exact mirror of the verified final bundle. See README for acceptance limits.' \
  "$delivery_stage/assets/fusion-pair-d065-bundle.tar.gz" \
  "$delivery_stage/assets/manifest.json" \
  "$delivery_stage/assets/fusion_config.json" \
  "$delivery_stage/assets/SHA256SUMS.txt" \
  docs/delivery/demo_20260928/falcon-demo-90s.mp4
```

Не зеркалить все старые Releases и необследованные архивы. Если tag/asset уже
существует — проверить SHA,не перезаписывать. Внутренние provenance URLs в bundle
не изменять: публичный архив должен остаться побайтно тем же.

## 3. Передать B ссылки и пройти публичную проверку

После upload B сможет анонимно скачать clone+weights и проверить новый build.
Платный GPU уже в Shelve; восстановление только при необходимости/согласованном
бюджете. Пока не пройден anonymous clone→build→offline run,не отмечать егоPASS.

Планируемые ссылки (assets **не были** публично доступны на момент передачи):

- Код: https://github.com/Timur713/hf_2026
- Документация: https://github.com/Timur713/hf_2026/blob/main/README.md
- Прототип: https://github.com/Timur713/hf_2026/releases/download/fusion-pair-d065-20260927-rc1/falcon-demo-90s.mp4
- Дополнительные материалы: https://github.com/Timur713/hf_2026/releases/tag/fusion-pair-d065-20260927-rc1
- Презентация: добавить фактические PDF/PPTX отA,сверить шаблон/цифры/публичныйдоступ.

В кабинете Макар виден как участник,Тимур—капитан. У Макара в просмотренных разделах
нет формы сдачи; капитану проверить собственный кабинет и заполнить поля.
До отправки совместно сверить ссылки. После подтверждённой подачи сохранить
скриншот/receipt и заморозить сданную ветку и assets. Сейчас submissionформы нет.
