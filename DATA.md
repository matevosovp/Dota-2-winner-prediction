# Контракт данных

Исходные данные не хранятся в Git. Это предотвращает случайную публикацию файлов, распространяемых отдельно, и не смешивает данные с исходным кодом.

## Ожидаемая структура

```text
.
├── train_features.csv
├── train_targets.csv
└── DOTA 2/
    └── train_matches.jsonl
```

`test_matches.jsonl` не нужен для offline-оценки без доступного таргета.

## `train_features.csv`

Одна строка соответствует одному snapshot матча.

Обязательные условия:

- `match_id_hash` существует, уникален и не содержит пропусков;
- значения отражают только состояние, доступное к зафиксированному `game_time`;
- таблица не содержит `radiant_win` и post-match полей;
- если присутствует одна колонка метрики вида `r1_gold`, должны присутствовать все десять колонок `r1-r5` и `d1-d5` этой метрики.

Пайплайн явно блокирует известные признаки завершённого матча: итоговый счёт, статусы башен, `hero_damage`, `total_damage`, `towers_killed`, `roshans_killed` и аналогичные поля.

## `train_targets.csv`

Обязательные поля:

- `match_id_hash` — тот же уникальный идентификатор;
- `radiant_win` — бинарный target `0/1` без пропусков.

Множества идентификаторов в feature и target таблицах должны совпадать полностью. Inner join с потерей строк считается ошибкой.

## `train_matches.jsonl`

Каждая строка — JSON-объект OpenDota с `match_id_hash` и массивом `players`.

Сырой объект описывает завершённый матч, поэтому большинство его полей являются потенциальным future leakage. `dota_predictor.data.load_pregame_metadata_jsonl` использует allowlist. В текущем эксперименте извлекается только:

- `hero_name` — выбранный герой и сторона, известные до начала матча.

Поля `damage`, `xp`, `towers_killed`, `roshans_killed` и другие итоговые показатели функция присоединить не позволит. Если в будущем понадобится временной ряд, он должен быть явно обрезан по snapshot timestamp и покрыт отдельными тестами границы времени.

## Проверка

```bash
python -m dota_predictor.train \
  --features train_features.csv \
  --targets train_targets.csv \
  --matches-jsonl "DOTA 2/train_matches.jsonl" \
  --outer-splits 5 \
  --inner-splits 3
```

Если схема, идентификаторы, target или temporal contract нарушены, команда завершается с `DataContractError` до обучения модели.
