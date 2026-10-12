# Пакетная загрузка DOCX для RAG проекта ФОС

Скрипт `ingest_docx_folder.py` находит `.docx` в папке (включая вложенные папки)
и добавляет их в тот же SQLite индекс, который читает `fos_rag.py`. Временные
файлы Word `~$...docx` и символические ссылки пропускаются. Для каждого файла
он сохраняет статус: `indexed`, `unchanged` или `failed`.

Установите зависимости из `requirements-rag.txt` и задайте `OPENAI_API_KEY`.

## Папки с разными дисциплинами

Если РПД и ФОС лежат в разных папках, а внутри каждой папки есть документы
разных дисциплин, используйте `parsed_curriculum.json` предыдущего парсера.
Скрипт ищет код дисциплины (`Б1.В.07`) или её название (`Маркетинг`) в имени
файла. Компетенции берёт из учебного плана.

```bash
python ingest_docx_folder.py ./RPD --type rpd \
  --curriculum parsed_curriculum.json --db index.sqlite3 --dry-run

python ingest_docx_folder.py ./FOS --type fos_example \
  --curriculum parsed_curriculum.json --db index.sqlite3 --dry-run
```

Если все соответствия верны, повторите команды без `--dry-run`. Укажите один
и тот же `--db` для обеих папок. Например:

```bash
python ingest_docx_folder.py ./RPD --type rpd \
  --curriculum parsed_curriculum.json --db index.sqlite3

python ingest_docx_folder.py ./FOS --type fos_example \
  --curriculum parsed_curriculum.json --db index.sqlite3
```

Если название не найдено или подходит нескольким дисциплинам, скрипт покажет
все проблемные файлы и не начнёт индексацию. Для таких документов задайте код
явно в манифесте, как показано ниже. Автоматическое сопоставление по имени
работает только для РПД и примеров ФОС; общие шаблоны/методички задавайте
через манифест с `fos_template` или `methodology`.

Если в имени указан код, но он отсутствует в `parsed_curriculum.json`, или
название противоречит коду плана, сначала проверьте, к какой образовательной
программе относятся документы. Например, одно и то же обозначение в разных
планах может относиться к разным дисциплинам. Для документов из другого плана
лучше построить новый `parsed_curriculum.json` и использовать отдельный `--db`.
Если такого плана пока нет, разрешите индексацию кодов из имён файлов явно:

```bash
python ingest_docx_folder.py ./RPD --type rpd \
  --db other_program.sqlite3 --allow-unknown-codes --dry-run
```

В этом режиме компетенции неизвестных дисциплин не берутся из учебного плана;
при необходимости задайте их через манифест. Перед загрузкой повторите
команду без `--dry-run`. Не объединяйте такую базу с индексом данного плана.

## Другие способы назначения метаданных

Если все документы одной дисциплины являются РПД:

```bash
python ingest_docx_folder.py ./rpd/marketing \
  --db index.sqlite3 --type rpd \
  --discipline-code 'Б1.В.07' --competencies 'ПК-7,ПК-9' \
  --report ingest_report.json
```

Для папки с разными типами документов используйте JSON манифест. Ключ —
относительный путь от папки, включая вложенные каталоги:

```json
{
  "rpd/marketing.docx": {
    "document_type": "rpd",
    "discipline_code": "Б1.В.07",
    "competencies": ["ПК-7", "ПК-9"]
  },
  "examples/approved_marketing.docx": {
    "document_type": "fos_example",
    "discipline_code": "Б1.В.07"
  },
  "templates/fos_template.docx": {
    "document_type": "fos_template"
  },
  "methodology/requirements.docx": {
    "document_type": "methodology"
  }
}
```

Сначала проверьте назначение метаданных без вызовов API:

```bash
python ingest_docx_folder.py ./documents --manifest documents.json --dry-run
```

Затем индексируйте:

```bash
python ingest_docx_folder.py ./documents --manifest documents.json \
  --db index.sqlite3 --report ingest_report.json
```

Если используется манифест, для каждого DOCX должна быть запись или должны
быть заданы подходящие общие значения `--type` и `--discipline-code`.
Скрипт проверяет метаданные всех найденных файлов до начала запросов API.
Ошибка чтения одного файла записывается в отчёт, остальные обрабатываются;
код выхода 1 означает, что часть файлов не удалось загрузить. Повторный запуск
не пересчитывает эмбеддинги для неизменившихся файлов.

`--no-recursive` ограничивает поиск только указанной папкой. Установите
одинаковые `--db` и `--embedding-model` при загрузке и поиске. Как и в
одиночном режиме, документы передаются провайдеру для вычисления эмбеддингов.
