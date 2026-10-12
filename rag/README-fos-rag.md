# RAG для PoC генерации ФОС

Парсер `fos_curriculum_parser.py` извлекает факты из Excel. Этот модуль индексирует
только текстовые источники: РПД, методические требования, шаблоны и примеры ФОС.
Он возвращает найденные отрывки с точным указанием документа и места, а при
запросе `draft` вызывает модель для подготовки черновика. Решение рассчитано
на небольшой корпус документов и проверку преподавателем.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-parser.txt -r requirements-rag.txt
export OPENAI_API_KEY="<ваш API-ключ>"
```

Сначала создайте `parsed_curriculum.json` командой из README парсера или
используйте предоставленный файл.

## Индексация

```bash
python fos_rag.py --db index.sqlite3 ingest rpd_marketing.pdf \
  --type rpd --discipline-code 'Б1.В.07' --competencies 'ПК-7,ПК-9'

python fos_rag.py --db index.sqlite3 ingest fos_template.docx \
  --type fos_template

python fos_rag.py --db index.sqlite3 ingest methodology.pdf \
  --type methodology

python fos_rag.py --db index.sqlite3 ingest approved_marketing_fos.docx \
  --type fos_example --discipline-code 'Б1.В.07'
```

Для первого запуска достаточно собственной РПД и одного методического
документа. `rpd` и `fos_example` обязательно привязываются к дисциплине;
глобальные методики/шаблоны могут использоваться для всех дисциплин.
Индексирование повторно не отправляет неизменившийся файл за эмбеддингами.
При изменении файла или его метаданных старые фрагменты заменяются в транзакции.

## Поиск, контекст и черновик

```bash
python fos_rag.py --db index.sqlite3 search \
  --discipline-code 'Б1.В.07' --competency 'ПК-7' \
  --query 'Рыночные риски спортивных услуг и критерии оценивания'

python fos_rag.py --db index.sqlite3 context \
  --curriculum parsed_curriculum.json \
  --discipline-code 'Б1.В.07' \
  --query 'Темы, индикаторы ПК-7, экзаменационные задания и критерии' \
  -o marketing_context.json

python fos_rag.py --db index.sqlite3 draft \
  --curriculum parsed_curriculum.json \
  --discipline-code 'Б1.В.07' \
  --query 'Темы и задания для проверки ПК-7 и ПК-9' \
  --generation-model '<доступная текстовая модель>' \
  -o marketing_draft.json
```

Аргумент `--embedding-model` должен быть одинаковым для индексации и поиска.
Для каждого фрагмента хранится файл, раздел, страница PDF или номер абзаца DOCX,
а также устойчивый `source_id`. В запросе к модели отрывкам присваиваются
краткие ссылки `[S1]`, `[S2]`, которые возвращаются вместе с черновиком.

Индекс основан на SQLite и точном вычислении косинусного сходства с небольшим
бонусом за совпадение терминов. Поиск сначала строго фильтрует дисциплину:
чужая РПД и чужой утверждённый ФОС не участвуют. Для большой базы этот
класс хранения можно заменить на векторную БД.

Для работы с облачным API тексты индексируемых документов и найденных
фрагментов передаются провайдеру модели. Размещайте в корпусе только документы,
для которых это допустимо.

Сканированные PDF требуют OCR до индексации. Из DOCX извлекаются абзацы и
таблицы. Данные из Excel никогда не эмбеддятся: модель получает их отдельным
полем `curriculum_facts`.

Проверка без API:

```bash
python -m unittest -v test_fos_rag.py
```

Для полноценного ФОС нужны РПД, индикаторы достижения компетенций и локальные
требования вуза. Отсутствующие данные модель обязана явно назвать; результат
`draft` всегда имеет статус `requires_teacher_review`.
