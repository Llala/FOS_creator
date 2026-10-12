# ФОС RAG с GigaChat

Этот вариант заменяет OpenAI в существующем проекте: GigaChat создаёт
эмбеддинги и генерирует черновик ФОС. Парсер учебного плана менять не требуется.
SQLite по-прежнему хранит метаданные и векторы; поиск использует косинусное
сходство и небольшой бонус за совпадение слов. Это PoC для небольшого корпуса.

## Установка

Сохраните прежние файлы перед заменой. Скопируйте из архива в каталог `rag/`
три файла: `fos_rag.py`, `ingest_docx_folder.py`, `load_rpd_folder.py`.
Нужны все три: прежний загрузчик импортировал OpenAIEmbedder.
Рядом положите `requirements-gigachat.txt`.

```bash
pip install -r requirements-gigachat.txt
```

Версия SDK зафиксирована: gigachat 0.2.3. Используется поддерживаемый
совместимый вызов `client.chat(Chat(...))`, ответ читается из
`response.choices[0].message.content`. Эмбеддинги: `client.embeddings()`.
Python 3.10–3.13.

## Авторизация

Получите **ключ авторизации** GigaChat API в личном кабинете. Это значение
для `credentials`, а не Client ID, Client Secret по отдельности и не токен
доступа. SDK получает и обновляет токен автоматически.

В Linux/bash можно ввести ключ без отображения и сохранения самого ключа
в истории команд:

```bash
read -rsp 'Ключ авторизации GigaChat: ' GIGACHAT_CREDENTIALS
export GIGACHAT_CREDENTIALS
export GIGACHAT_SCOPE='GIGACHAT_API_PERS'
```

Для ИП/юрлиц используйте scope, соответствующий подключению:
`GIGACHAT_API_B2B` или `GIGACHAT_API_CORP`.
Ключ не записывается в SQLite, JSON-результат и исходный код.
Переменные нужно задать в том же терминале, где запускаете Python.

Проверка TLS включена по умолчанию. Если требуется сертификат Минцифры,
установите доверенные сертификаты по официальной инструкции и укажите
путь к PEM-файлу CA-сертификатов:

```bash
export GIGACHAT_CA_BUNDLE_FILE='/полный/путь/ca-bundle.pem'
```

Не задавайте этот путь, если такого файла у вас нет.
SDK также поддерживает `GIGACHAT_BASE_URL`, если для вашего подключения
нужен другой адрес API.

Проверить авторизацию и доступные идентификаторы моделей:

```bash
python fos_rag.py models
```

Генерация по умолчанию использует `GigaChat-2`; эмбеддинги —
`EmbeddingsGigaR`. Если модель недоступна вашему аккаунту, выберите
доступный идентификатор из `models`. Модель эмбеддингов при загрузке
и поиске должна совпадать.

## Загрузка всех РПД

Создайте **новую базу**: векторы старого индекса OpenAI несовместимы
с эмбеддингами GigaChat. Простого переименования старой базы недостаточно.

Команды из каталога `rag/`, если папка `rpd/` находится в нём:

```bash
python load_rpd_folder.py ./rpd --db rpd_gigachat.sqlite3 --dry-run
python load_rpd_folder.py ./rpd --db rpd_gigachat.sqlite3 --report rpd_upload_report.json
python fos_rag.py --db rpd_gigachat.sqlite3 status --discipline-code 'Б1.О.05'
```

Если папка находится уровнем выше, замените `./rpd` на `../rpd`.
Загрузчик берёт полный код дисциплины из имени файла, например
`Б1.В.ДВ.01.02 Название дисциплины РП.docx`. Для имени без кода
используйте `--manifest rpd_map.json`:

```json
{
  "sports_marketing.docx": {
    "discipline_code": "Б1.В.07"
  }
}
```

Код в этом примере нужно заменить фактическим кодом из вашего плана.
Для разных дисциплин создавайте отдельные записи в манифесте.

Загрузчик всех DOCX также сохранён. Например, для папки с примерами ФОС:

```bash
python ingest_docx_folder.py ./fos --db rpd_gigachat.sqlite3 --type fos_example \
  --curriculum ../parser/parsed_curriculum.json --dry-run
```

После проверки уберите `--dry-run`. Для РПД/примеров ФОС требуется код
дисциплины. Методика и шаблон могут быть общими:

```bash
python fos_rag.py --db rpd_gigachat.sqlite3 ingest ./methodology.docx --type methodology
python fos_rag.py --db rpd_gigachat.sqlite3 ingest ./template.docx --type fos_template
```

## Поиск и генерация

Получить JSON-контекст для дисциплины «Физическая культура и спорт»:

```bash
python fos_rag.py --db rpd_gigachat.sqlite3 context \
  --curriculum ../parser/parsed_curriculum.json \
  --discipline-code 'Б1.О.05' \
  --query 'Темы, задания для зачёта и критерии оценивания' \
  -o physical_culture_context.json
```

В `Б1.О.05` используется русская буква О. Код должен точно совпадать
в учебном плане и индексе.

Создать черновик ФОС:

```bash
python fos_rag.py --db rpd_gigachat.sqlite3 draft \
  --curriculum ../parser/parsed_curriculum.json \
  --discipline-code 'Б1.О.05' \
  --query 'Темы, задания для зачёта и критерии оценивания' \
  --generation-model GigaChat-2 \
  --max-tokens 4000 \
  -o physical_culture_fos.json
```

Например, `--generation-model GigaChat-2-Pro` можно использовать, если
этот идентификатор есть в списке моделей вашего аккаунта. Генеративную
модель можно менять без переиндексации.

Для изменения **модели эмбеддингов** явно задайте её обеим командам и
загрузите документы в новую базу:

```bash
python load_rpd_folder.py ./rpd --db rpd_embeddings.sqlite3 --embedding-model Embeddings
python fos_rag.py --db rpd_embeddings.sqlite3 --embedding-model Embeddings search \
  --discipline-code 'Б1.О.05' --query 'Критерии зачёта'
```

`--db` и `--embedding-model` у `fos_rag.py` ставятся **до** подкоманды.

## Результат и ограничения

JSON-черновик содержит `draft`, `sources`, `warnings`, `generation_model`,
`finish_reason` и статус `requires_teacher_review`. Ссылки `[S1]` ведут
к сохранённым источникам. Скрипт предупреждает о неизвестных ссылках,
отсутствии ссылок и обрезанном ответе. Он не проверяет, действительно ли
каждый источник подтверждает текст задания. Итог проверяет преподаватель.

Поиск выбирает до 6 фрагментов (можно задать `--limit`), поэтому не
гарантирует охват всех тем РПД. PDF без текстового слоя требуют OCR.
Файл результата — JSON; экспорт в Word здесь не реализован.

`status` и `--dry-run` работают без SDK-запросов и ключа. Остальные
команды обращаются к GigaChat API; найденные фрагменты отправляются
модели при генерации. При ошибках провайдера выводится тип ошибки
и HTTP-статус, если он доступен, без тела ответа.

## Проверка

```bash
python -m unittest -v test_gigachat_provider.py test_fos_rag.py \
  test_ingest_docx_folder.py test_load_rpd_folder.py
```

Все 17 тестов прошли. Тесты адаптера используют настоящий SDK 0.2.3
с MockTransport: проверяются формат запросов, порядок векторов,
повторные вызовы, предупреждения генерации и обработка ошибок.
Реальные запросы с пользовательским ключом не выполнялись.

Официальная документация:

- SDK: https://developers.sber.ru/docs/ru/gigachat/guides/using-sdks
- Модели: https://developers.sber.ru/docs/ru/gigachat/models/main
- Авторизация: https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-create-project
- Сертификаты: https://developers.sber.ru/docs/ru/gigachat/certificates
- Совместимость SDK: https://github.com/ai-forever/gigachat
