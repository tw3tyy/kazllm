# KAZLLM · показ проекта на защите

Открой Terminal в папке проекта и запусти меню.

```sh
cd "/path/to/kazllm-github"
./demo.sh
```

Если открыта старая версия меню, закрой её через Ctrl+C в её окне и запусти
команду заново. Если в отдельном окне идёт сбор ответов, открой меню в новом окне.
Введи номер пункта и нажми Enter. После просмотра нажми Enter для возврата или сразу введи номер следующего пункта.

| Пункт | Английская надпись | Что появляется |
|---|---|---|
| 1 | Run progress | Число сохранённых ответов из 830 для каждой модели. Пробные ответы посчитаны отдельно. |
| 2 | Saved examples | Автоматически выбранные сохранённые примеры перевода, QA и probe. |
| 3 | Inspect an item | Задание по номеру и сохранённые ответы выбранной модели. Для ожидающих ответов выводятся текст запроса и статус Pending. |
| 4 | Results and baselines | Готовая таблица или статус Pending и пересчитанные локальные baseline. |
| 5 | Method and dataset audit | Устройство эксперимента, хеш набора и пересечение pilot/test. |
| 0 | Exit | Выход. |

## Команды без меню

```sh
./demo.sh --status
./demo.sh --examples --task mt
./demo.sh --examples --task qa
./demo.sh --examples --task probe
./demo.sh --item 201 --model 1
./demo.sh --results
./demo.sh --method
```

Модель 1 означает Gemini, 2 означает Qwen, 3 означает Gemma. Номер задания
можно выбрать от 1 до 430. Позиция 1 содержит перевод, 201 содержит QA,
401 содержит языковой probe. Для произвольного номера меню показывает
языковые условия этого задания.

На экране примера сначала идёт PROMPT, затем SAVED RESPONSE. Для QA выводится
сравнение буквы с эталоном. Для перевода выводятся эталон и chrF2 отдельного
предложения. В RESPONSE RECORD видны точная модель, время, ID ответа и папка
лога. outputs/ хранит ответы эксперимента. smoke_outputs/ хранит пробные ответы.

## Новый запрос на защите

Когда другие запуски в этой папке закончились и проект Google остаётся на Free tier
с отключённым billing, выполни команду ниже. Ввод ключа скрыт.

```sh
./demo.sh --live-gemini 201 --free-tier-confirmed
```

Команда делает один новый запрос для первой языковой версии задания 201.
Ответ сохраняется в demo_outputs/. Эта папка отделена от данных для подсчёта.

## Подготовка полной таблицы до защиты

Для Gemini повтори команду сбора. Скрипт продолжит со следующего задания после
последнего сохранённого ответа.

```sh
./run_gemini.sh --free-tier-confirmed
```

Для локальных моделей открой окно Terminal в папке проекта и запусти Ollama.

```sh
./start_ollama_local.sh
```

Оставь сервер открытым. После окончания текущего запуска модели в другом окне,
в той же папке, выполни подготовку и сбор ответов Qwen и Gemma.

```sh
./prepare_local.sh
./run_local.sh
```

Когда все три модели достигнут 830/830, выполни подсчёт заранее и открой таблицу.

```sh
./reproduce.sh
./demo.sh --results
```

Подсчёт читает сохранённые ответы. QA accuracy показывает долю совпавших букв.
chrF2 измеряет символьное совпадение перевода с эталоном. MT gap вычисляется
как chrF2 перевода на казахский минус chrF2 перевода на английский.
В scores.json находятся подробные метрики и интервалы. Holm в holm.json
корректирует три парных сравнения QA. Отдельный sensitivity-анализ исключает
70 совпадающих заданий pilot/test. После исключения остаются 180 пар MT,
180 пар QA и 0 probes.

Пока таблица моделей ожидает расчёта, пункт 4 показывает отдельные baseline.
Always A даёт один и тот же вариант для всех QA. Seeded random выбирает букву
с фиксированным seed. Dictionary MT использует подстановки из словаря.
Эти строки служат точками сравнения для моделей.

При необходимости отдельно выполни NLLB по README.md и ручную разметку по
annotation/HOW_TO_ANNOTATE.md. В итоговой таблице они имеют собственный статус.

## Короткое объяснение на английском

We evaluate Gemini, Qwen and Gemma on Kazakh and English translation and
question answering. The benchmark contains 430 items, which become 830 prompts
per model. The runner stores each response with its model version, prompt ID
and timestamp. The scoring step uses these logs to calculate QA accuracy, chrF2
and paired comparisons. Here I can open one item and trace its score to the
saved response. The progress screen shows which parts of the experiment are ready.
