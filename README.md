**Repository Purpose**

Этот репозиторий содержит скрипт обучения KMeans на спарке и все необходимые зависимости и инструкции для запуска.

**Проекты**: `kmeans.py` (KMeans на OpenFoodFacts CSV), `wordcount.py` (простая обработка текста),

**Quick Start**

1. Установить зависимости и подготовить данные:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Скачать en.openfoodfacts.org.products.csv с https://world.openfoodfacts.org/data


2. Запуск WordCount (локально через Spark):

```bash
spark-submit wordcount.py
```

3. Запуск KMeans (рекомендуется через `spark-submit`):

```bash
spark-submit kmeans.py
```

