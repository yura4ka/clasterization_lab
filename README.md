# Кластеризація числових даних

PAM (K-Medoids), WaveCluster/Haar-DWT-2D, GMM і MCL.

Основне виконання в `./notebooks/clustering.ipynb`

## Запуск

Перевірене середовище: Python 3.14, локальний `.venv`.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Дані й результати

- Wine (178×13), Seeds (210×7), Dry Bean (13611×16).
- Wine / Dry Bean: бібліотека `ucimlrepo`; Seeds: `kagglehub`.
- Перший запуск потребує мережі; повторний читає `data/raw/`.
- Завантаження Kaggle кешується в `data/raw/kaggle_cache/`, не в домашньому каталозі.
- Синтетика: 7 сценаріїв, детермінована сітка, PCG64 і MT19937.
- `quick`: 21 синтетичний набір, до 360 об'єктів у головному порівнянні.
- `full`: 28 синтетичних наборів, до 600 об'єктів; розширені параметри,
  групові δ-збурення, 3 повтори ресурсних тестів до n=1200 на двох джерелах.
- `data/processed/` містить окремі підкаталоги запусків.
- `artifacts/<UTC>_<mode>/` містить таблиці CSV/Parquet, масиви NPZ,
  розріджені графи, PNG/SVG, конфігурації, EDA, логи, notebook і manifest SHA256.
- Останній запуск: `artifacts/latest.json`.
- Точні прямі залежності: `requirements.txt`; повний `pip freeze` — у середовищі запуску.
