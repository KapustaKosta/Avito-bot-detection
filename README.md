# Детекция ботов на Авито

Решение задачи о куках сервисов автоматизированного сбора данных. Весь ход решения в `solution.ipynb`,
код в модулях рядом, данные в `data`, материалы для прогона графовой нейросети на Kaggle в `kaggle`.

Запуск выполнялся на Python 3.12.14, версии пакетов в `requirements.txt`:

```
pip install -r requirements.txt
```

Порядок скриптов тот же, что в ноутбуке:

```
python features.py --data data --out out
python hypotheses.py --features out/features.csv --out out
python compare_models.py --out out
python propagate.py --out out
python stack.py --out out --write
python gnn_model.py --out out --write --device cuda --seeds 11,22,33,44,55,66,77,88,99,100
```

Финальный файл предсказаний: `out/submission_gnn_stack.csv`.
