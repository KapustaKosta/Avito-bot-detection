# Прогон графовой нейросети на Kaggle с GPU

1. Datasets, New Dataset, загрузить `kaggle_bundle.zip`, назвать датасет, например `avito-bot-bundle`.
2. Code, New Notebook, File, Import Notebook, выбрать `kaggle_gnn.ipynb`.
3. Add Input, подключить датасет. Settings, Accelerator, выбрать GPU. Интернет не нужен.
4. Run All. Около 5 минут.
5. В панели Output забрать `submission_gnn_stack.csv`, это финальный файл.

В архиве папка `data` с исходными данными, папка `out` с готовыми признаками, парами соседей и результатами
проверки гипотез, модули `seed.py`, `features.py`, `modeling.py`, `propagate.py`, `stack.py`, `gnn_model.py`,
`seq_model.py` и ноутбук `kaggle_gnn.ipynb`.
