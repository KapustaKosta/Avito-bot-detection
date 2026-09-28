"""
Фиксирует seed для random, numpy и torch и хранит его в переменной окружения BOT_SEED.
Все остальные скрипты берут отсюда SEED, значение по умолчанию 11.
"""
import os
import random

# seed читается из окружения, чтобы ноутбук задал его один раз в первой ячейке,
# а скрипты, запущенные из ноутбука через "!python", унаследовали окружение и получили то же значение
SEED = int(os.environ.get("BOT_SEED", "11"))


def set_global_seed(seed=SEED):
    """Фиксирует генераторы random, numpy и torch (если он установлен) и записывает seed в окружение."""
    global SEED
    SEED = seed
    # дочерние процессы читают seed из окружения, поэтому он записывается туда же
    os.environ["BOT_SEED"] = str(seed)
    # PYTHONHASHSEED управляет хешированием строк, а значит порядком обхода множеств строк
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)               # стандартный генератор Python
    import numpy as np
    np.random.seed(seed)            # глобальный генератор numpy, его используют sklearn и бутстрэп
    try:
        import torch
        torch.manual_seed(seed)             # инициализация весов сети и перемешивание батчей
        torch.cuda.manual_seed_all(seed)    # то же для всех видеокарт
        # без этих флагов свёртки на GPU могут выбирать разные алгоритмы от запуска к запуску
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass                        # torch нужен только нейросетям, остальные скрипты работают без него
    return seed
