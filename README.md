# «Белый шум»: психология одной белой линии

Код и данные к визуальному исследованию о рекламе iPod «Silhouettes» (Apple, TBWA\Chiat\Day, 2003–2008).

## Что здесь считается

Ключевой вопрос исследования: за счёт чего работает реклама, в которой нет ни лица героя, ни описания продукта.
Гипотеза: кампания держится на одной белой линии — проводе наушников, который при крошечной площади забирает непропорционально много внимания.

Чтобы проверить это не «на глаз», ключевые кадры роликов пропущены через модель зрительного внимания **DeepGaze IIE** — алгоритм, обученный на записях айтрекинга. Для каждого кадра считаются:

- **площадь белого** — какую долю кадра занимают провод, наушники и плеер;
- **доля внимания** — какую долю предсказанного внимания забирает эта белая деталь;
- **во сколько раз** внимание больше площади (`lift`);
- **пик на белом** — попадает ли точка максимального внимания на белую деталь.

## Результаты

8 роликов, по 4 ключевых кадра — 32 кадра. Сводка по роликам (`out/summary.csv`):

| Ролик | Доля внимания на белом | Во сколько раз больше площади |
|---|---|---|
| 2001, до силуэтов | 14,1% | ×2,1 |
| 2003, Jet | 7,2% | ×8,5 |
| 2004, Daft Punk | 1,7% | ×6,7 |
| 2004, U2 | 0,3% | ×7,9 |
| 2005, Eminem | 0,25% | ×6,7 |
| 2006, Дилан | 1,9% | ×6,3 |
| 2008, Coldplay | 0 (iPod в кадре нет) | — |
| 2022, AirPods | 0,9% | ×24,1 |

Доля внимания — среднее по кадрам ролика. «Во сколько раз» — сумма внимания, делённая на сумму площади по кадрам ролика, чтобы крошечные детали в отдельных кадрах не раздували число.

Вывод исследования: когда белая деталь в кадре, она тянет взгляд всегда — в 6–8 раз сильнее своей площади. Слабеет не стимул: Apple сама вытесняет его из кадра лицами звёзд, реалистичной съёмкой и роликом без продукта, а в 2022 году линия превращается в точку.

## Состав репозитория

| Файл / папка | Что это |
|---|---|
| `frames.py` | шаг 1: кадры-кандидаты (кадр в секунду) и контактные листы для выбора ключевых кадров |
| `measure.py` | шаг 2: замер по ключевым кадрам — маска белого, карта внимания, числа |
| `chart.py` | шаг 3: два итоговых графика |
| `summary.py` | сводка по роликам — числа для таблицы |
| `common.py` | кадры из видео, обрезка чёрных полей, маска белого в OKLab |
| `saliency.py` | модель DeepGaze IIE и запасной метод spectral residual |
| `keyframes.csv` | 32 ключевых кадра: ролик, время, подпись, область поиска |
| `out/results.csv` | числа по каждому кадру |
| `out/summary.csv` | числа по каждому ролику |
| `out/chart_attention.png`, `out/chart_lift.png`, `out/table.png` | графики и таблица из исследования |
| `out/frames/` | ключевые кадры |
| `out/heatmaps/` | карты внимания поверх кадров |
| `out/masks/` | что скрипт посчитал белой деталью (красным) |

Сами ролики в репозиторий не входят: права на них принадлежат Apple. Ссылки — ниже.

## Ролики

Файлы кладутся в папку `videos/` под такими именами (номер задаёт хронологию на графиках):

| Файл | Ролик | Источник |
|---|---|---|
| `01_2001_first_ipod.webm` | Первая реклама iPod, 2001 (Propellerheads — Take California) | https://www.youtube.com/watch?v=mE_bDNaYAr8 |
| `02_2003_jet.webm` | iPod + iTunes: Silhouettes, 2003 (Jet — Are You Gonna Be My Girl) | https://www.youtube.com/watch?v=TaVFCdwT0hk |
| `03_2004_daft_punk.webm` | iPod + iTunes: Silhouettes, 2004 (Daft Punk — Technologic) | https://www.youtube.com/watch?v=JVdFAFe1T2I |
| `04_2004_u2.webm` | iPod + iTunes, 2004 (U2 — Vertigo) | https://www.youtube.com/watch?v=dFWxfAKt8H4 |
| `05_2005_eminem.mp4` | iPod + iTunes, 2005 (Eminem — Lose Yourself) | https://www.youtube.com/watch?v=mYNfcIliDp0 |
| `06_2006_dylan.webm` | iPod + iTunes, 2006 (Bob Dylan — Someday Baby) | https://www.youtube.com/watch?v=oTxiYOsdcKY |
| `07_2008_coldplay.mp4` | iTunes, 2008 (Coldplay — Viva la Vida) | https://www.youtube.com/watch?v=O3mYc1m3lsM |
| `08_2022_airpods.mp4` | AirPods with Spatial Audio + Music for a Sushi Restaurant, 2022 | https://www.instagram.com/reels/CeVw8EbjrSS/ |

Время ключевых кадров в `keyframes.csv` привязано к этим копиям. В другой загрузке того же ролика время может сдвинуться на секунду-другую — сверяйся с контактными листами из `frames.py`.

## Как повторить замер

Нужны Python 3.10–3.13 и ffmpeg (`brew install ffmpeg` на Mac, `sudo apt install ffmpeg` на Linux).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision   # Linux/Windows без видеокарты: добавь --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install git+https://github.com/matthias-k/DeepGaze.git
```

Первый запуск модели скачивает около 1 ГБ весов (GitHub, Bitbucket, download.pytorch.org) в `~/.cache/torch/hub/checkpoints/`. Если загрузка оборвалась, удали недокачанный файл оттуда и запусти снова.

```bash
.venv/bin/python frames.py   --videos videos --out out                          # кадры-кандидаты и контактные листы
.venv/bin/python measure.py  --videos videos --keyframes keyframes.csv --out out # замер
.venv/bin/python chart.py    --results out/results.csv --out out                # графики
.venv/bin/python summary.py  --results out/results.csv --out out/summary.csv    # сводка
```

На Mac с Apple-чипом замер ускоряется флагом `--device mps`. Числа в `out/` получены с настройками по умолчанию.

### Колонка `roi` в `keyframes.csv`

- пусто — белая деталь ищется во всём кадре;
- `none` — продукта в кадре нет (белое там — свет или фон, а не провод): все кадры Coldplay и два кадра Дилана;
- `x0,y0,x1,y1` — прямоугольник в долях кадра. Использован один раз: кадр Дилана 00:13.5, чтобы исключить кусок светлого фона между рукой и телом.

### Настройки `measure.py`

| Параметр | По умолчанию | Зачем |
|---|---|---|
| `--white-l`, `--white-c` | `0.80`, `0.08` | пороги белого в OKLab: светлота и цветность |
| `--max-component` | `0.15` | белое пятно больше 15% кадра — фон, а не деталь |
| `--max-border-component` | `0.01` | белое пятно у края кадра больше 1% — тоже фон |
| `--centerbias` | `mit1003` | поправка «люди чаще смотрят в центр», как в айтрекинге |
| `--width` | `1024` | ширина кадра для замера |
| `--model spectral` | `deepgaze` | запасной метод без нейросети (только контраст, лица не видит) |

Чёрные поля по краям (ролик 4:3 в кадре 16:9) обрезаются автоматически.

### Технические заметки

- Свежий пакет `deepgaze_pytorch` при импорте подгружает модель MSDB, которой нужен пакет `clip`. Здесь используется только DeepGaze IIE, поэтому `saliency.py` подставляет пустой модуль `clip`; пакет `einops` ставится из `requirements.txt`.
- Для Apple MPS модель переводится в float32 (`.float()`): MPS не поддерживает float64.
- Кадры-титры «iPod» и «Mac or PC» (Jet, 00:22 и 00:26), которые есть в исследовании, в замер не входят: это кадры-кандидаты из `out/candidates/02_2003_jet/` (`0023.jpg`, `0027.jpg`), их создаёт `frames.py`.

## Ограничения

- Модель предсказывает взгляд, а не измеряет его, и обучена на статичных изображениях, а не на видео. Это первый шаг, а не замена айтрекингу на людях.
- Ролики 2001–2008 годов сохранились только в телевизионном качестве своего времени (320–720 px) и в сжатых копиях; в общих планах тонкий провод иногда теряется, и цифры по ним занижены.
- Кадры отобраны вручную, хотя и по одному принципу: общий план, крупный план белой детали и, если есть, кадр с лицом.
- В маску попадает любое белое (например, микрофон Eminem) — это видно в `out/masks/`.

## Источники

- Linardos A., Kümmerer M., Press O., Bethge M. DeepGaze IIE: Calibrated Prediction in and Out-of-Domain for State-of-the-Art Saliency Modeling // ICCV. — 2021. — P. 12919–12928. Код модели: https://github.com/matthias-k/DeepGaze
- Ottosson B. A perceptual color space for image processing (OKLab). — 2020. https://bottosson.github.io/posts/oklab/

## Лицензия

Код — MIT (см. `LICENSE`). Кадры из роликов в `out/frames/`, `out/heatmaps/` и `out/masks/` принадлежат правообладателям роликов и приведены в учебных и исследовательских целях; под лицензию MIT они не подпадают.
