# Конструктор эксперимента

Эта страница проведёт вас через сборку полного ASR-эксперимента шаг за шагом --
от выбора данных до готового кода, который можно скопировать и запустить.

Если вы уже знаете, что вам нужно, пропустите пояснения и переходите сразу
к [интерактивному сборщику](#interactive-builder) внизу страницы.

---

## Шаг 1. На каких данных будет замер?

Прежде чем выбирать модели и метрики, нужно понять, **на чём** вы будете измерять качество распознавания.

### Готовые датасеты

plantain2asr поставляется с загрузчиками для нескольких русскоязычных корпусов.
Каждый загрузчик автоматически парсит структуру корпуса и отдаёт одинаковый интерфейс `AudioSample`.

#### Golos

Открытый корпус Сбера. ~1 200 часов русской речи. Два подмножества:

- **crowd** -- запись через краудсорсинг (чистая, рядовые дикторы)
- **farfield** -- запись с дальних микрофонов (шумнее, реалистичнее)

| | |
|---|---|
| Размер | ~1 200 ч |
| Формат аудио | WAV / OGG |
| Скачать | [github.com/sberdevices/golos](https://github.com/sberdevices/golos) |
| Загрузчик | `GolosDataset("data/golos")` |
| Автозагрузка | да (`auto_download=True`) |

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

#### DaGRuS

Корпус разговорной русской речи с детальной разметкой: смех, шум, неразборчивые слова, филлеры.

| | |
|---|---|
| Размер | ~60 ч |
| Особенности | Разговорная речь, разметка событий |
| Скачать | по запросу у авторов корпуса |
| Загрузчик | `DagrusDataset("data/dagrus")` |

!!! tip "Нормализация для DaGRuS"
    Используйте `DagrusNormalizer()` -- он умеет убирать специфическую разметку корпуса
    (`[laugh]`, `[noise]`, `{word*}`) и нормализовать разговорные формы ("щас" -> "сейчас").

```python
from plantain2asr import DagrusDataset, DagrusNormalizer

ds = DagrusDataset("data/dagrus")
norm = ds >> DagrusNormalizer()
```

#### RuDevices

Корпус записей с различных устройств (ноутбуки, телефоны, умные колонки).

| | |
|---|---|
| Загрузчик | `RuDevicesDataset("data/rudevices")` |
| Особенности | Разные устройства, условия записи |

```python
from plantain2asr import RuDevicesDataset

ds = RuDevicesDataset("data/rudevices")
```

### Свой датасет

Если ваши данные не покрываются готовыми загрузчиками, есть два пути.

#### Путь 1: NeMo-формат JSONL

Если у вас есть аудиофайлы и JSONL-манифест, используйте `NeMoDataset`:

```json
{"audio_filepath": "audio/001.wav", "text": "привет мир", "duration": 2.1}
{"audio_filepath": "audio/002.wav", "text": "как дела", "duration": 1.8}
```

```python
from plantain2asr import NeMoDataset

ds = NeMoDataset(root_dir="data/my_corpus", manifest_path="data/my_corpus/manifest.jsonl")
```

#### Путь 2: свой класс-загрузчик

Наследуйтесь от `BaseASRDataset` и верните список `AudioSample`:

```python
from plantain2asr.dataloaders.base import BaseASRDataset
from plantain2asr.dataloaders.types import AudioSample

class MyDataset(BaseASRDataset):
    def __init__(self, root_dir):
        super().__init__()
        self.name = "my-dataset"
        self._samples = [
            AudioSample(id="s1", audio_path=f"{root_dir}/001.wav", text="эталонный текст"),
        ]
```

Подробнее: [Расширение -> Своя модель](extending/custom_model.md)

---

## Шаг 2. Какие метрики вам нужны?

Метрики показывают, **насколько хорошо** модель распознала речь.

### Основные метрики

| Метрика | Что измеряет | Когда нужна |
|---|---|---|
| **WER** (Word Error Rate) | Доля ошибочных слов. Считает вставки, удаления и замены на уровне слов. | Универсальная основная метрика. Используйте всегда. |
| **CER** (Character Error Rate) | То же, но на уровне символов. | Когда важна точность написания, а не только слов. |
| **MER** (Match Error Rate) | Нормализованный вариант WER с учётом длины обеих строк. | Для более устойчивой оценки на коротких фразах. |
| **Accuracy** | `1 - MER`. Показывает долю правильного распознавания. | Когда нужна интуитивно понятная цифра "процент верных". |

### Дополнительные метрики

| Метрика | Что измеряет |
|---|---|
| **WIL** | Word Information Lost -- потеря информации на уровне слов |
| **WIP** | Word Information Preserved -- сохранённая информация |
| **IDR** | Insertion/Deletion Ratio -- соотношение вставок и удалений |
| **LengthRatio** | Отношение длины гипотезы к длине эталона |
| **BERTScore** | Семантическое сходство через BERT-эмбеддинги (нужен extra `analysis`) |
| **POSAnalysis** | Анализ ошибок по частям речи (нужен extra `analysis`) |

### Что выбрать?

!!! note "Рекомендация"
    Для первого замера используйте `Metrics.composite()` -- он посчитает WER, CER, MER,
    WIL, WIP, Accuracy, IDR и LengthRatio за один проход.

```python
from plantain2asr import Metrics

norm >> Metrics.composite()
```

Если нужна одна метрика:

```python
norm >> Metrics.WER()
```

---

## Шаг 3. Какие модели сравнить?

plantain2asr поддерживает несколько семейств ASR-моделей. Все они используются одинаково:
`dataset >> Models.XXX()`.

### Локальные модели

| Модель | Что это | Устройство | Extra для pip | Когда выбрать |
|---|---|---|---|---|
| **GigaAM v3** | Крупная модель от Сбера, e2e-RNNT архитектура. Лучшее качество на русском. | CUDA / MPS / CPU | `gigaam` | Когда важно качество и есть GPU |
| **GigaAM v2** | Предыдущее поколение GigaAM. | CUDA / MPS / CPU | `gigaam` | Для сравнения с v3 |
| **Whisper** | Модель OpenAI, large-v3. Сильный мультиязычный baseline. | CUDA / MPS / CPU | `whisper` | Универсальный baseline |
| **T-One** | Модель T-Bank на ONNX Runtime. Быстрый инференс. | CUDA / CPU | `tone` + source archive T-One | Когда важна скорость |
| **Vosk** | Лёгкая offline-модель на Kaldi. Работает только на CPU. | CPU | `vosk` | Когда нет GPU и нужен offline |
| **Canary** | NVIDIA NeMo Canary. Тяжёлая, требует GPU. | CUDA | `canary` | Исследовательские сравнения |

### Облачные модели

| Модель | Что это | Extra | Когда выбрать |
|---|---|---|---|
| **SaluteSpeech** | Облачный API Сбера. | нет | Когда нужно облачное распознавание |

### Установка

Каждая модель требует свой набор зависимостей. Ставьте только то, что нужно:

```bash
pip install plantain2asr[gigaam]
pip install plantain2asr[whisper]
pip install plantain2asr[vosk]
pip install plantain2asr[tone]
pip install "tone @ https://github.com/voicekit-team/T-one/archive/3c5b6c015038173840e62cea99e10cdb1c759116.tar.gz"
```

Или сразу весь CPU/GPU-стек:

```bash
pip install plantain2asr[asr-cpu]
pip install plantain2asr[asr-gpu]
```

### Пример запуска

```python
from plantain2asr import Models

ds >> Models.GigaAM_v3()
ds >> Models.Whisper()
ds >> Models.Vosk(model_path="path/to/vosk-model")
```

Результаты кешируются: повторный запуск не пересчитывает уже распознанные семплы.

---

## Шаг 4. Нормализация текста

Перед подсчётом метрик нужно привести эталон и гипотезу к единому виду: убрать пунктуацию,
привести регистр, обработать специфику корпуса.

| Нормализатор | Что делает | Когда использовать |
|---|---|---|
| `SimpleNormalizer()` | Lowercase, убирает пунктуацию, `ё` -> `е`, схлопывает пробелы | Для большинства корпусов |
| `DagrusNormalizer()` | Всё что SimpleNormalizer + убирает разметку DaGRuS + нормализует разговорные формы | Для корпуса DaGRuS |
| Без нормализации | Метрики считаются на исходном тексте | Только если тексты уже нормализованы |

```python
from plantain2asr import SimpleNormalizer

norm = ds >> SimpleNormalizer()
```

---

## Шаг 5. Собираем цепочку `>>`

Теперь, когда вы выбрали данные, модели, нормализатор и метрики, соберите их
в пайплайн через оператор `>>`:

```python
from plantain2asr import GolosDataset, Models, SimpleNormalizer, Metrics

ds = GolosDataset("data/golos")

# шаг 1: прогнать модели
ds >> Models.GigaAM_v3()
ds >> Models.Whisper()

# шаг 2: нормализовать
norm = ds >> SimpleNormalizer()

# шаг 3: посчитать метрики
norm >> Metrics.composite()

# шаг 4: посмотреть результаты
df = norm.to_pandas()
print(df.groupby("model")[["WER", "CER"]].mean().sort_values("WER"))
```

Каждый `>>` создаёт новый слой результатов поверх датасета.
Можно ветвить (`.filter()`), брать подвыборки (`.take(n)`) и рекомбинировать.

### Обёртка `Experiment`

Если не нужен ручной контроль, `Experiment` оборачивает те же `>>` шаги:

```python
from plantain2asr import Experiment, GolosDataset, Models, SimpleNormalizer

experiment = Experiment(
    dataset=GolosDataset("data/golos"),
    models=[Models.GigaAM_v3(), Models.Whisper()],
    normalizer=SimpleNormalizer(),
)

experiment.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])
```

| Метод | Что делает |
|---|---|
| `compare_on_corpus()` | Сравнение моделей с таблицей метрик |
| `prepare_thesis_tables()` | CSV-таблицы для диссертации |
| `export_appendix_bundle()` | Полный пакет: таблицы + отчёт + бенчмарк |
| `benchmark_models()` | Замеры latency, throughput, RTF |

---

## Интерактивный сборщик {#interactive-builder}

Выберите ваши компоненты, и сборщик покажет готовый код, команду установки
и список артефактов.

<div markdown="0">

<style>
.p2a-builder{font-family:inherit;color:var(--md-typeset-color, #333)}
.p2a-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
@media(max-width:768px){.p2a-row{grid-template-columns:1fr}}
.p2a-group{margin-bottom:24px}
.p2a-group-title{font-weight:700;font-size:.95rem;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.p2a-group-title .step-num{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;background:var(--md-primary-fg-color, #4caf50);color:#fff;font-size:.8rem;font-weight:800;flex-shrink:0}
.p2a-choices{display:flex;flex-wrap:wrap;gap:8px}
.p2a-choice{border:2px solid var(--md-default-fg-color--lightest, #ddd);background:var(--md-default-bg-color, #fff);color:var(--md-default-fg-color, #333);padding:8px 14px;border-radius:10px;cursor:pointer;font-size:.85rem;transition:.15s;line-height:1.4}
.p2a-choice:hover{border-color:var(--md-primary-fg-color, #4caf50)}
.p2a-choice.active{border-color:var(--md-primary-fg-color, #4caf50);background:var(--md-primary-fg-color--light, #e8f5e9);font-weight:600}
.p2a-choice small{display:block;font-weight:400;opacity:.7;font-size:.78rem}
.p2a-select{width:100%;padding:10px 12px;border-radius:10px;border:2px solid var(--md-default-fg-color--lightest, #ddd);background:var(--md-default-bg-color, #fff);color:var(--md-default-fg-color, #333);font-size:.88rem}
.p2a-output{margin-top:24px;padding:20px;border-radius:12px;border:2px solid var(--md-primary-fg-color, #4caf50);background:var(--md-code-bg-color, #f5f5f5)}
.p2a-output h3{margin:0 0 4px;font-size:.95rem}
.p2a-output pre{margin:12px 0 0;padding:14px;border-radius:8px;background:var(--md-code-bg-color, #263238);color:var(--md-code-fg-color, #eee);overflow-x:auto;font-size:.82rem;line-height:1.6;white-space:pre-wrap}
.p2a-install{margin-top:16px;padding:12px 14px;border-radius:10px;background:var(--md-admonition-bg-color, #e8f5e9);font-size:.85rem}
.p2a-install code{font-weight:700}
.p2a-artifacts{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px}
.p2a-artifact{padding:6px 12px;border-radius:8px;background:var(--md-primary-fg-color--light, #e8f5e9);border:1px solid var(--md-primary-fg-color, #4caf50);font-size:.8rem}
.p2a-copy-btn{border:none;background:var(--md-primary-fg-color, #4caf50);color:#fff;padding:6px 14px;border-radius:8px;cursor:pointer;font-size:.8rem;float:right}
.p2a-toggles{display:grid;grid-template-columns:1fr 1fr;gap:8px}
@media(max-width:768px){.p2a-toggles{grid-template-columns:1fr}}
.p2a-toggle{display:flex;align-items:flex-start;gap:8px;padding:8px 12px;border-radius:10px;border:1px solid var(--md-default-fg-color--lightest, #ddd);font-size:.83rem;cursor:pointer}
.p2a-toggle input{margin-top:3px}
.p2a-toggle strong{display:block;font-size:.83rem}
.p2a-toggle small{opacity:.7}
</style>

<div class="p2a-builder" id="p2a-builder">

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">1</span> Какой результат нужен?</div>
<div class="p2a-choices" id="p2a-preset"></div>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">2</span> Какой датасет?</div>
<select class="p2a-select" id="p2a-dataset"></select>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">3</span> Какие модели?</div>
<div class="p2a-choices" id="p2a-models"></div>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">4</span> Нормализатор</div>
<select class="p2a-select" id="p2a-norm"></select>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">5</span> Метрики</div>
<div class="p2a-choices" id="p2a-metrics"></div>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">6</span> Дополнительно</div>
<div class="p2a-toggles">
  <label class="p2a-toggle"><input type="checkbox" id="p2a-art-report"><span><strong>HTML-отчёт</strong><small>self-contained статический файл</small></span></label>
  <label class="p2a-toggle"><input type="checkbox" id="p2a-art-browser"><span><strong>Браузерный отчёт</strong><small>интерактивный, с аудио</small></span></label>
  <label class="p2a-toggle"><input type="checkbox" id="p2a-art-bench"><span><strong>Benchmark</strong><small>latency / throughput / RTF</small></span></label>
  <label class="p2a-toggle"><input type="checkbox" id="p2a-art-pandas"><span><strong>Pandas DataFrame</strong><small>для своего анализа</small></span></label>
</div>
</div>

<div class="p2a-output" id="p2a-output">
  <h3>Команда установки</h3>
  <div class="p2a-install" id="p2a-install"></div>
  <h3 style="margin-top:16px">Готовый код <button class="p2a-copy-btn" id="p2a-copy">Скопировать</button></h3>
  <pre><code id="p2a-code"></code></pre>
  <h3 style="margin-top:16px">Артефакты</h3>
  <div class="p2a-artifacts" id="p2a-artifacts"></div>
</div>

</div>

<script>
(function(){
const PRESETS=[
{id:"pipeline",title:"Пайплайн >>",hint:"dataset >> model >> norm >> metric",route:"dataset >> model >> norm >> metric"},
{id:"compare",title:"Experiment: сравнить",hint:"обёртка для summary + leaderboard",route:"Experiment.compare_on_corpus()"},
{id:"thesis",title:"Experiment: диссертация",hint:"CSV для results / leaderboard / errors",route:"Experiment.prepare_thesis_tables()"},
{id:"bundle",title:"Experiment: bundle",hint:"таблицы + отчёт + бенчмарк",route:"Experiment.export_appendix_bundle()"}
];
const DATASETS=[
{id:"golos",label:"GolosDataset",code:'GolosDataset("data/golos")',dir:'"data/golos"'},
{id:"dagrus",label:"DagrusDataset",code:'DagrusDataset("data/dagrus")',dir:'"data/dagrus"'},
{id:"rudevices",label:"RuDevicesDataset",code:'RuDevicesDataset("data/rudevices")',dir:'"data/rudevices"'},
{id:"nemo",label:"NeMoDataset (свой JSONL)",code:'NeMoDataset(root_dir="data/custom", manifest_path="data/custom/manifest.jsonl")',dir:'"data/custom"'}
];
const MODELS=[
{id:"gigaam_v3",label:"GigaAM v3",code:"Models.GigaAM_v3()",extra:"gigaam"},
{id:"gigaam_v2",label:"GigaAM v2",code:"Models.GigaAM_v2()",extra:"gigaam"},
{id:"whisper",label:"Whisper",code:"Models.Whisper()",extra:"whisper"},
{id:"vosk",label:"Vosk",code:"Models.Vosk()",extra:"vosk"},
{id:"tone",label:"T-One",code:"Models.Tone()",extra:"tone"},
{id:"canary",label:"Canary",code:"Models.Canary()",extra:"canary"},
{id:"salute",label:"SaluteSpeech",code:"Models.SaluteSpeech()",extra:null}
];
const NORMS=[
{id:"none",label:"Без нормализации",code:null},
{id:"simple",label:"SimpleNormalizer",code:"SimpleNormalizer()"},
{id:"dagrus",label:"DagrusNormalizer",code:"DagrusNormalizer()"},
{id:"golos",label:"GolosNormalizer",code:"GolosNormalizer()"}
];
const METRICS=[
{id:"composite",label:"Composite (все базовые)",code:"Metrics.composite()"},
{id:"wer",label:"WER",code:"Metrics.WER()"},
{id:"cer",label:"CER",code:"Metrics.CER()"},
{id:"accuracy",label:"Accuracy",code:"Metrics.Accuracy()"}
];

const S={preset:"pipeline",dataset:"golos",models:new Set(["gigaam_v3","whisper"]),norm:"simple",metrics:new Set(["composite"]),art:{"report":false,"browser":false,"bench":false,"pandas":true}};

function el(id){return document.getElementById(id)}

function render(){
  el("p2a-preset").innerHTML=PRESETS.map(p=>'<button class="p2a-choice'+(S.preset===p.id?' active':'')+'" data-v="'+p.id+'"><strong>'+p.title+'</strong><small>'+p.hint+'</small></button>').join("");
  el("p2a-preset").querySelectorAll("[data-v]").forEach(b=>b.onclick=()=>{S.preset=b.dataset.v;render()});

  el("p2a-dataset").innerHTML=DATASETS.map(d=>'<option value="'+d.id+'">'+d.label+'</option>').join("");
  el("p2a-dataset").value=S.dataset;
  el("p2a-dataset").onchange=function(){S.dataset=this.value;if(S.dataset==="dagrus"&&S.norm==="simple")S.norm="dagrus";render()};

  el("p2a-models").innerHTML=MODELS.map(m=>'<button class="p2a-choice'+(S.models.has(m.id)?' active':'')+'" data-v="'+m.id+'"><strong>'+m.label+'</strong></button>').join("");
  el("p2a-models").querySelectorAll("[data-v]").forEach(b=>b.onclick=()=>{var id=b.dataset.v;if(S.models.has(id)){if(S.models.size>1)S.models.delete(id)}else S.models.add(id);render()});

  el("p2a-norm").innerHTML=NORMS.map(n=>'<option value="'+n.id+'">'+n.label+'</option>').join("");
  el("p2a-norm").value=S.norm;
  el("p2a-norm").onchange=function(){S.norm=this.value;render()};

  el("p2a-metrics").innerHTML=METRICS.map(m=>'<button class="p2a-choice'+(S.metrics.has(m.id)?' active':'')+'" data-v="'+m.id+'"><strong>'+m.label+'</strong></button>').join("");
  el("p2a-metrics").querySelectorAll("[data-v]").forEach(b=>b.onclick=()=>{var id=b.dataset.v;if(S.metrics.has(id)){if(S.metrics.size>1)S.metrics.delete(id)}else S.metrics.add(id);render()});

  ["report","browser","bench","pandas"].forEach(k=>{var cb=el("p2a-art-"+k);cb.checked=S.art[k];cb.onchange=function(){S.art[k]=this.checked;render()}});

  var ds=DATASETS.find(d=>d.id===S.dataset);
  var ms=MODELS.filter(m=>S.models.has(m.id));
  var nm=NORMS.find(n=>n.id===S.norm);
  var mt=METRICS.filter(m=>S.metrics.has(m.id));
  var extras=[...new Set(ms.map(m=>m.extra).filter(Boolean))];
  var needsTone=ms.some(m=>m.id==="tone");
  var pip=extras.length?"pip install plantain2asr["+extras.join(",")+"]":"pip install plantain2asr";
  if(needsTone){
    pip += '\n' + 'pip install "tone @ https://github.com/voicekit-team/T-one/archive/3c5b6c015038173840e62cea99e10cdb1c759116.tar.gz"';
  }
  el("p2a-install").innerHTML="<code>"+pip+"</code>";

  var code="";
  if(S.preset==="pipeline"){
    var imps=new Set(["Models","Metrics"]);
    imps.add(ds.label.split(" ")[0]);
    if(nm.code)imps.add(nm.label);
    code="from plantain2asr import "+[...imps].join(", ")+"\n\nds = "+ds.code+"\n\n";
    ms.forEach(m=>{code+="ds >> "+m.code+"\n"});
    code+="\n";
    if(nm.code)code+="norm = ds >> "+nm.code+"\n";else code+="norm = ds\n";
    code+="norm >> "+(mt.length===1?mt[0].code:"Metrics.composite()")+"\n";
    if(S.art.pandas)code+="\ndf = norm.to_pandas()\nprint(df.groupby(\"model\").mean(numeric_only=True))\n";
    if(S.art.browser)code+="\nfrom plantain2asr import ReportServer\nReportServer(norm, audio_dir="+ds.dir+").serve()\n";
  }else{
    var imps=new Set(["Experiment","Models"]);
    imps.add(ds.label.split(" ")[0]);
    if(nm.code)imps.add(nm.label);
    imps.add("Metrics");
    code="from plantain2asr import "+[...imps].join(", ")+"\n\n";
    code+="ds = "+ds.code+"\n\n";
    code+="study = Experiment(\n    dataset=ds,\n    models=[\n";
    ms.forEach(m=>{code+="        "+m.code+",\n"});
    code+="    ],\n    normalizer="+(nm.code||"None")+",\n    metrics=[\n";
    mt.forEach(m=>{code+="        "+m.code+",\n"});
    code+="    ],\n)\n\n";
    if(S.preset==="compare"){
      code+='report = study.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])\nprint(report["leaderboard"])\n';
    }else if(S.preset==="thesis"){
      code+='tables = study.prepare_thesis_tables(\n    output_dir="artifacts/thesis",\n    metrics=["WER", "CER", "Accuracy"],\n)\nprint(tables["leaderboard_csv"])\n';
    }else{
      code+='bundle = study.export_appendix_bundle(\n    output_dir="artifacts/appendix",\n    include_benchmark='+( S.art.bench?"True":"False")+',\n    include_static_report='+(S.art.report?"True":"False")+',\n)\n';
    }
  }
  el("p2a-code").textContent=code;

  var arts=[];
  if(S.preset==="compare")arts.push("comparison","leaderboard","summary");
  if(S.preset==="thesis")arts.push("results.csv","summary.csv","leaderboard.csv","error_cases.csv");
  if(S.preset==="bundle")arts.push("results.csv","summary.csv","leaderboard.csv");
  if(S.art.report)arts.push("report.html");
  if(S.art.browser)arts.push("browser report");
  if(S.art.bench)arts.push("benchmark.csv");
  if(S.art.pandas)arts.push("pandas DataFrame");
  el("p2a-artifacts").innerHTML=arts.map(a=>'<span class="p2a-artifact">'+a+'</span>').join("");
}

el("p2a-copy").onclick=async function(){
  try{await navigator.clipboard.writeText(el("p2a-code").textContent);this.textContent="Готово!";setTimeout(()=>{this.textContent="Скопировать"},1200)}catch(e){this.textContent="Ошибка"}
};

render();
})();
</script>

</div>

---

## Что дальше?

- [Быстрый старт](quickstart.md) -- канонический рабочий сценарий от начала до конца
- [Справочник API -> Датасеты](api/dataloaders.md) -- полная документация всех методов датасета
- [Справочник API -> Метрики](api/metrics.md) -- все доступные метрики
- [Расширение](extending/index.md) -- как добавить свой нормализатор, модель или метрику
