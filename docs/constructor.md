# Experiment Constructor

This page walks you through assembling a complete ASR experiment step by step --
from choosing your data to ready-to-run code you can copy and execute.

If you already know what you need, skip the explanations and jump straight to
the [interactive builder](#interactive-builder) at the bottom.

---

## Step 1. What data will you measure on?

Before picking models and metrics, you need to decide **what** data you will evaluate on.

### Built-in datasets

plantain2asr ships with loaders for several Russian speech corpora.
Each loader parses the corpus structure automatically and provides a uniform `AudioSample` interface.

#### Golos

An open-source corpus by Sber. ~1 200 hours of Russian speech. Two subsets:

- **crowd** -- crowdsourced recordings (clean, diverse speakers)
- **farfield** -- far-field microphone recordings (noisier, more realistic)

| | |
|---|---|
| Size | ~1 200 h |
| Audio format | WAV / OGG |
| Download | [github.com/sberdevices/golos](https://github.com/sberdevices/golos) |
| Loader | `GolosDataset("data/golos")` |
| Auto-download | yes (`auto_download=True`) |

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

#### DaGRuS

A conversational Russian speech corpus with detailed annotations: laughter, noise, unclear words, fillers.

| | |
|---|---|
| Size | ~60 h |
| Key feature | Conversational speech, event annotations |
| Download | available on request from corpus authors |
| Loader | `DagrusDataset("data/dagrus")` |

!!! tip "Normalization for DaGRuS"
    Use `DagrusNormalizer()` -- it knows how to strip corpus-specific annotations
    (`[laugh]`, `[noise]`, `{word*}`) and normalize colloquial forms.

```python
from plantain2asr import DagrusDataset, DagrusNormalizer

ds = DagrusDataset("data/dagrus")
norm = ds >> DagrusNormalizer()
```

#### RuDevices

A corpus of recordings from various devices (laptops, phones, smart speakers).

| | |
|---|---|
| Loader | `RuDevicesDataset("data/rudevices")` |
| Key feature | Different devices and recording conditions |

```python
from plantain2asr import RuDevicesDataset

ds = RuDevicesDataset("data/rudevices")
```

### Using your own data

If your data is not covered by the built-in loaders, there are two paths.

#### Path 1: NeMo-format JSONL

If you have audio files and a JSONL manifest, use `NeMoDataset`:

```json
{"audio_filepath": "audio/001.wav", "text": "hello world", "duration": 2.1}
{"audio_filepath": "audio/002.wav", "text": "how are you", "duration": 1.8}
```

```python
from plantain2asr import NeMoDataset

ds = NeMoDataset(root_dir="data/my_corpus", manifest_path="data/my_corpus/manifest.jsonl")
```

#### Path 2: custom loader class

Subclass `BaseASRDataset` and return a list of `AudioSample`:

```python
from plantain2asr.dataloaders.base import BaseASRDataset
from plantain2asr.dataloaders.types import AudioSample

class MyDataset(BaseASRDataset):
    def __init__(self, root_dir):
        super().__init__()
        self.name = "my-dataset"
        self._samples = [
            AudioSample(id="s1", audio_path=f"{root_dir}/001.wav", text="reference text"),
        ]
```

More details: [Extending -> Custom Model](extending/custom_model.md)

---

## Step 2. Which metrics do you need?

Metrics show **how well** a model recognized speech.

### Core metrics

| Metric | What it measures | When to use |
|---|---|---|
| **WER** (Word Error Rate) | Fraction of erroneous words. Counts insertions, deletions, and substitutions at the word level. | Universal primary metric. Always include. |
| **CER** (Character Error Rate) | Same idea, but at the character level. | When spelling accuracy matters, not just words. |
| **MER** (Match Error Rate) | Normalized variant of WER accounting for both string lengths. | More stable on short utterances. |
| **Accuracy** | `1 - MER`. The fraction of correctly recognized content. | When you want an intuitive "percent correct" number. |

### Additional metrics

| Metric | What it measures |
|---|---|
| **WIL** | Word Information Lost |
| **WIP** | Word Information Preserved |
| **IDR** | Insertion / Deletion Ratio |
| **LengthRatio** | Hypothesis length divided by reference length |
| **BERTScore** | Semantic similarity via BERT embeddings (requires `analysis` extra) |
| **POSAnalysis** | POS-tag error analysis (requires `analysis` extra) |

### What should I choose?

!!! note "Recommendation"
    For a first evaluation, use `Metrics.composite()` -- it computes WER, CER, MER,
    WIL, WIP, Accuracy, IDR, and LengthRatio in a single pass.

```python
from plantain2asr import Metrics

norm >> Metrics.composite()
```

If you only need one metric:

```python
norm >> Metrics.WER()
```

---

## Step 3. Which models to compare?

plantain2asr supports several ASR model families. They all share the same interface:
`dataset >> Models.XXX()`.

### Local models

| Model | Description | Device | pip extra | When to choose |
|---|---|---|---|---|
| **GigaAM v3** | Large Sber model, e2e-RNNT architecture. Best Russian quality. | CUDA / MPS / CPU | `gigaam` | When quality matters and you have a GPU |
| **GigaAM v2** | Previous GigaAM generation. | CUDA / MPS / CPU | `gigaam` | For comparison with v3 |
| **Whisper** | OpenAI model, large-v3. Strong multilingual baseline. | CUDA / MPS / CPU | `whisper` | Universal baseline |
| **T-One** | T-Bank model on ONNX Runtime. Fast inference. | CUDA / CPU | `tone` | When speed matters |
| **Vosk** | Lightweight offline model on Kaldi. CPU only. | CPU | `vosk` | No GPU, need offline |
| **Canary** | NVIDIA NeMo Canary. Heavy, requires GPU. | CUDA | `canary` | Research comparisons |

### Cloud models

| Model | Description | Extra | When to choose |
|---|---|---|---|
| **SaluteSpeech** | Sber cloud API. | none | Cloud-based recognition |

### Installation

Each model requires its own set of dependencies. Install only what you need:

```bash
pip install plantain2asr[gigaam]
pip install plantain2asr[whisper]
pip install plantain2asr[vosk]
```

Or the full CPU/GPU stack at once:

```bash
pip install plantain2asr[asr-cpu]
pip install plantain2asr[asr-gpu]
```

### Running models

```python
from plantain2asr import Models

ds >> Models.GigaAM_v3()
ds >> Models.Whisper()
ds >> Models.Vosk(model_path="path/to/vosk-model")
```

Results are cached: re-running skips already processed samples.

---

## Step 4. Text normalization

Before computing metrics, you need to bring references and hypotheses to a common form:
remove punctuation, normalize case, handle corpus-specific markup.

| Normalizer | What it does | When to use |
|---|---|---|
| `SimpleNormalizer()` | Lowercase, strip punctuation, `ё` -> `е`, collapse whitespace | Most corpora |
| `DagrusNormalizer()` | Everything SimpleNormalizer does + strips DaGRuS markup + normalizes colloquial forms | DaGRuS corpus |
| No normalization | Metrics are computed on raw text | Only if texts are already normalized |

```python
from plantain2asr import SimpleNormalizer

norm = ds >> SimpleNormalizer()
```

---

## Step 5. Putting it all together

Now that you have chosen data, models, normalizer, and metrics, there are two ways
to run the experiment.

### Option 1: `Experiment` (recommended)

A ready-made orchestrator that runs the full chain for you:

```python
from plantain2asr import Experiment, GolosDataset, Models, SimpleNormalizer

experiment = Experiment(
    dataset=GolosDataset("data/golos"),
    models=[Models.GigaAM_v3(), Models.Whisper()],
    normalizer=SimpleNormalizer(),
)

comparison = experiment.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])
print(comparison["leaderboard"])
```

What else `Experiment` can do:

| Method | What it does |
|---|---|
| `compare_on_corpus()` | Model comparison with metric table |
| `leaderboard()` | Model ranking by a chosen metric |
| `prepare_thesis_tables()` | CSV tables for thesis/paper |
| `export_appendix_bundle()` | Full package: tables + report + benchmark |
| `benchmark_models()` | Latency, throughput, RTF measurements |
| `save_report_html()` | Static HTML report |

### Option 2: `>>` pipeline (full control)

```python
from plantain2asr import GolosDataset, Models, SimpleNormalizer, Metrics

ds = GolosDataset("data/golos")

ds >> Models.GigaAM_v3()
ds >> Models.Whisper()

norm = ds >> SimpleNormalizer()
norm >> Metrics.composite()

df = norm.to_pandas()
print(df.groupby("model")[["WER", "CER"]].mean().sort_values("WER"))
```

---

## Interactive Builder {#interactive-builder}

Pick your components below, and the builder will show you ready-to-use code,
the install command, and a list of output artifacts.

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
<div class="p2a-group-title"><span class="step-num">1</span> What result do you need?</div>
<div class="p2a-choices" id="p2a-preset"></div>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">2</span> Which dataset?</div>
<select class="p2a-select" id="p2a-dataset"></select>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">3</span> Which models?</div>
<div class="p2a-choices" id="p2a-models"></div>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">4</span> Normalizer</div>
<select class="p2a-select" id="p2a-norm"></select>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">5</span> Metrics</div>
<div class="p2a-choices" id="p2a-metrics"></div>
</div>

<div class="p2a-group">
<div class="p2a-group-title"><span class="step-num">6</span> Additional outputs</div>
<div class="p2a-toggles">
  <label class="p2a-toggle"><input type="checkbox" id="p2a-art-report"><span><strong>HTML report</strong><small>self-contained static file</small></span></label>
  <label class="p2a-toggle"><input type="checkbox" id="p2a-art-browser"><span><strong>Browser report</strong><small>interactive, with audio</small></span></label>
  <label class="p2a-toggle"><input type="checkbox" id="p2a-art-bench"><span><strong>Benchmark</strong><small>latency / throughput / RTF</small></span></label>
  <label class="p2a-toggle"><input type="checkbox" id="p2a-art-pandas"><span><strong>Pandas DataFrame</strong><small>for custom analysis</small></span></label>
</div>
</div>

<div class="p2a-output" id="p2a-output">
  <h3>Install command</h3>
  <div class="p2a-install" id="p2a-install"></div>
  <h3 style="margin-top:16px">Ready-to-run code <button class="p2a-copy-btn" id="p2a-copy">Copy</button></h3>
  <pre><code id="p2a-code"></code></pre>
  <h3 style="margin-top:16px">Artifacts</h3>
  <div class="p2a-artifacts" id="p2a-artifacts"></div>
</div>

</div>

<script>
(function(){
const PRESETS=[
{id:"compare",title:"Compare models",hint:"summary + leaderboard",route:"Experiment.compare_on_corpus()"},
{id:"thesis",title:"Thesis tables",hint:"CSV for results / leaderboard / errors",route:"Experiment.prepare_thesis_tables()"},
{id:"bundle",title:"Full bundle",hint:"tables + report + benchmark",route:"Experiment.export_appendix_bundle()"},
{id:"pipeline",title:"Manual pipeline",hint:"full control via >>",route:"dataset >> model >> norm >> metric"}
];
const DATASETS=[
{id:"golos",label:"GolosDataset",code:'GolosDataset("data/golos")',dir:'"data/golos"'},
{id:"dagrus",label:"DagrusDataset",code:'DagrusDataset("data/dagrus")',dir:'"data/dagrus"'},
{id:"rudevices",label:"RuDevicesDataset",code:'RuDevicesDataset("data/rudevices")',dir:'"data/rudevices"'},
{id:"nemo",label:"NeMoDataset (custom JSONL)",code:'NeMoDataset(root_dir="data/custom", manifest_path="data/custom/manifest.jsonl")',dir:'"data/custom"'}
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
{id:"none",label:"No normalization",code:null},
{id:"simple",label:"SimpleNormalizer",code:"SimpleNormalizer()"},
{id:"dagrus",label:"DagrusNormalizer",code:"DagrusNormalizer()"},
{id:"golos",label:"GolosNormalizer",code:"GolosNormalizer()"}
];
const METRICS=[
{id:"composite",label:"Composite (all core)",code:"Metrics.composite()"},
{id:"wer",label:"WER",code:"Metrics.WER()"},
{id:"cer",label:"CER",code:"Metrics.CER()"},
{id:"accuracy",label:"Accuracy",code:"Metrics.Accuracy()"}
];

const S={preset:"compare",dataset:"golos",models:new Set(["gigaam_v3","whisper"]),norm:"simple",metrics:new Set(["composite"]),art:{"report":false,"browser":false,"bench":false,"pandas":false}};

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
  var pip=extras.length?"pip install plantain2asr["+extras.join(",")+"]":"pip install plantain2asr";
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
  try{await navigator.clipboard.writeText(el("p2a-code").textContent);this.textContent="Copied!";setTimeout(()=>{this.textContent="Copy"},1200)}catch(e){this.textContent="Failed"}
};

render();
})();
</script>

</div>

---

## What's next?

- [Quick Start](quickstart.md) -- a canonical runnable workflow from start to finish
- [API Reference -> Datasets](api/dataloaders.md) -- full dataset method documentation
- [API Reference -> Metrics](api/metrics.md) -- all available metrics
- [Extending](extending/index.md) -- how to add your own normalizer, model, or metric
