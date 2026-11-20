<div align="center">

# 🦙 LLAMAREC

### Cross-Domain Recommendations with Local Large Language Models

*Recommend a film from someone's bookshelf. Recommend an album from their watchlist.*<br>
*All of it running on your own machine, with an explanation for every pick.*

<br>

[![CI](https://img.shields.io/github/actions/workflow/status/ThanigaiSingaravelan/llamarec-crossdomain/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/ThanigaiSingaravelan/llamarec-crossdomain/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-000000?style=for-the-badge&logo=ollama&logoColor=white)
![Llama 3](https://img.shields.io/badge/Llama_3-8B-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-NCF_Baseline-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**MSc Artificial Intelligence Dissertation · De Montfort University · 2025**

[Overview](#-overview) · [How It Works](#-how-it-works) · [Quick Start](#-quick-start) · [Usage](#-usage) · [Evaluation](#-evaluation) · [Results](#-results) · [Citation](#-citation)

</div>

---

## 📖 Overview

Most recommendation systems stay inside one category. If you have rated a hundred books, a typical system can suggest more books, but it has very little to say about which films you might enjoy.

LLAMAREC tries to close that gap. It gives a Llama 3 model a user's ratings in one domain, asks it to work out what the person actually likes, and then has it recommend items from a different domain. The model also explains why each item made the list.

Everything runs locally through [Ollama](https://ollama.com/), so user data never leaves the machine.

<table>
<tr>
<td width="50%" valign="top">

### 🔀 Cross-domain transfer
Moves preferences between **Books**, **Movies & TV**, **Music** and **CDs**.

### 🔒 Private by design
Llama 3 runs locally through Ollama. No user history is sent to an external API.

### 💬 Explainable picks
Every recommendation comes with a short, readable reason linked to the user's history.

</td>
<td width="50%" valign="top">

### 🧊 Cold vs warm users
Compares **cold** users (fewer than 50 interactions) with **warm** users (more than 200).

### 🧠 Three prompt strategies
Standard, few-shot and chain-of-thought prompting, each combined with retrieval.

### 📊 Proper evaluation
Precision@K, Recall@K, NDCG, diversity, novelty and coverage, compared against NCF and content-based baselines.

</td>
</tr>
</table>

---

## ⚙️ How It Works

```mermaid
flowchart LR
    A[(Amazon Reviews<br>2023)] --> B[Preprocessing]
    B --> C[Cross-domain<br>user profiles]
    C --> D{Cold or warm?}
    D --> E[Prompt generator<br>standard · few-shot · CoT]
    E --> F[🦙 Llama 3<br>via Ollama]
    F --> G[Recommendations<br>+ explanations]
    G --> H[Evaluator<br>vs NCF baseline]
```

1. **Collect.** Amazon review data is cleaned and converted to CSV.
2. **Profile.** Users who rated items in more than one domain are identified and split into cold and warm groups.
3. **Prompt.** The user's highly rated items in the source domain are placed into one of three prompt templates.
4. **Generate.** A local Llama 3 model returns three recommendations for the target domain, each with a reason.
5. **Evaluate.** Results are scored on ranking quality, diversity and novelty, then compared with traditional baselines.

---

## 🗂️ Project Structure

<details>
<summary><b>Click to expand the folder layout</b></summary>

<br>

```
llamarec-crossdomain/
├── app.py                                # Single entry point: python app.py <command>
├── requirements.txt                      # Runtime dependencies
├── requirements-dev.txt                  # + pytest, black, flake8
├── pytest.ini
├── data/                                 # Not in git (large) - created by preprocessing
│   ├── raw/                              # Raw Amazon review files go here
│   ├── processed/                        # CSVs created by amazon_collector
│   └── splits/                           # Cross-domain, cold/warm, train/test, k-fold splits
├── preprocessing/
│   ├── amazon_collector.py               # Loads and cleans the Amazon data
│   ├── domain_preprocessor.py            # Finds users active in several domains
│   └── data_splitter.py                  # Builds the data splits
├── prompts/
│   ├── prompt_generator.py               # Builds recommendation prompts
│   └── prompt_templates.py               # Standard, few-shot and CoT templates
├── runners/
│   ├── run_ollama_llamarec.py            # Main LLAMAREC runner
│   ├── run_cold_warm_scenarios.py        # Cold vs warm experiments
│   ├── run_enhanced_llamarec.py          # Template / model comparison study
│   ├── evaluation_runner.py              # Automated evaluation
│   └── baseline_evaluation.py            # LLAMAREC vs baselines
├── baselines/
│   ├── neural_cf.py                      # Neural Collaborative Filtering
│   ├── content_based.py                  # Content-based filtering
│   └── ncf_evaluation_runner.py
├── evaluations/
│   ├── quantitative_evaluator.py         # Main evaluation system
│   └── advanced_statistics.py            # Significance tests, effect sizes
├── experiments/
│   └── advanced_experimental_designs.py  # Factorial and ablation designs
├── testers/
│   ├── interactive_llamarec_tester.py    # Interactive demo
│   └── test_prompt_generator_cold_warm.py  # Cold/warm prompt harness (used by runners)
├── utils/                                # Ollama client, metrics, logging, file helpers
├── scripts/
│   └── ollama_chat.py                    # Quick Ollama connection / chat check
├── tests/                                # pytest unit tests
├── results/                              # Saved experiment outputs (JSON / reports)
├── docs/
│   └── index.html                        # Project web page
└── README.md
```

</details>

---

## 🚀 Quick Start

### Requirements

| | Minimum | Recommended |
|---|---|---|
| **Python** | 3.10 | 3.12 |
| **RAM** | 8 GB | 16 GB for full datasets |
| **Storage** | About 100 GB for processed data | |
| **CPU** | Multi-core | Multi-core (parallel processing is supported) |
| **LLM runtime** | [Ollama](https://ollama.com/) | |

> 💡 The dataset is the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) collection from McAuley Lab on Hugging Face. Most of the experiments used [`llama3.1:8b`](https://ollama.com/library/llama3.1:8b).

### 1 · Install

```bash
git clone https://github.com/ThanigaiSingaravelan/llamarec-crossdomain.git
cd llamarec-crossdomain

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 · Set up Ollama

```bash
ollama pull llama3:8b
ollama pull llama3:70b          # optional, for comparison
ollama serve
```

Then check Ollama, the model and the data folders:

```bash
python app.py check
```

### 3 · Prepare the data

Download the Amazon review files (Books, Movies & TV, CDs & Vinyl, Digital Music) and place them in `data/raw/`. The `data/` folder is git-ignored.

```bash
# All three steps: clean raw files -> build cross-domain profiles -> create splits
python app.py preprocess all

# Or one step at a time
python app.py preprocess collect     # data/raw       -> data/processed
python app.py preprocess profile     # data/processed -> data/splits (user_history.json, cold/warm users)
python app.py preprocess split       # data/splits    -> train/test, cold-start, k-fold splits
```

### 4 · Run it

```bash
# One recommendation for a random eligible user (or pass --user <id>)
python app.py recommend --source Books --target Movies_and_TV

# Try it interactively
python app.py interactive

# Cold vs warm comparison
python app.py coldwarm --mode comparative --max-users 10

# Standard run
python app.py run
```

Run `python app.py --help` for every command. Options after a command are passed to the underlying script, e.g. `python app.py coldwarm --help`.

### 5 · Evaluate

```bash
python app.py evaluate --batch --results-dir results
python app.py ncf --domain Books
python app.py baselines
```

---

## 💻 Usage

### Recommend films from a reading history

```python
from testers.interactive_llamarec_tester import InteractiveLlamaRecTester

tester = InteractiveLlamaRecTester(
    user_history_path="data/splits/user_history.json",
    ollama_model="llama3:8b",
)

rec = tester.generate_recommendation(
    user_id="A1B2C3D4E5",
    source_domain="Books",
    target_domain="Movies_and_TV",
    num_recs=3,
)

print(rec["recommendations"])
```

### Compare cold and warm users

```python
from runners.run_cold_warm_scenarios import ColdWarmScenarioRunner
from testers.test_prompt_generator_cold_warm import PromptGeneratorTester

base_tester = PromptGeneratorTester(splits_dir="data/splits")
runner = ColdWarmScenarioRunner(base_tester)

results = runner.run_comparative_study(
    max_users_per_scenario=20,
    temperature=0.7,
)
```

### Score the results

```python
from evaluations.quantitative_evaluator import RecommendationEvaluator

evaluator = RecommendationEvaluator("data/splits/user_history.json")

results = evaluator.evaluate_recommendations("results/llamarec_results.json", k=3)
evaluator.generate_evaluation_report(results, "evaluation_report.txt")
```

<details>
<summary><b>Model configuration</b></summary>

<br>

```python
model_configs = {
    "base_model": {
        "model_name": "llama3:8b",
        "temperature": 0.7,
        "max_tokens": 512,
    },
    "large_model": {
        "model_name": "llama3:70b",
        "temperature": 0.7,
        "max_tokens": 512,
    },
}
```

</details>

<details>
<summary><b>Prompt templates</b></summary>

<br>

**Standard**

```text
Based on the user's preferences in {source_domain}, generate personalised
recommendations for items from {target_domain}.

User's highly-rated items: {user_history}
Target domain: {target_domain}

Recommend top 3 items with explanations.
```

**Chain-of-thought**

```text
You are an expert recommendation system. Analyse the user's preferences step-by-step.

Step 1: Analyse patterns in user's {source_domain} preferences: {user_history}
Step 2: Identify transferable preferences to {target_domain}
Step 3: Recommend 3 {target_domain} items that match these patterns

Provide detailed reasoning for each recommendation.
```

</details>

<details>
<summary><b>Experimental design and statistics</b></summary>

<br>

```python
from experiments.advanced_experimental_designs import (
    AdvancedExperimentalDesigner,
    ExperimentalStatistics,
)

designer = AdvancedExperimentalDesigner(base_tester)

# Temperature × user type × domain × template
factorial_results = designer.run_full_experiment_suite("factorial")

# How much each component contributes
ablation_results = designer.run_full_experiment_suite("ablation")

stats = ExperimentalStatistics()
effect_size = stats.compute_effect_size(group1_scores, group2_scores)
required_n = stats.power_analysis(effect_size=0.5, power=0.8)
```

</details>

---

## 📏 Evaluation

| Category | Metric | What it tells you |
|---|---|---|
| **Ranking** | Precision@K | How many of the top K picks were relevant |
| | Recall@K | How many of the user's liked items were found |
| | NDCG@K | Whether the best items were ranked near the top |
| **Diversity** | Intra-list diversity | How different the picks are from each other |
| | Coverage | How much of the catalogue gets recommended overall |
| | Novelty | How far the picks move away from the user's history |
| **System** | Response time | How long the model takes to answer |
| | Success rate | How often a call returns a usable result |
| | Quality score | An automated check on the recommendation quality |

**Baselines:** Neural Collaborative Filtering (NCF), content-based filtering and a hybrid approach.

---

## 🏆 Results

| Finding | Detail |
|---|---|
| 🧊 **Cold-start users** | LLAMAREC did best for users with little history, which is where traditional methods tend to struggle. |
| 🔀 **Strongest transfers** | Books ↔ Movies and Music ↔ Books had the highest transfer success rates. |
| 💬 **Explanations** | Over 95% of recommendations came with a clear reason tied to the user's source preferences. |
| 🔥 **Warm vs cold** | In the `llama3.1:8b` study (33 users, 39 prompts) warm users scored 3.83 vs 3.74 for cold users on the 5-point heuristic quality score. The difference was not statistically significant (t = −0.97, p = 0.34); see [results/cold_warm_study_report_20250530_111755.txt](results/cold_warm_study_report_20250530_111755.txt). |

---

## 🛣️ Roadmap

**Features**
- [ ] Recommendations that use images as well as text
- [ ] Real-time serving
- [ ] Fine-tuning support
- [ ] More domains, such as games and electronics
- [ ] Web interface
- [ ] API endpoint for other apps

**Research**
- [ ] Conversational recommendations
- [ ] Causal analysis of cross-domain transfer
- [ ] Tracking user satisfaction over time
- [ ] Checking for cultural and demographic bias

---

## 🤝 Contributing

Contributions are welcome. Open an issue or pull request.

```bash
pip install -r requirements-dev.txt

python app.py test          # or: python -m pytest

black .
flake8 .
```

---

## 📚 Citation

If LLAMAREC helps your research, please cite it:

```bibtex
@mastersthesis{senthilkumar2025llamarec,
  title  = {LLAMAREC: Cross-Domain Recommendations Through Large Language Model Intelligence},
  author = {Senthil Kumar, Thanigai Singaravelan},
  school = {De Montfort University},
  year   = {2025},
  type   = {MSc Dissertation},
  note   = {Supervised by Dr Aboozar Taherkhani. Available at https://github.com/ThanigaiSingaravelan/llamarec-crossdomain}
}
```

---

## 🙏 Acknowledgements

- **Dr Aboozar Taherkhani** for supervising the project
- **De Montfort University** for research support and resources
- **Meta AI** for the Llama 3 model family
- **The Ollama team** for making local LLMs easy to run
- **McAuley Lab** for the public Amazon Reviews dataset

---

## 📄 License

This project is released under the [MIT License](LICENSE). You are free to use, change and share the code, as long as you keep the copyright notice.

---

<div align="center">

Built by **[Thanigai Singaravelan Senthil Kumar](https://www.linkedin.com/in/than-tsv/)**

⭐ If you found this useful, a star on the repo is always appreciated.

</div>
