# RADAR

### Quick Start

Install all required packages including pytorch geometric, openai, etc. 

### Project Structure

```
RADAR
├── data
│   └── AQuA
├── mas
│   ├── agents
│   ├── datasets
│   ├── domain
│   ├── gnn
│   ├── graph
│   ├── __init__.py
│   ├── llm
│   ├── prompt
│   ├── tools
│   └── utils
├── model
│   ├── denoising.py
│   ├── gd.py
│   ├── ordering.py
│   └── utils.py
├── accuracy.py
├── process_datasets.py
├── run_aqua.py
├── run_gsm8k.py
├── run_humaneval.py
├── run_mmlu.py
├── run_multiarith.py
├── run_svamp.py
├── template.env
└── utils.py
```

### Add API keys in template.env and change its name to .env
```
BASE_URL = "" # your base url
API_KEY = "" # your api key
```

### Download Datasets
Download MMLU, HumanEval and GSM8K ect. And put them in different folders.

### Run on MMLU dataset

```
python run_mmlu.py
```