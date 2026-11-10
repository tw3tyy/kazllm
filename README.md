# KAZLLM: Kazakh–English Model Evaluation

KAZLLM evaluates Gemini, Qwen, and Gemma on Kazakh–English translation, multiple-choice question answering, and targeted language probes. Gemini runs through the Google Gemini API. Qwen and Gemma run locally through Ollama.

## Benchmark

| Task | Items | Prompts per model |
|---|---:|---:|
| Translation in both directions | 200 | 400 |
| Question answering in both languages | 200 | 400 |
| Language probes | 30 | 30 |
| **Total** | **430** | **830** |

Each saved response includes the prompt ID, model version, timestamp, and original model response. Reference answers are stored separately from model prompts.

### Repository status

This snapshot contains **276 of 830 benchmark responses** and **three smoke-test responses** for Gemini 3.5 Flash-Lite. The Qwen and Gemma benchmark runs, NLLB baseline, and human annotation are pending. The final comparison requires complete model runs. Smoke-test responses are stored separately.

## Setup

```sh
git clone https://github.com/tw3tyy/kazakhllm.git
cd kazakhllm
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Ollama is required for local inference. The demo can display saved responses and calculate simple baselines offline.

## Demonstration

```sh
./demo.sh
```

Enter a menu number and press **Enter**. The menu shows run progress, saved examples, individual benchmark items, results, and the experimental method.

| Command | Output |
|---|---|
| `./demo.sh --status` | Saved response counts for each model |
| `./demo.sh --examples --task mt` | A saved translation example |
| `./demo.sh --examples --task qa` | A saved multiple-choice QA example |
| `./demo.sh --examples --task probe` | A saved language probe |
| `./demo.sh --item 5 --model 1` | Item 5 and Gemini’s saved translations in both directions |
| `./demo.sh --results` | Completed results or baseline scores with run progress |
| `./demo.sh --method` | Experimental method and dataset audit |

Model numbers are `1` for Gemini, `2` for Qwen, and `3` for Gemma. Each saved example shows the prompt, model response, reference, single-item score, and response record.

## Run the experiment

### Gemini

Use a Google AI Studio project on the **Free tier with billing disabled**. Check the model’s quota in AI Studio before starting. The script prompts for the API key with hidden input.

```sh
./run_gemini.sh --check-api
./run_gemini.sh --smoke --free-tier-confirmed
./run_gemini.sh --free-tier-confirmed
```

The first command retrieves model metadata. The smoke test makes three requests on a fresh setup. The final command collects benchmark responses and resumes from the saved files when run again. If the project reaches its quota, the runner stops and preserves completed responses.

For one fresh request during a demonstration, run the following after any active benchmark run has stopped:

```sh
./demo.sh --live-gemini 201 --free-tier-confirmed
```

The live response is stored separately in `demo_outputs/`.

### Qwen and Gemma

Start the local Ollama server in one terminal:

```sh
./start_ollama_local.sh
```

In another terminal, from the project directory:

```sh
./prepare_local.sh
./run_local.sh
```

The preparation script downloads the configured local model weights. The runner evaluates Qwen 2.5 1.5B and Gemma 3 sequentially.

## Score the responses

After all three models reach 830 saved benchmark responses, run:

```sh
./reproduce.sh
./demo.sh --results
```

- **QA accuracy** is the share of model answers that match the reference option.
- **chrF2** measures character overlap between a translation and its reference.
- **Translation gap** is chrF2 for translation into Kazakh minus chrF2 for translation into English.
- **Paired QA comparisons** compare answers to the same items across languages. The scorer applies a Holm correction across the three model comparisons.

The main table is written to `results/main_results_table.md`. Detailed scores, corrected comparisons, and the sensitivity analysis are saved as JSON in `results/`.

## Pilot overlap and limitations

The pilot set overlaps with the benchmark on **70 items**: 20 translation pairs, 20 QA pairs, and all 30 probes. The sensitivity analysis excludes those items and retains 180 translation pairs and 180 QA pairs. The remaining set contains zero probes.

NLLB is a separate translation baseline. Human error annotation is a separate stage described in [the annotation guide](annotation/HOW_TO_ANNOTATE.md). Protocol changes are documented in [DEVIATIONS.md](DEVIATIONS.md).

## Project files

| Path | Contents |
|---|---|
| `data/` | Benchmark, input hashes, and overlap audit |
| `pilot/` | Pilot dataset |
| `outputs/` | Saved benchmark responses |
| `smoke_outputs/` | Saved smoke-test responses |
| `results/` | Run status and scoring outputs |
| `kazllm/` | Prompt construction, model adapters, and scoring |
| `tests/` | Automated checks |

Run the automated checks with:

```sh
python3 -m unittest discover -s tests -v
```
