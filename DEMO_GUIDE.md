# KAZLLM Project Demo

## Open the demo

Open Terminal in the project directory and run:

```sh
./demo.sh
```

Enter a menu number and press **Enter**. After viewing a screen, press **Enter** to return to the menu, or enter the next menu number directly. If a benchmark run is active, open the demo in a separate Terminal window.

| Option | Menu label | What it shows |
|---|---|---|
| `1` | Run progress | The number of saved responses out of 830 for each model. Smoke-test responses are counted separately. |
| `2` | Saved examples | Saved examples of translation, question answering, and a language probe. |
| `3` | Inspect an item | The prompt and saved responses for a selected benchmark item and model. If a response is pending, the screen displays the prompt and its status. |
| `4` | Results and baselines | The completed results table when available. While model runs are pending, the screen shows progress and calculates simple baselines. |
| `5` | Method and dataset audit | The experiment setup, benchmark checksum, and overlap between the pilot and test sets. |
| `0` | Exit | Closes the demo. |

## Run individual views

These commands open a specific view directly:

```sh
./demo.sh --status
./demo.sh --examples --task mt
./demo.sh --examples --task qa
./demo.sh --examples --task probe
./demo.sh --item 201 --model 1
./demo.sh --results
./demo.sh --method
```

Model `1` is **Gemini**, model `2` is **Qwen**, and model `3` is **Gemma**. Item numbers range from `1` to `430`. Item `1` is a translation pair, item `201` is a question-answering pair, and item `401` is a language probe.

The **PROMPT** section shows what was sent to the model. **SAVED RESPONSE** shows its recorded answer. **SINGLE-ITEM CHECK** compares the selected answer with its reference. For QA, the check compares the selected option with the correct option. For translation, it shows the reference translation and sentence-level chrF2.

**RESPONSE RECORD** shows the model version, timestamp, response ID, and source folder. The `outputs/` directory contains benchmark responses. The `smoke_outputs/` directory contains smoke-test responses.

## Send one live request

Wait for any active run in the project directory to finish. With a Google AI Studio project on the Free tier and billing disabled, run:

```sh
./demo.sh --live-gemini 201 --free-tier-confirmed
```

Enter the API key when the hidden prompt appears. The command sends one new Gemini request for the first language condition of item `201`. The response is stored in `demo_outputs/`, separately from the benchmark responses used for scoring.

## Prepare the full results table

To collect or resume Gemini responses, run:

```sh
./run_gemini.sh --free-tier-confirmed
```

The runner skips prompts that already have saved responses.

For Qwen and Gemma, start Ollama in one Terminal window:

```sh
./start_ollama_local.sh
```

Leave the server running. In another Terminal window, from the project directory, run the following after any active benchmark run has finished:

```sh
./prepare_local.sh
./run_local.sh
```

Once all three models reach **830/830 saved benchmark responses**, calculate and display the results:

```sh
./reproduce.sh
./demo.sh --results
```

## Explain the results

- **QA accuracy** is the proportion of answers that match the reference option.
- **chrF2** measures character overlap between a translation and its reference.
- **Translation gap** is chrF2 for translation into Kazakh minus chrF2 for translation into English.
- **Paired QA comparisons** compare each model’s answers to the same items across the two languages. The Holm correction is applied across the three model comparisons.

While model results are pending, the demo displays simple baselines. **Always A** selects option A for every QA prompt. **Seeded random** selects an option using a fixed random seed. **Dictionary MT** translates through dictionary substitutions.

Detailed scores are saved in `results/scores.json`. Corrected comparisons are saved in `results/holm.json`. The sensitivity analysis excludes the 70 items shared by the pilot and test sets, leaving 180 translation pairs, 180 QA pairs, and zero probes.

NLLB and human error annotation are separate stages. See [README.md](README.md) for NLLB setup and [the annotation guide](annotation/HOW_TO_ANNOTATE.md) for the annotation procedure.

## Suggested presentation script

> We evaluate Gemini, Qwen, and Gemma on Kazakh and English translation and question answering, with additional language probes. The benchmark contains 430 items, which produce 830 prompts per model. The runner saves each response with its model version, prompt ID, and timestamp. The scoring step uses those records to calculate QA accuracy, chrF2, and paired comparisons. Here, I can open a benchmark item, show the original prompt and saved response, and explain how that response was scored.
