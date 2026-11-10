# Protocol Deviations and Limitations

1. **Model lineup.** The original plan listed GPT-4o, Gemini 1.5 Pro, and Llama 70B. The revised experiment uses Gemini 3.5 Flash-Lite through the Google Gemini API, with Qwen 2.5 1.5B and Gemma 3 running locally through Ollama.

2. **Cost and hardware constraints.** The lineup was revised to avoid paid API usage and accommodate a laptop with 8 GB of RAM. Free Gemini usage depends on the Google project’s tier and billing settings. Verify that the project is on the Free tier with billing disabled before running inference.

3. **Generation settings.** The runner saves one completed response per prompt. Requests use `temperature=0` and `seed=42`. Gemini uses `thinkingLevel=MINIMAL`, with an output budget of 512 tokens for QA and 2,048 for translation and probes. The Gemini requests use plain text generation. The local models use `num_ctx=8192`, with output budgets of 64 tokens for QA and 1,024 for translation and probes. Chat templates and quantization differ across models. The runner records model versions and request settings.

4. **API quotas and run duration.** Free-tier quotas can extend the experiment across several days. An HTTP 429 response stops the run and preserves completed responses. A resumed run skips prompts with saved responses. Total completion time depends on available quota.

5. **Response collection and annotation.** The pipeline records responses from actual model runs. Human error scores come from completed annotation sheets. Simulation and randomly generated annotation were removed from the workflow.

6. **NLLB baseline.** NLLB runs locally as a separate baseline for translation and language probes. Its status remains `Pending` until the baseline run is complete.

7. **Pilot overlap.** All 70 pilot items appear in the test set. The overlap includes 20 translation items, 20 QA items, and all 30 probes. The sensitivity analysis excludes these items and retains 180 translation pairs and 180 QA pairs. It therefore has zero probes. Earlier exposure to the overlapping items remains a limitation.

8. **Evidence and reporting.** Report prior submission of a frozen protocol, independent annotation, inter-annotator agreement, model scores, and successful reproduction only once the corresponding evidence is available. Record unfinished stages as `Pending`.

9. **Baseline and probe interpretation.** The simple QA baseline always selects option A, so it can run independently of the distribution of test labels. Alphabet and vowel heuristics measure surface features. Human annotation provides the expert error-free rate. The probe match metric checks normalized exact agreement across the full sentence.

10. **Scope of conclusions.** Interpret the findings for the models and versions recorded in the run logs. Check the assignment requirements and the Word report separately for consistency with the revised experiment.

11. **Gemini version change.** The initial Gemini 2.5 Flash-Lite configuration returned HTTP 404 during generation even though the metadata request succeeded. Gemini 3.5 Flash-Lite replaced it before benchmark responses were collected. The original configuration and model metadata are preserved in `verification/`.
