# Draft protocol-change note — adapt before submission

To avoid paid API usage and accommodate an 8 GB laptop, the planned model set was
replaced with Gemini 3.5 Flash-Lite accessed directly through the Google Gemini API,
plus Qwen 2.5 1.5B and Gemma 3 executed locally through Ollama. Free-tier eligibility
and disabled billing must be checked in the Google project before inference; the
API key alone does not establish billing status. Quota exhaustion stops the runner,
and completed outputs are retained for a later continuation.

These models are not equivalents of GPT-4o, Gemini 1.5 Pro, or Llama 3.1 70B.
Findings must be attributed only to the actual models and versions recorded in the
run logs. Server inference and quantized local inference also differ in runtime
conditions; latency measurements should not be interpreted as hardware-controlled
model comparisons.

The inherited benchmark contains all 70 pilot items: 20 MT, 20 QA, and all 30 probes.
Sensitivity excludes them, leaving 180 MT and 180 QA items and no independent probes.
This analysis does not reverse prior pilot exposure. NLLB and human annotation must
remain pending until actually completed.

Insert actual dates, exact returned model versions, results, and assignment-specific
justification after the runs. This draft is not evidence that runs have occurred.
