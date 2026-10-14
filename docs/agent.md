# VizJudge v0.4 Agent loop

The objective is **少画图、看重点、给出下一步**: build a plan, call bounded analysis
tools, judge the evidence, improve the selected charts, and produce actionable results.

## Run

```bash
vizjudge agent data.csv --target outcome --task classification --output outputs/agent
vizjudge agent data.csv --target price --task regression --baseline age --top-k 5
python examples/agent_loop/demo.py --output outputs/demo-v04
```

`analyze` still runs the direct analysis pipeline. `agent` adds planning, isolated
tool execution, judging, refinement, confidence, and an execution record. The
MCP server exposes both `analyze_dataset` and `run_agent`. Python users can import
`ExecutionBudget` and `run_agent` from `vizjudge.agent`.

## Plan and execution

The planner is deterministic in v0.4. It does not require an LLM to decide which
tools are available. Its explicit plan is recorded before execution:

1. `profile`: inspect quality and record original/analyzed row counts.
2. `analyze`: resolve the task, generate capped candidates, measure associations,
   validate incremental predictive value, and select diverse charts without rendering.
3. `judge`: assign statistical confidence and optionally obtain a grounded LLM review.
4. `refine`: apply one supported action and call `judge` again. Stop at convergence,
   no change, or the iteration limit.
5. `render`: render the final selected specs within the image budget.

Each tool runs in a separate Python subprocess through a fixed allowlist. User
scripts, LLM-provided code, shell commands, `eval`, and `exec` are not accepted.
Timed-out workers are terminated. A worker uses its parent's OS permissions;
this is execution control, not an OS security sandbox or a hard memory quota.

`prefer_validated` lowers attention to signals with no measured incremental gain;
`increase_diversity` removes repeated features and relationships flagged as
redundant; `reduce_charts` halves the selection, retaining at least one chart.
Advice is regenerated after refinement. Refinement changes the selected set and
priority; it does not invent new statistics or change the original dataset.

## Statistical validation and confidence

Validation uses seeded shuffled folds (stratified for classification), a ridge
regression baseline or a ridge least-squares classifier, and identical folds for
the baseline and baseline plus each feature. Classification measures balanced
accuracy; regression measures RMSE. Positive `incremental_gain` means improvement
for either task. By default the baseline contains only an intercept. `--baseline`
adds existing features; each proposed feature is still added individually.

Numeric medians, scaling, missingness indicators, and category vocabularies are
fit on training folds alone. Categorical vocabularies are capped at 20 frequent
levels; unknown levels map to the all-zero encoding. Validation is capped at 5,000
labeled rows and 20 features, chosen by initial heuristic priority. Row sampling,
skips, fold gains, and fold variability are explicit in the report. Classification
validation requires 2–20 classes and at least `cv_folds` examples per class; all
tasks require at least 30 labeled rows. `--no-validation` disables these checks.

Confidence is an explainable **prioritization label**, not a probability or a
statistical confidence interval. High confidence requires at least 100 complete
observations, positive gain in every validation fold, and no chart warnings.
Inconsistent or non-positive validation produces low confidence. Small samples,
sparse groups, truncation, and leakage warnings limit confidence. A live LLM
cannot raise this measured confidence or increase statistical priority.

IID folds do not protect against temporal, grouped, repeated-entity or semantic
leakage. Features are screened using the full analysis sample, so these checks
are exploratory, not an unbiased estimate of a selected production model. Recheck
the final hypotheses on a domain-appropriate independent validation split.

## LLM Judge

Three explicit modes are supported:

- No adapter: statistical-only review; the report does not claim an LLM was called.
- `--judge-file response.json`: replay a recorded response, without a live model.
- `--judge-url https://your-service/judge`: invoke a configured JSON Judge service.
  HTTP is allowed for localhost; other URLs must use HTTPS. Set
  `VIZJUDGE_JUDGE_TOKEN` for optional bearer authentication. No endpoint is invoked
  unless configured by the caller.

The HTTP service receives a POST object containing `instruction` and `evidence`.
The evidence contains task, feature names, chart titles, aggregate metrics,
warnings, and stable evidence IDs. It excludes raw rows and profile `top_values`.
An explicitly configured service therefore receives schema names and aggregate
statistics. Tokens are not written to the execution record. The service can wrap
any preferred LLM; this provider contract is not a vendor SDK endpoint.

Return a JSON object matching this shape (copy actual IDs from the request):

```json
{
  "reviews": [
    {
      "chart_id": "<exact chart_id from request>",
      "evidence_ids": ["<exact evidence ID belonging to this chart>"],
      "priority": 70,
      "verdict": "keep",
      "rationale": "Why this evidence deserves attention.",
      "next_step": "A concrete validation or modeling action."
    }
  ],
  "refinement": "increase_diversity"
}
```

Every supplied chart must be reviewed exactly once. IDs must exist and belong to
that chart. Priority must be finite and in `[0, 100]`; text fields are capped at
2,000 characters. Supported verdicts are `keep` and `defer`; refinements are `none`,
`prefer_validated`, `increase_diversity`, and `reduce_charts`. Invalid responses
fail the run with a saved failure record. These checks validate references and
structure, not the truth of every natural-language claim.

The combined priority is the minimum of the statistical and LLM priorities.
The LLM may defer a chart but cannot reverse a statistical deferral. File replay
is used only on the first pass; later passes use statistics. Earlier LLM rationale
and next steps remain attached to retained charts with their source iteration.
HTTP adapters may judge each iteration independently.

## Budgets and trace

| Limit | Default | CLI option |
| --- | ---: | --- |
| Python tool calls | 10 | `--max-tool-calls` |
| Judge iterations | 2 | `--max-iterations` |
| Worker execution wall time | 120 s | `--max-seconds` |
| Each Python worker | 30 s | `--python-timeout` |
| Analysis rows | 10,000 | `--max-rows` |
| Candidate charts | 200 | `--max-candidates` |
| Newly rendered images | 8 | `--max-rendered-charts` |
| Input file bytes | 50,000,000 | `--max-input-bytes` |

Worker timeouts use the smaller of the per-call and remaining total limits.
Saving diagnostics after timeout is allowed outside the worker execution budget.
File loading is checked against input bytes before parsing; row sampling follows
parsing. Memory usage is not enforced by an OS quota. Candidate generation favors
target relationships under tight limits. Untargeted numeric pair searches examine
at most `max_candidates * 5` pairs in column order. The output size cap is checked
after receiving a worker response, rather than acting as a streaming memory cap.

`report.json` is the full analysis with `agent` provenance; `execution.json` contains
the plan, budgets, iteration judgments, dataset SHA-256, Python version, and tool
events. Each event records the tool, reason, parameters, input/output SHA-256,
status, and duration. A source file change during execution fails the run.
`summary.md` includes images, confidence, and next steps. Existing unrelated
images in a reused output directory are not counted against a run's new-image limit.

Budget exhaustion saves a partial report with `agent.status = "budget_exhausted"`;
CLI exits with code 3. Failures save the trace and raise an error (CLI code 2).
Completion exits with code 0, with a stop reason such as `converged`, `no_change`,
`iteration_budget`, or `render_budget`.
