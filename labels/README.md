# Report-derived soft labels

Only 58 of the 4,407 training studies carry the organisers' labels, but every training study comes with a free-text
radiology report, in one of nine languages. `p8_prompts.py` is the prompt with which Claude (`claude-opus-5`, effort
high) read each report in its original language and returned, for each of the 12 findings, a continuous score in
[0, 1] and the verbatim span of the report that drove it.

- **Definitions.** The prompt quotes the threshold wording of the competition's label definitions ("definitely contacts
  the meniscal surface on at least two images", "greater than 50 percent of cartilage thickness", ...), because the
  exact words move the positive rate a lot.
- **Calibration (withheld).** For three findings the organisers' labels on the 58 studies sat measurably away from the
  written definitions. The prompt that made the labels told the model where, in a CALIBRATION block. This copy
  withholds that block's measured paragraphs, so `prompt_id()` returns `1c677253acbd8064`, not the original
  `631a59565622e7cc`.
- **A continuous scale.** The metric is AUC, so ties cost. The prompt gives landmarks on a continuum and asks the model
  to spread its scores, with silence (0.10) above an explicit negation (0.03).
- **Structured output.** One call per report. `SCHEMA` was passed as a JSON-schema output format (`OUTPUT_CONFIG_FORMAT`),
  so a missing label or a parse failure could not happen. A JSON schema cannot bound a number, so a score outside
  [0, 1] was set to missing with a warning, never clipped.

How the scores were used:

- the first labels, **v1**, were the mean of these scores and an earlier run of Gemma 3 27B. v1 stayed the target of the
  validation metric for the whole competition;
- from late September the members were trained on these Claude scores alone (**claude_v1**). Submission 31, which
  retrained the CoAtNet member and refitted the stacker on claude_v1, scored public 0.938 / private 0.925, against
  0.903 / 0.900 for submission 21 before it;
- after the deadline, five members were retrained on drizlo's image-read labels instead (CC BY 4.0, see `NOTICE`).
  Each gained 18 to 24 private ticks against its claude_v1 version.

The labels themselves are not published: they are derived from the competition's reports, and the quotes are report
text.

The API caller is not included. It sent `SYSTEM` as the system prompt and `user_msg(report)` as the user turn, with
adaptive thinking.
