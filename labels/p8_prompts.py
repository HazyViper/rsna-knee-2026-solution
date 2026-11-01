"""Phase 8-A prompt: Claude API label extraction from raw multilingual reports.

Design notes that are load-bearing and should not be "tidied" away:

* The official threshold wording is quoted VERBATIM (plan 1-1). Phase 2 showed
  the Effusion threshold alone swings the positive rate from 0% to 92%, so the
  exact words are the interface, not decoration.

* The RUBRIC ASYMMETRY block (plan 1-2) is the differentiator over the local
  Gemma run. A4 compared the organisers' 58 against their own written
  definition and found they do not apply it symmetrically: stricter than the
  definition on the trauma labels, looser on the effusion family. Telling the
  model the definition alone reproduces Gemma's errors; telling it where the
  annotators actually drew the line is the only new information Phase 8 has.

* One call, not two. A5's two-stage design produced evidence that contradicted
  its own label in 26% of positive cells. Adaptive thinking does the evidence
  pass internally, and the quote we ask for is a citation, not a separate
  extraction that the verdict can drift away from.

* output_config.format (json_schema) rather than "return JSON only" in prose.
  It makes a parse failure or a missing label a server-side impossibility
  rather than something to measure and repair.

* Scores are continuous. AUC is a ranking metric, so ties are the enemy: the
  anchors below are landmarks on a continuum, and the prompt says so twice.
"""

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

# --------------------------------------------------------------------------
# 1-1  Official definitions, thresholds quoted verbatim.
# --------------------------------------------------------------------------
LABEL_DEFS = """\
1. ACL - Anterior cruciate ligament tear.
   POSITIVE: complete or partial tear, rupture, discontinuity of fibres,
   non-visualisation of the ligament, or a failed/torn ACL graft.
   NEGATIVE: intact ligament; sprain without fibre disruption; mucoid
   degeneration alone; ganglion cyst alone; an intact graft.

2. MCL - Medial collateral ligament tear.
   POSITIVE: tear or rupture of any grade.
   NEGATIVE: intact ligament; periligamentous oedema alone; grade I sprain
   with no fibre disruption.
   The LATERAL collateral ligament (LCL) is NOT this label. Nothing on the
   lateral side of the knee belongs here.

3. Medial Meniscus - Medial meniscus tear.
   POSITIVE only when the abnormal signal "definitely contacts the meniscal
   surface on at least two images", or when a tear is stated outright
   (bucket-handle, radial, horizontal, longitudinal, complex, root tear,
   displaced fragment, post-operative re-tear).
   NEGATIVE: intrasubstance or grade 1-2 degeneration that does not reach the
   surface; signal touching the surface on only one image; meniscal extrusion
   alone; a post-operative meniscus with no re-tear stated.

4. Lateral Meniscus - Lateral meniscus tear.
   Identical rule: signal that "definitely contacts the meniscal surface on at
   least two images", or a tear stated outright.
   A discoid lateral meniscus WITHOUT a tear is NEGATIVE.

5. Medial OA - Osteoarthritis of the MEDIAL tibiofemoral compartment.
   POSITIVE only for cartilage loss "greater than 50 percent of cartilage
   thickness" (full-thickness or near-full-thickness loss; ICRS or Outerbridge
   grade 3-4; bone-on-bone), or an outright statement of advanced/severe
   osteoarthritis of that compartment.
   NEGATIVE: osteophytes alone; subchondral oedema or sclerosis alone; joint
   space narrowing alone; grade 1-2 chondral thinning (<= 50 percent).

6. Lateral OA - Osteoarthritis of the LATERAL tibiofemoral compartment.
   Same "greater than 50 percent of cartilage thickness" rule.
   This label is about the lateral FEMOROTIBIAL COMPARTMENT and nothing else.
   The words "lateral", "external", "aussen", "externo", "externe" appear
   constantly in these reports attached to other structures - the lateral
   meniscus, the lateral collateral ligament, the lateral patellar facet, the
   lateral retinaculum, the lateral recess, a lateral tibial plateau contusion.
   None of those make this label positive. A degenerative finding and the word
   "lateral" in the same sentence is not enough; the degeneration must be OF
   the lateral tibiofemoral compartment cartilage.

7. PF OA - Osteoarthritis of the PATELLOFEMORAL compartment (patella and
   trochlea). Same "greater than 50 percent of cartilage thickness" rule.
   Chondromalacia patellae grade 3-4 counts; grade 1-2 does not. Retropatellar
   or trochlear full-thickness cartilage loss counts.

8. Effusion - Joint effusion. Official definition: POSITIVE only when the
   effusion is "moderate or large". But see the CALIBRATION section - the
   annotators did not apply it that way.

9. Synovitis - Synovial inflammation: synovial thickening, synovial
   enhancement, pannus, or a stated synovitis. See CALIBRATION.

10. Baker's - Baker's (popliteal) cyst.
    POSITIVE only when the cyst is "roughly 1 cm or greater" in any dimension,
    or is described as other than tiny/minimal.
    NEGATIVE: a trace or tiny cyst; a few millilitres of fluid in the
    gastrocnemio-semimembranosus bursa.

11. Contusion - "acute" bone contusion / bone marrow oedema of traumatic
    origin (bone bruise, trabecular microfracture with no cortical break).
    NEGATIVE: degenerative subchondral oedema adjacent to arthritic cartilage
    loss; old, chronic, or healed marrow change; a subchondral cyst.
    See CALIBRATION - the annotators were stricter here than this text.

12. Fracture - "acute" fracture: a cortical break, an impaction, or an
    avulsion (including a Segond fragment).
    NEGATIVE: old, healed, or chronic fracture; a bone contusion with no
    cortical break (that is label 11, not this one); an os fabella or other
    accessory ossicle.
"""

# --------------------------------------------------------------------------
# 1-2  Measured rubric asymmetry. This is the Phase 8 differentiator.
# --------------------------------------------------------------------------
# WITHHELD IN THE PUBLIC COPY: the paragraphs measured on the 58 training studies
# that carry the organisers' labels.  The labels were made with them (prompt_id
# 631a59565622e7cc); without them prompt_id() returns a different value.
CALIBRATION = """\
CALIBRATION - HOW THE ANNOTATORS ACTUALLY GRADED

[Withheld in the public copy of this prompt: three paragraphs, measured on the
studies that carry the organisers' labels, on how Effusion, Synovitis and
Contusion were graded.]

Everywhere else, and for every label not named above, apply the definition as
written and resolve doubt downward: "ambiguous or borderline findings were
graded as negative to favor specificity".
"""

# --------------------------------------------------------------------------
# 1-3  Output scale.
# --------------------------------------------------------------------------
SCALE = """\
SCORING

Every label gets a CONTINUOUS score in [0, 1]. This is not a yes/no dressed up
as a number: the only thing that is measured downstream is the RANKING of the
4,407 studies against each other, so two studies that differ in the strength of
their evidence must not receive the same number. Use two decimal places and
spread them out.

Landmarks, not a menu - interpolate freely between them:

  0.97  stated outright and clearly over the threshold
        ("complete ACL rupture", "bone-on-bone medial compartment")
  0.85  stated positive, threshold very likely met but not spelled out
  0.70  stated positive, threshold plausible but unstated or borderline
  0.50  genuinely ambiguous - the report supports both readings
  0.30  mentioned but explicitly BELOW threshold
        (grade 1-2 chondropathy, intrasubstance signal, tiny cyst)
  0.15  hedged mention only ("cannot be excluded", "questionable")
  0.10  NOT MENTIONED ANYWHERE in the report
  0.03  explicitly excluded ("ACL intact", "no effusion", "menisci normal")

Note the order of the last two: silence scores HIGHER than an explicit
negation. A structure the radiologist did not comment on is weaker evidence of
absence than one they checked and called normal. Do not collapse them.

Within a landmark, differentiate. Two positive ACL tears where one says
"complete rupture with retracted fibres" and the other says "tear" are 0.97 and
0.88, not both 0.97. Two unmentioned labels where one report is a three-line
normal study and the other is a long report that discusses every other
structure are 0.12 and 0.07.

For each label also return `quote`: the span from the report that drove the
score, copied VERBATIM in the original language, no translation and no
paraphrase. Empty string only when the finding is genuinely not mentioned. If
the quote you pick is a NEGATION, the score must be low - a quote that
contradicts its own score is a bug.
"""

SYSTEM = f"""\
You are grading knee MRI reports for a research dataset. The reports are
written in one of nine languages: English, Spanish, Turkish, Croatian, Greek,
German, Bulgarian, Dutch, French. Read the report in its original language. Do
not translate it, and do not let the language, the length, or the house style
of the report influence the scores - only its clinical content.

Assign all 12 labels below to every report.

{LABEL_DEFS}
{CALIBRATION}
{SCALE}

Two standing rules:
- "Medial" and "lateral" always refer to the COMPARTMENT or the SIDE OF THE
  KNEE, never to the left or right leg. A report about a left knee does not
  make anything "lateral".
- A finding described in a comparison to a prior study, in a clinical-history
  line, or as the indication for the scan is not a finding on THIS scan unless
  the body of the report confirms it.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        lb: {
            "type": "object",
            "properties": {
                "score": {"type": "number"},   # range enforced by the prompt and
                                           # audited client-side: json_schema
                                           # rejects minimum/maximum on numbers
                "quote": {"type": "string"},
            },
            "required": ["score", "quote"],
            "additionalProperties": False,
        } for lb in LABELS
    },
    "required": LABELS,
    "additionalProperties": False,
}

OUTPUT_CONFIG_FORMAT = {"type": "json_schema", "schema": SCHEMA}


def user_msg(report: str) -> str:
    return f"REPORT:\n{report}\n\nScore all 12 labels."


def prompt_id() -> str:
    """Stable hash of everything that defines a run's semantics, for the
    provenance record the winners' requirements will ask for."""
    import hashlib, json
    h = hashlib.sha256()
    h.update(SYSTEM.encode())
    h.update(json.dumps(SCHEMA, sort_keys=True).encode())
    return h.hexdigest()[:16]
