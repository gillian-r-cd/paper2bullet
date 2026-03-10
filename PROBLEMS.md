# Problems

## Remediation Update

- Updated at: `2026-03-07`
- Scope: `CARD_CONTENT_REMEDIATION` execution in pipeline, schema, prompt contract, deterministic gates, duplicate governance, review diagnostics, and quality metrics.
- Current state after implementation:
  - `Same-Evidence Near-Duplicate Cards`: `mitigated_with_monitoring`
  - `Same-Paper Cross-Topic Resurfacing`: `reported_needs_remediation`
  - `Abstract-Level Evidence Bias`: `mitigated_with_monitoring`
  - `Paper-Specific Value And Mechanism Loss`: `mitigated_with_monitoring`
- Notes:
  - The governance chain now enforces body-priority evidence selection, paper-specific object fields, grounding gates, and same-paper duplicate suppression before persistence.
  - Review/API now exposes grounding and duplicate diagnostics, and quality metrics are queryable via `/api/quality/metrics`.
  - Historical cards generated before this remediation are not automatically rewritten; compare pre-fix/post-fix runs and rerun where needed.

## Same-Evidence Near-Duplicate Cards

- Reported at: `2026-03-07`
- Status: `mitigated_with_monitoring`
- Severity: `high`

### Symptom

Multiple candidate cards are being generated from the same paper, same topic, and same core evidence paragraph, while only changing framing or teaching angle slightly.

Concrete example cluster:

- `card_e077a630d091467c8cd16e107f458156`
- `card_bdbc64a583734c3c9f22f7c740e2f177`
- `card_260801fc497140fab9eb376276b2f780`

These cards all belong to:

- run: `run_7e69e5eae78e49c58d61ee37d133fbf6`
- paper: `Towards The Ultimate Brain: Exploring Scientific Discovery with ChatGPT AI`
- topic: `learning in the age of AI`

At least two of the cards reuse the exact same primary evidence section:

- `section_f5d0caee755b44c389d3e11b6e789086`

The third card adds supplementary evidence, but still centers on the same core source paragraph and insight cluster.

### Why This Is A Problem

- It weakens the system's goal of extracting distinct `aha moment` candidates rather than producing multiple angle variants of the same idea.
- It increases reviewer burden because reviewers must manually identify near-duplicate cards without strong system assistance.
- It allows multiple semantically overlapping cards to survive into accepted/exported outputs.
- It suggests current generation is better at reframing than at enforcing atomic distinction between cards.

### Prevalence Check

Observed database scan at the time of reporting:

- total cards: `145`
- duplicate-evidence groups within same `run + paper + topic`: `20`
- cards inside those groups: `48`
- share of cards inside duplicate-evidence groups: `33.1%`
- largest observed duplicate-evidence group size: `3`

This indicates the issue is not isolated.

### Assessment Against Project Goals

- This is not fully aligned with the product goal.
- The project does allow atomic cards and explicitly avoids auto-merging cards by default.
- However, the current rate of same-evidence multi-card generation is too high to treat as acceptable atomic variation.
- The current behavior should be treated as a systematic quality gap in duplicate/near-duplicate governance.

### Suspected Root Cause

- Current extraction/judgement can split one source paragraph into multiple teaching-angle variants.
- The system currently lacks a strong intra-paper/intra-topic duplicate suppression or duplicate-awareness step during generation.
- Review UI does not yet expose near-duplicate neighbors strongly enough to help reviewers collapse these clusters efficiently.

### Desired Direction

- Add stronger near-duplicate detection inside the same `paper + topic + run` scope.
- Surface nearest-neighbor / duplicate-cluster hints in review workflows.
- Preserve atomic-card behavior, but distinguish true separate insights from simple framing variants of the same core evidence.

## Same-Paper Cross-Topic Resurfacing

- Reported at: `2026-03-09`
- Status: `reported_needs_remediation`
- Severity: `critical`

### Symptom

The same paper can enter a run under multiple overlapping topics, then generate cards repeatedly across those topics even when the underlying paper object or learner shift is the same.

This is not a formatting issue. It happens upstream in retrieval, topic attachment, and card generation.

### Why This Is A Problem

- It inflates the apparent evidence base because one paper looks like several independent discoveries.
- It biases review attention toward papers that simply matched more overlapping keywords.
- It makes medium-quality technical papers feel more important than they are because they resurface repeatedly.
- It weakens the product goal of curating distinct `aha` candidates rather than repeating the same paper under different topic shells.

### Observed Pattern

In the analyzed run on `2026-03-09`, the same paper repeatedly attached to multiple topics before and during card generation. This included cases where:

- one paper appeared in several overlapping AI-agent / memory / workflow topics
- one paper later produced multiple cards across different topics despite centering on the same core paper object
- keyword overlap increased exposure without increasing true `aha` diversity

### Assessment Against Project Goals

- Topic multiplicity is a retrieval artifact, not a value signal.
- A paper should not gain priority merely because it matched many neighboring topics.
- Topic assignment should help routing and analysis, but it must not become a license for repeated card generation.

### Suspected Root Cause

- Topic sets are currently broad and semantically overlapping.
- Paper-topic attachment happens before a stronger paper-level dedupe and strongest-aha selection step.
- Prompt/rubric logic historically focused on `paper + topic + card` quality, but not enough on `same paper across topics`.
- The system lacks a paper-level rule that says: default to the strongest single aha before allowing additional topic-specific framings.

### Desired Direction

- Dedupe at the paper level before topic-level resurfacing can multiply cards.
- Treat topic as a routing lens, not as independent evidence that the paper deserves more cards.
- Prefer the strongest single aha from a paper by default; require clear independence before preserving a second one.
- Extend duplicate governance from `same-evidence near-duplicate` to `same-paper cross-topic resurfacing`.

## Abstract-Level Evidence Bias

- Reported at: `2026-03-07`
- Status: `mitigated_with_monitoring`
- Severity: `high`

### Symptom

A noticeable share of generated cards uses evidence from abstract-like or front-matter paragraphs rather than from the paper's concrete mechanism, method, result, or detailed argument sections.

This is not just an isolated card-level issue. In several runs, cards repeatedly cite evidence that clearly begins with `Abstract` or otherwise reflects paper framing rather than body-level evidence.

### Prevalence Check

Observed database scan at the time of reporting:

- total cards: `145`
- total evidence items: `206`
- abstract-like evidence items: `29`
- full-database abstract-like evidence share: `14.08%`

Run-level concentration is materially worse in some cases:

- `run_b97155b2aa684f7fbdf94d2858f986ad`: `37.5%`
- `run_f8c9aae8368d42eca2646ff024f1a363`: `24.53%`
- `run_7e69e5eae78e49c58d61ee37d133fbf6`: `24.39%`

This indicates a systematic bias that becomes severe in some runs.

### Why This Is A Problem

- It undermines the project's goal of extracting real learner-facing `aha` moments rather than paper framing or high-level summary.
- It makes cards feel generic and unhelpful because they fail to surface the paper's actual mechanism, value, model, or evidence detail.
- It causes the system to overfit to what the paper says it is about, instead of what the paper concretely demonstrates.
- It weakens downstream review because reviewers receive cards that already lack substantive grounding.

### Suspected Root Cause

- Prompt section selection currently truncates to the first `MAX_PROMPT_SECTIONS`, which biases extraction toward front-matter content.
- Markdown-derived parsing currently stores many sections under generic titles like `Markdown Extraction`, so the system loses explicit structural knowledge about `Abstract` vs. body sections.
- Extraction prompts prohibit generic summaries in principle, but do not explicitly instruct the model to prefer body evidence over abstract evidence when both are available.
- Once extraction selects an abstract-heavy section, normalization directly materializes the entire selected section as card evidence, so the bias persists into final cards.

### Desired Direction

- Improve section sampling so prompts are not dominated by the first few paragraphs.
- Preserve or reconstruct stronger section structure metadata, especially abstract/body distinctions.
- Explicitly discourage abstract-first evidence selection when body evidence is available.
- Add review/debug visibility so operators can see whether a card is grounded mainly in abstract/front-matter evidence.

## Paper-Specific Value And Mechanism Loss

- Reported at: `2026-03-07`
- Status: `mitigated_with_monitoring`
- Severity: `critical`

### Symptom

Many generated cards fail to surface the paper's actual value-bearing object: the concrete model, mechanism, method, experiment, result, failure mode, or operational pattern that makes the paper worth reading.

Instead, cards often stay at one of these weaker levels:

- abstract-level framing of what the paper is about
- generic research motivation
- literature- or theme-level commentary
- course-language reframing that is only loosely grounded in the paper's unique contribution

This produces cards that may look fluent, but do not materially help a course designer understand what the paper actually contributes.

### Latest Run Evidence

The issue is visible in the latest analyzed run:

- run: `run_7e69e5eae78e49c58d61ee37d133fbf6`
- topic: `learning in the age of AI`
- cards in run: `29`
- abstract-backed cards in run: `10`
- abstract-backed card share: `34.48%`

In sampled papers from this run, the prompt window was dominated by title/authors/front matter/abstract, while body sections containing the real mechanism or contribution appeared only after the section cutoff.

### Why This Is A Problem

- It breaks the core project promise of extracting `aha moments` from the paper itself rather than from the paper's self-description.
- It prevents cards from revealing the specific thing that should change the learner's mind.
- It makes cards feel interchangeable across papers because they capture framing, not contribution.
- It weakens boss/reviewer decision-making because the cards do not expose the real object that would later become a course artifact, framework, exercise, or story.

### Suspected Root Cause

- The current section feeding strategy truncates too early and often excludes the body sections where the paper's real mechanism or result lives.
- Parser output loses structural distinctions, so the system cannot reliably separate abstract/front matter from model/method/result sections.
- Extraction prompt rules describe desired `aha` shape, but do not force the model to identify the paper's distinctive object before writing the card.
- Judgement prompt validates card plausibility and teachability, but cannot recover paper-specific value if extraction never exposed it.
- Review currently happens too late to fix this systematically; by review time the cards are already grounded in the wrong material.

### Desired Direction

- Make `paper-specific value capture` a first-class requirement in extraction, judgement, review, and evaluation.
- Require every card to identify the concrete object it is teaching: model, mechanism, framework, empirical result, experimental comparison, failure mode, or operational recipe.
- Reject cards that are paper-framing-only even if they sound teachable.
- Evaluate content quality not only by readability and bilingual evidence completeness, but also by whether the card surfaces the paper's unique contribution.

