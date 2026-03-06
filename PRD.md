# PRD: Paper to Bullet / Aha Moment Extraction System

## 1. Document Purpose

This document consolidates the current consensus on the product into an execution-ready PRD.

Its purpose is to define:

- the product objective
- the end-to-end workflow
- the core data objects
- the engineering architecture direction
- the phased delivery plan
- the detailed, traceable TODO list

This PRD is the working contract for future implementation. It is intentionally more operational than `CONCEPT.md`.

---

## 2. Product Goal

Build a system that continuously discovers, ingests, reads, evaluates, and exports high-value academic-paper-derived `aha moment` candidate cards for course design.

The system is not a generic paper summarizer.

The system must:

- extract evidence-backed, course-usable atomic cards from papers
- preserve the original evidence chain from paper text and figures
- support scalable batch processing across multiple topics in parallel
- support both search-driven ingestion and manually supplied local PDF ingestion
- support internal review through a simple list-style interface
- export curated output into Google Docs for the boss to read

The boss is not a user of the internal system UI.
The boss only consumes the final Google Doc output.

---

## 3. Product Principles

### 3.1 Atomic First

Cards remain atomic and independent.
The system may cluster cards by semantic proximity, but must not automatically merge them into one synthesized conclusion.

### 3.2 Evidence First

Every accepted or reviewable card must be traceable back to:

- one or more original paper paragraphs
- optional figures / tables
- caption and source context

### 3.2.1 Aha First, Not Summary First

The system must optimize for `aha moment` extraction, not generic paper summarization.

A valid card is not merely:

- a paper takeaway
- an abstract restatement
- a literature background note
- a concept glossary item

A valid card must instead expose a meaningful gap between:

- what the target learner would normally assume
- what the paper evidence implies instead

If the system can only produce a correct summary but cannot articulate the learner-facing insight, the correct behavior is to emit no card.

### 3.3 Access Is Part of the Workflow

The system must distinguish between:

- paper discovered
- metadata retrieved
- abstract retrieved
- full text accessible
- full text unavailable and needs manual acquisition

Lack of access is a first-class operational state, not an edge case.

### 3.4 Structured Middle Layer

The system must not rely on raw prompts and Google Docs alone.

It must maintain a structured internal middle layer composed of:

- a paper evidence store
- a card judgement ledger

Google Docs is the final rendering/export layer, not the source of truth.

### 3.5 Parallel by Default

The system must support:

- multiple topics submitted together
- concurrent paper retrieval and parsing
- concurrent candidate extraction
- concurrent judgement and clustering jobs

The user should be able to input multiple topics and start the whole workflow with one action.

---

## 4. Users

### 4.1 Primary Internal User

Research/operator who:

- inputs topics
- optionally uploads or references local PDFs
- monitors progress
- reviews candidate cards
- triggers export to Google Docs

### 4.2 Secondary Internal User

Research lead / project owner who:

- reviews intermediate output when needed
- decides whether judgement rules need calibration
- selects what goes into boss-facing docs

### 4.3 External Consumer

Boss / decision maker who:

- does not use the internal UI
- only reads the final exported Google Docs

---

## 5. In Scope

- topic-based paper discovery
- local PDF path ingestion
- access-state tracking
- PDF and HTML paper parsing
- paragraph / section segmentation
- figure extraction and caption association
- candidate card generation
- green / yellow / red judgement
- internal list-style review surface
- semantic neighbor lookup for dedupe assistance
- topic-level and run-level parallel execution
- Google Docs export
- run history and traceability

---

## 6. Out of Scope for Initial Versions

- a boss-facing standalone web portal
- automatic course outline generation
- fully autonomous stop/no-stop strategic decisions without reviewability
- complex visual graph UI for clusters
- full Google Scholar automation as a primary ingestion backbone
- replacing all human access acquisition efforts for paywalled papers

---

## 7. Core User Flows

### 7.1 Batch Topic Run

1. User enters one or more topics.
2. User clicks one button to start processing.
3. System creates one `run` and multiple parallel `topic jobs`.
4. For each topic, the system discovers papers from configured sources.
5. System separates papers into:
   - accessible now
   - promising but inaccessible
   - low-priority / filtered out
6. Accessible papers enter parsing and extraction.
7. Candidate cards are generated.
8. Judgement is applied.
9. Similar-card neighbors are retrieved.
10. Internal reviewer scans the list and marks decisions.
11. Selected cards are exported to Google Docs.

### 7.2 Manual Local PDF Ingestion

1. User provides a local PDF path.
2. User optionally tags the PDF with one or more topics.
3. System validates that the file exists and is readable.
4. System creates a paper record with source = local upload / local path.
5. System parses the PDF, extracts sections and figures, and runs the same card workflow.
6. Output appears in the same internal review list and can be exported to Google Docs.

This local-PDF path must be a standard entry point, not a hacky side path.

### 7.3 Internal Review to Boss Doc

1. Internal user opens the review list.
2. User filters by topic / run / color / status / source.
3. User inspects card evidence, figure, judgement reason, and neighbor cards.
4. User marks cards as:
   - accepted for export
   - rejected
   - keep for later
   - needs manual evidence check
5. User selects cards and triggers Google Docs export.
6. System creates or updates a Google Doc with the chosen structure.

---

## 8. Functional Requirements

### 8.1 Topic Input

- Support batch input of multiple topics in a single submission.
- Support plain text input, one topic per line.
- Support tagging each run with optional metadata.
- Support rerunning a topic without rerunning all topics.

### 8.1.1 Run Metadata Input

Run metadata is optional structured context attached to a run.

The system must:

- accept a JSON object as run metadata
- default the metadata input to a valid empty object: `{}`
- allow users to submit without editing the default
- reject invalid JSON before the run starts
- store the metadata together with the run for later traceability

Typical metadata may include:

- operator
- project
- note
- owner
- batch tag

### 8.2 Local PDF Path Input

- Support entering a local absolute file path.
- Support batch local PDF input.
- Support associating local PDFs with one or more topics.
- Support mixing local PDFs with search-discovered papers in the same run.
- Support validation errors for missing files, unreadable files, unsupported formats, and duplicate ingestion.

### 8.3 Paper Discovery

- Query multiple discovery sources, not a single provider.
- Recommended sources:
  - OpenAlex
  - Semantic Scholar
  - Crossref
  - arXiv
  - manual seed papers
- Track which source yielded which paper.
- Track confidence / relevance signal for each discovered paper.

### 8.4 Access Tracking

Each paper must have an explicit access state:

- `open_fulltext`
- `abstract_only`
- `metadata_only`
- `paywalled`
- `broken_link`
- `manual_needed`

The system must maintain a visible acquisition queue for papers that appear relevant but do not yet have accessible full text.

### 8.5 Parsing and Evidence Extraction

For accessible papers, the system must:

- store raw source file or HTML snapshot
- segment text into sections / paragraphs
- extract figures when possible
- capture figure captions and figure numbers
- link text evidence to nearby figures where possible

### 8.5.1 Parsing Quality Gate

Parsing quality is a hard gate, not a cosmetic enhancement.

The system must not generate candidate cards from corrupted, binary-like, or structurally invalid extracted text.

At minimum, the parser layer must detect and handle:

- PDF object-stream noise mistakenly treated as正文
- garbled character output
- extraction dominated by control characters or binary fragments
- placeholder strings such as PDF headers, object metadata, or obviously non-semantic tokens

If parsing quality fails, the system must:

- mark the paper as `parse_failed` or `quality_failed`
- store the failure reason
- keep the paper visible for reprocessing
- prevent candidate card generation for that paper

The system must prefer parser correctness over partial noisy output.

### 8.5.2 Card Safety Rule

The system must never emit a candidate card when:

- card title is unreadable
- evidence quote is unreadable
- the extracted text clearly comes from binary/PDF structure noise
- similarity results are being computed on corrupted text

In those cases, the correct behavior is to fail the paper, not to produce a low-quality card.

### 8.6 Candidate Card Generation

For each paper, the system may output:

- zero cards
- one card
- multiple cards

The system must not force a quota.
Cards should be generated at the appropriate granularity:

- whole framework / method
- partial sub-pattern
- detail-level evidence / data point

### 8.6.1 Content Calibration and Aha Quality Gate

This is an interrupt-priority requirement.

Even when parsing quality is acceptable, the system must block cards that are content-valid summaries but not real `aha moment` candidates.

At minimum, the card-generation layer must distinguish between:

- a genuine learner-facing insight
- a plain paper summary
- theoretical background or literature review context
- technical detail that is too far from course use
- policy or management recommendation aimed at the wrong audience

The system must prefer `0 cards` over summary-like cards that do not create a meaningful learner-facing cognitive shift.

### 8.6.2 Content Shape of a Good Card

Card content should follow the agreed operator-facing standard even if the stored transport format remains JSON.

A good card must contain:

- a title that names the insight rather than the paper topic
- original paper evidence as the primary body material
- very short interleaved analysis explaining why each quoted part matters
- an explicit `what this becomes in the course` statement
- a one-sentence teachable articulation of the idea

The system must avoid cards that read like:

- an abstract rewrite
- a generic conclusion paragraph
- a vague recommendation with no sharp insight
- a taxonomy recap
- a paper-level summary that lacks a specific teachable point

### 8.6.3 Excluded Content List

For each paper, the system should not only output candidate cards but also identify major paper content that was intentionally rejected.

The exclusion list should capture items such as:

- background theory that is already common knowledge for the target learner
- technical mechanism detail with weak course transferability
- content that is interesting academically but not teachable in the target course
- ideas that are valid but weaker than another stronger card from the same paper

Each excluded item should have a short rejection reason.

### 8.6.4 Calibration Corpus and Evaluation Loop

This is an interrupt-priority requirement.

The system must not rely on prompt wording and schema validation alone to hold the `aha` quality boundary.

It must introduce a formal calibration and evaluation layer composed of:

- positive examples that should produce valid cards
- negative examples that should not produce cards
- boundary-case examples that test subtle judgement edges

This calibration layer must become part of the system source of truth, not remain as external notes in documents.

At minimum, the system must support:

- versioned calibration sets
- stored examples with expected outcomes
- evaluation runs against a calibration set
- comparison of prompt / rubric / model changes against prior evaluation results

The goal is to prevent silent drift toward:

- summary-like cards
- weak-transfer cards
- wrong-audience cards
- over-rejection of true `aha` insights

### 8.6.5 Split Candidate Extraction from Judgement

The system should distinguish between:

- extracting candidate insights and excluded content from paper evidence
- judging whether each candidate truly qualifies as a card under the current rubric

These must be modeled as separate stages even if they are both LLM-backed.

This separation is necessary so the system can:

- tell whether a failure came from discovery/extraction or from judgement
- apply calibration examples directly to the judgement boundary
- re-judge previously extracted candidates without reparsing the paper
- compare prompt and rubric revisions without conflating them with extraction behavior

### 8.6.6 Excluded Content as First-Class Review Data

Excluded content must not be treated as a hidden debug artifact.

It must be reviewable operational data with:

- a label
- an exclusion type
- a short reason
- linked source evidence
- a visible place in the internal review workflow

Reviewers must be able to:

- inspect excluded items directly
- filter for excluded items only
- mark an exclusion as accepted
- reopen an exclusion for reconsideration
- promote an excluded item back into candidate review when the rejection appears wrong

### 8.7 Judgement

Each card must receive:

- a color status: green / yellow / red
- a reason summary
- a statement of what it becomes in the course
- a stored prompt / rubric version reference

Judgement must explicitly consider:

- whether the card contains a real learner-facing cognitive shift
- whether the idea can be named as a concrete course object
- whether the transfer distance to course use is acceptably short
- whether the evidence is strong enough for the strength of the claim
- whether the content is merely a summary, background, or classification

### 8.8 Neighbor Retrieval and Dedup Assistance

Each new card should retrieve semantically similar prior cards.

The purpose is to support:

- dedupe assistance
- edge-case review
- saturation tracking

This does not mean auto-merging cards.

### 8.9 Internal Review Surface

The review interface can be simple.
It does not need to be a rich exploratory app in the first version.

The minimum useful form is:

- a Google Sheet / Airtable-like list
- sortable and filterable
- drill-down available for evidence and figure preview
- ability to change review status
- ability to push selected cards to Google Docs export

### 8.10 Google Docs Export

The system must support exporting selected cards into Google Docs.

The Google Doc is the final boss-facing deliverable.

Export should support:

- document creation
- append to existing document
- title and section headings
- paper metadata block
- card blocks
- quoted evidence paragraphs
- optional figure insertion
- exclusion notes if desired

---

## 9. Non-Functional Requirements

### 9.1 Parallelism

Parallelism is mandatory, not optional.

The system must support:

- multiple topics processed concurrently
- multiple papers within a topic processed concurrently
- extraction and judgement queued concurrently with bounded workers

The system must also support safe retries without duplicating final card records.

### 9.2 Traceability

Every exported card must be traceable to:

- topic
- run
- paper
- paragraph(s)
- figure(s) if any
- model / rubric version
- internal review decision

### 9.3 Re-runnability

The system must allow rerunning only one stage where possible, for example:

- re-run discovery only
- re-run parsing only
- re-run judgement only
- re-export to Google Docs only

### 9.4 Observability

The system should expose:

- run status
- topic job status
- topic job current stage
- topic job stage start time
- topic job stage duration
- failed paper count
- parsing failures
- parsing quality failures
- access-blocked papers
- in-progress discovery count
- in-progress acquisition/download count
- in-progress parsing count
- in-progress card generation count
- export success / failure

The UI must make it obvious whether jobs are:

- truly queued
- actively running
- blocked on network fetch
- blocked on acquisition
- blocked on parsing
- failed but not yet surfaced clearly

### 9.4.1 Timeout and Stall Visibility

Long-running topic jobs must not appear as vague `running` states indefinitely.

The system must:

- assign timeout budgets to discovery and asset acquisition stages
- surface when a topic job exceeds expected time budget
- expose the last successful sub-step for each topic job
- distinguish `running`, `stalled`, `waiting_for_access`, and `failed`

### 9.5 Deterministic Records

Even if LLM output is probabilistic, the system must record:

- model name
- prompt version
- rubric version
- run timestamp

so later output can be explained and compared.

---

## 10. Proposed System Shape

### 10.1 Source of Truth

Recommended source of truth:

- relational database for structured records
- vector index for semantic nearest-neighbor lookup
- object storage for paper and figure assets

Practical implementation direction:

- `Postgres` as relational storage
- `pgvector` for embeddings / similarity lookup
- local disk or object storage for raw PDFs, HTML snapshots, extracted figures, parsed artifacts

### 10.2 Why This Structure

This structure is needed so the system can:

- keep evidence separate from judgement
- re-run judgement without reparsing the paper
- re-export docs without rerunning LLM extraction
- maintain an acquisition queue for inaccessible papers
- support semantic dedupe assistance efficiently

### 10.3 Minimal Core Entities

- `runs`
- `topics`
- `topic_runs`
- `papers`
- `paper_sources`
- `paper_sections`
- `figures`
- `candidate_cards`
- `card_evidence_links`
- `judgements`
- `review_decisions`
- `access_queue`
- `exports`

Optional later entities:

- `clusters`
- `saturation_metrics`
- `rubric_versions`
- `prompt_versions`

---

## 11. Data Model Requirements

### 11.1 `runs`

Represents one user-triggered execution batch.

Fields should include:

- run id
- created by
- created at
- run type
- input summary
- status

### 11.2 `topics`

Represents a normalized topic.

Fields should include:

- topic id
- topic name
- topic description
- created at

### 11.3 `topic_runs`

Represents one topic processed within one run.

Fields should include:

- topic run id
- run id
- topic id
- status
- start time
- end time
- counts summary

### 11.4 `papers`

Represents one paper regardless of whether it came from search or a local path.

Fields should include:

- paper id
- canonical title
- authors
- year
- DOI / arXiv id / external ids
- source type
- local path if applicable
- original URL if applicable
- access status
- ingestion status
- parse status

### 11.5 `paper_sections`

Represents section / paragraph evidence units.

Fields should include:

- section id
- paper id
- section order
- section title
- paragraph text
- page number if available
- embedding

### 11.6 `figures`

Represents extracted figure assets.

Fields should include:

- figure id
- paper id
- figure label
- caption
- storage path
- linked section ids if available

### 11.7 `candidate_cards`

Represents atomic aha-moment candidates.

Fields should include:

- card id
- paper id
- topic id
- title
- granularity level
- course transformation statement
- draft body
- current status

### 11.8 `judgements`

Represents the AI judgement pass.

Fields should include:

- judgement id
- card id
- color
- reason
- model version
- prompt version
- rubric version
- created at

### 11.9 `review_decisions`

Represents human handling of the card.

Fields should include:

- review id
- card id
- reviewer
- decision
- note
- decision time

### 11.10 `access_queue`

Represents papers that look relevant but need human follow-up.

Fields should include:

- queue id
- paper id
- reason
- priority
- owner
- status

### 11.11 `exports`

Represents Google Docs export jobs.

Fields should include:

- export id
- run id or selection id
- destination type
- Google Doc id
- export status
- created at
- completed at

---

## 12. Workflow Design

### 12.1 Stage A: Intake

Inputs:

- one or more topics
- optional local PDF paths

Outputs:

- run record
- topic job records
- local paper ingestion tasks

### 12.2 Stage B: Discovery

For each topic:

- query configured external discovery sources
- deduplicate paper candidates
- assign access state
- queue inaccessible but relevant papers into acquisition tracking

### 12.3 Stage C: Acquisition and Parse

For accessible papers:

- fetch or register source artifact
- parse structure
- segment text
- extract figures
- store parsed artifacts

### 12.4 Stage D: Candidate Extraction

Generate zero-to-many candidate cards using the extracted evidence.

### 12.5 Stage E: Judgement

Evaluate each candidate according to rubric and assign:

- green
- yellow
- red

### 12.6 Stage F: Neighbor Retrieval

Compute or retrieve embedding neighbors for each card and expose them for review assistance.

### 12.7 Stage G: Internal Review

Internal reviewer approves / rejects / defers cards.

### 12.8 Stage H: Google Docs Export

Selected cards are rendered into boss-facing Google Docs.

---

## 13. Parallel Execution Requirements

### 13.1 User-Facing Requirement

The user can submit multiple topics at once and click once to start the whole run.

### 13.2 System Behavior

The system creates a separate worker path per topic while sharing infrastructure.

At minimum, the following can happen in parallel:

- topic discovery jobs
- paper parsing jobs
- candidate extraction jobs
- judgement jobs

### 13.3 Concurrency Controls

The system must have bounded concurrency to avoid:

- API overrun
- LLM cost explosions
- duplicate processing
- database contention

### 13.4 Idempotency

Retries must not create duplicate:

- paper records
- section records
- card records
- export jobs

---

## 14. Internal Review List Requirements

The internal review surface should initially behave like a structured list, not a complex application.

Minimum fields visible in list view:

- topic
- paper title
- card title
- color
- what it becomes in the course
- review status
- access status
- export eligibility

Minimum drill-down details:

- evidence paragraphs
- figure preview
- judgement reason
- nearby similar cards
- rejection or review notes

Minimum actions:

- mark accepted
- mark rejected
- mark needs manual check
- mark keep for later
- export selected

---

## 15. Google Docs Export Requirements

### 15.1 Export Objective

Export is the final boss-facing deliverable layer.

### 15.2 Export Structure

A document should support the following layout:

- document title
- topic section
- paper block
- card block(s)
- quoted evidence paragraph(s)
- figure and caption if available
- one-line course transformation statement

### 15.3 Export Modes

Support at least:

- create a new document
- append to an existing document

### 15.4 Technical Direction

Google Docs export can be implemented using:

- direct Google Docs / Drive API calls
- optionally `googleworkspace/cli` as a practical tooling layer for prototyping or scripted export

The export mechanism is not the system source of truth.

### 15.5 LLM Provider Configuration

The system must support `.env`-style configuration for LLM provider settings.

At minimum, the card-generation and judgement pipeline should be able to select between:

- `openai_compatible`
- `anthropic`
- `gemini`

Provider selection must be explicit and traceable through configuration, not hidden in code.

---

## 16. Acceptance Criteria by Capability

### 16.1 Multi-Topic Run

Accepted when:

- user submits multiple topics in one action
- each topic gets its own tracked job
- jobs run concurrently
- failures in one topic do not kill all others

### 16.1.1 Run Metadata Default

Accepted when:

- the run form pre-fills metadata with a valid default JSON object
- the default value can be submitted without manual editing
- invalid metadata JSON is blocked before run creation
- the submitted metadata is stored with the run

### 16.2 Local PDF Ingestion

Accepted when:

- user provides one or more local PDF paths
- valid files enter the same pipeline as discovered papers
- invalid files generate visible errors
- output cards appear in the same review list

### 16.3 Access Queue

Accepted when:

- relevant but inaccessible papers are stored separately
- reviewer can see which papers need manual retrieval
- papers can later be reintroduced into parsing once obtained

### 16.3.1 Parsing Quality Gate

Accepted when:

- corrupted PDF extraction does not create candidate cards
- unreadable evidence is blocked before card generation
- affected papers are visible as parse-quality failures
- reviewer can distinguish extraction failure from simple no-result papers

### 16.3.2 Content Calibration Gate

Accepted when:

- the system can produce cards that read like learner-facing `aha` insights rather than generic paper summaries
- cards include explicit course transformation statements that are teachable, not merely descriptive
- the system can identify and store major non-card content from the same paper with short rejection reasons
- theory background, taxonomy recap, and weak-transfer technical detail are routinely rejected instead of being promoted into cards
- prompt or rubric changes can be evaluated against positive, negative, and boundary-case examples

Accepted only when the following are also true:

- calibration examples are stored as structured system records rather than living only in documents
- the system can distinguish extraction-stage output from judgement-stage output
- excluded content can be reviewed in the internal workflow as a first-class object
- prompt or rubric revisions are compared through explicit evaluation runs before being promoted

### 16.4 Review List

Accepted when:

- reviewer can filter and sort cards
- reviewer can inspect evidence and decision basis
- reviewer can mark final internal decision

### 16.5 Google Docs Export

Accepted when:

- selected cards can be exported into a readable Google Doc
- headings, card blocks, and evidence paragraphs render correctly
- export job status is recorded

---

## 17. Delivery Plan

## Phase 0: Foundation and Validation

Goal:

- prove the workflow shape on a small scale

Deliverables:

- run intake
- topic batching
- local PDF path ingestion
- paper storage basics
- section extraction basics
- candidate card generation
- simple judgement
- basic Google Doc export

Interrupt requirement added during Phase 0:

- content calibration of LLM-generated cards so the system outputs `aha moment` candidates instead of paper summaries
- the system architecture must be extended toward a real calibration corpus, split extraction/judgement stages, excluded-content review flow, and evaluation loop before card quality work is considered fundamentally solved

Not required yet:

- saturation metrics
- advanced clustering UI
- sophisticated calibration framework

## Phase 1: Operational MVP

Goal:

- make the system usable by internal research workflows

Deliverables:

- multi-source discovery
- access queue
- review list
- export selection controls
- stable run tracking
- neighbor retrieval support

## Phase 2: Scalable Research Workflow

Goal:

- support repeated high-volume runs with better control and traceability

Deliverables:

- bounded worker orchestration
- rerunnable stages
- richer review metadata
- better dedupe assistance
- more stable export templates

## Phase 3: Calibration and Saturation

Goal:

- improve judgement quality and know when discovery is flattening out

Deliverables:

- rubric versioning
- prompt versioning
- edge-case handling workflows
- saturation metrics
- search-strategy comparison

---

## 18. Detailed Traceable TODO List

The following TODOs are written as implementation work items with identifiers, dependencies, and expected outputs.

### Track A: Run Intake and Parallel Execution

- [ ] `A-001` Define the run object and topic-run object.
  Output: schema for `runs`, `topics`, `topic_runs`.
  Dependency: none.

- [ ] `A-002` Implement batch topic input parsing.
  Output: multiple topics can be submitted in one action.
  Dependency: `A-001`.

- [ ] `A-002A` Add safe default run metadata handling.
  Output: the UI pre-fills metadata with `{}`, invalid JSON is blocked, and valid metadata is stored with the run.
  Dependency: `A-001`.

- [ ] `A-003` Implement one-click run creation for all submitted topics.
  Output: one run record with multiple topic jobs.
  Dependency: `A-002`.

- [ ] `A-004` Implement concurrent processing per topic.
  Output: independent topic workers with status tracking.
  Dependency: `A-003`.

- [ ] `A-005` Add bounded concurrency and retry safety.
  Output: worker limits, idempotent job behavior.
  Dependency: `A-004`.

### Track B: Local PDF Ingestion

- [ ] `B-001` Define local PDF input contract.
  Output: accepted input format for one or more local absolute paths.
  Dependency: none.

- [ ] `B-002` Validate local path existence, readability, and PDF type.
  Output: clear error handling for bad paths.
  Dependency: `B-001`.

- [ ] `B-003` Create paper records from local PDFs.
  Output: local files enter `papers` with source type = local.
  Dependency: `B-002`.

- [ ] `B-004` Allow local PDFs to be tagged to one or more topics.
  Output: local-paper-to-topic linkage.
  Dependency: `B-003`.

- [ ] `B-005` Route local PDFs through the same parse/card/judgement/export pipeline.
  Output: no separate special-case workflow downstream.
  Dependency: `B-004`.

### Track C: Discovery and Access Management

- [ ] `C-001` Define external discovery provider interface.
  Output: a standard input/output contract for OpenAlex, Semantic Scholar, Crossref, arXiv, and manual seed ingestion.
  Dependency: none.

- [ ] `C-002` Implement topic discovery against multiple sources.
  Output: discovered paper candidates with source attribution.
  Dependency: `C-001`.

- [ ] `C-003` Deduplicate discovered paper candidates.
  Output: canonical paper candidate list.
  Dependency: `C-002`.

- [ ] `C-004` Define and store access status for each paper.
  Output: `open_fulltext`, `abstract_only`, `metadata_only`, `paywalled`, `broken_link`, `manual_needed`.
  Dependency: `C-003`.

- [ ] `C-005` Build an acquisition queue for relevant but inaccessible papers.
  Output: separate list for manual retrieval follow-up.
  Dependency: `C-004`.

- [ ] `C-006` Support reactivating a paper from the acquisition queue once full text is obtained.
  Output: inaccessible paper can later enter parse stage.
  Dependency: `C-005`.

### Track D: Parsing and Evidence Store

- [ ] `D-001` Define storage layout for raw paper files and parsed artifacts.
  Output: convention for PDFs, HTML snapshots, figures, parsed JSON, and text chunks.
  Dependency: none.

- [ ] `D-002` Implement parser selection for PDF vs HTML.
  Output: source-specific parse pipeline.
  Dependency: `D-001`.

- [ ] `D-003` Segment papers into sections / paragraphs.
  Output: `paper_sections` records.
  Dependency: `D-002`.

- [ ] `D-004` Extract figures and captions where possible.
  Output: `figures` records and stored image assets.
  Dependency: `D-002`.

- [ ] `D-005` Link evidence paragraphs to related figures when possible.
  Output: paragraph-figure associations.
  Dependency: `D-003`, `D-004`.

- [ ] `D-006` Record parse failures and partial-success states.
  Output: visible operational states instead of silent failure.
  Dependency: `D-002`.

- [ ] `D-007` Add parsing quality validation before section persistence.
  Output: noisy or binary-like extraction is rejected before downstream card generation.
  Dependency: `D-002`.

- [ ] `D-008` Add explicit parse quality failure states and reasons.
  Output: papers can fail with `parse_failed` or `quality_failed` and remain reprocessable.
  Dependency: `D-007`.

- [ ] `D-009` Prevent corrupted evidence from reaching candidate card generation.
  Output: unreadable titles, unreadable evidence, and PDF-structure noise never become cards.
  Dependency: `D-007`, `E-002`.

### Track E: Candidate Card Generation

- [ ] `E-001` Define the canonical card schema.
  Output: fields for title, evidence list, figure list, course transformation statement, and status.
  Dependency: none.

- [ ] `E-002` Implement zero-to-many card extraction from one paper.
  Output: a paper may yield 0, 1, or many candidate cards.
  Dependency: `D-003`, `E-001`.

- [ ] `E-002A` Add an aha-quality gate for card content.
  Output: summary-like, background-like, and weak-transfer content is rejected instead of becoming cards.
  Dependency: `E-002`.

- [ ] `E-003` Support granularity-aware extraction.
  Output: cards can represent framework-level, sub-pattern-level, or detail-level insights.
  Dependency: `E-002`.

- [ ] `E-003A` Add card-body shaping rules aligned with the agreed strong-example style.
  Output: title names the insight, evidence stays primary, analysis remains minimal, and each card contains a teachable one-line articulation.
  Dependency: `E-002`.

- [ ] `E-004` Store explicit links from cards to source evidence.
  Output: card-to-paragraph and card-to-figure traceability.
  Dependency: `E-002`.

- [ ] `E-004A` Store paper-level excluded-content records.
  Output: rejected-but-noteworthy content from a paper is persisted with short rejection reasons.
  Dependency: `E-002A`.

### Track F: Judgement and Review Basis

- [ ] `F-001` Define the judgement output contract.
  Output: color, reason, course transformation statement, model version, prompt version, rubric version.
  Dependency: `E-001`.

- [ ] `F-002` Implement initial green/yellow/red judgement.
  Output: each candidate card receives an initial status.
  Dependency: `E-002`, `F-001`.

- [ ] `F-003` Store judgement history instead of overwriting silently.
  Output: traceable judgement ledger.
  Dependency: `F-002`.

- [ ] `F-004` Separate AI judgement from human review decision.
  Output: `judgements` and `review_decisions` remain distinct.
  Dependency: `F-003`.

- [ ] `F-005` Build a calibration set from positive, negative, and boundary-case card examples.
  Output: reusable example bank for prompt and rubric iteration.
  Dependency: `F-001`.

- [ ] `F-006` Evaluate prompts against the calibration set before promoting changes.
  Output: prompt/rubric revisions are tested for summary drift, weak-transfer drift, and missed-aha regressions.
  Dependency: `F-005`.

- [ ] `F-007` Persist calibration sets and examples as formal system records.
  Output: versioned calibration corpora stored in the product, not only in external documents.
  Dependency: `F-005`.

- [ ] `F-008` Split candidate extraction from judgement into separate executable stages.
  Output: the system can tell whether failure or drift comes from extraction or from judgement.
  Dependency: `E-002A`, `F-001`.

- [ ] `F-009` Support re-judging previously extracted candidates without reparsing papers.
  Output: judgement improvements can be rolled out and compared without touching the evidence store.
  Dependency: `F-008`.

- [ ] `F-010` Record evaluation runs and per-example outcomes for prompt/rubric/model comparisons.
  Output: a durable evaluation ledger for calibration regressions and improvements.
  Dependency: `F-006`, `F-007`.

### Track G: Semantic Neighbor Retrieval

- [ ] `G-001` Choose embedding generation approach for cards and evidence units.
  Output: embedding strategy documented and implemented.
  Dependency: `E-001`.

- [ ] `G-002` Store embeddings for candidate cards.
  Output: vector-search-ready card records.
  Dependency: `G-001`, `E-002`.

- [ ] `G-003` Implement nearest-neighbor lookup for each new card.
  Output: top-k similar cards available for review assistance.
  Dependency: `G-002`.

- [ ] `G-004` Expose neighbor cards in review details.
  Output: reviewer can see likely duplicates / near variants.
  Dependency: `G-003`, `I-003`.

- [ ] `G-005` Keep cards atomic and do not auto-merge.
  Output: explicit enforcement of non-merge behavior.
  Dependency: `G-003`.

### Track H: Internal Review List

- [ ] `H-001` Define the minimal review list fields and filters.
  Output: spec for the internal list view.
  Dependency: `E-001`, `F-001`.

- [ ] `H-002` Implement a list-style internal review surface.
  Output: Google Sheet / Airtable-like review table behavior.
  Dependency: `H-001`.

- [ ] `H-003` Add drill-down detail for evidence, figures, judgement reason, and neighbor cards.
  Output: review detail panel / page.
  Dependency: `H-002`, `D-004`, `G-003`.

- [ ] `H-004` Add review actions.
  Output: accept, reject, keep for later, needs manual check.
  Dependency: `H-002`.

- [ ] `H-005` Persist review decisions.
  Output: `review_decisions` records linked to cards.
  Dependency: `H-004`.

- [ ] `H-006` Expose excluded content as a first-class review list object.
  Output: reviewers can view excluded items directly rather than only through card drill-down.
  Dependency: `E-004A`, `H-002`.

- [ ] `H-007` Add filters and batch actions for excluded content.
  Output: reviewers can filter `cards only`, `excluded only`, or `both`, and handle exclusions in bulk.
  Dependency: `H-006`.

- [ ] `H-008` Generalize review decisions to apply to both cards and excluded content.
  Output: one unified review ledger for accepted cards, confirmed exclusions, and reopened items.
  Dependency: `H-005`, `H-006`.

### Track I: Google Docs Export

- [ ] `I-001` Define the boss-facing Google Doc template.
  Output: section order, paper block format, card block format, evidence formatting.
  Dependency: none.

- [ ] `I-002` Define export selection rules.
  Output: only reviewed / eligible cards can be exported.
  Dependency: `H-004`.

- [ ] `I-003` Implement export payload assembly from structured records.
  Output: selected cards can be rendered without rerunning extraction.
  Dependency: `E-004`, `F-002`, `H-005`, `I-001`.

- [ ] `I-004` Implement Google Doc creation.
  Output: create new boss-facing document.
  Dependency: `I-003`.

- [ ] `I-005` Implement append/update export mode.
  Output: add selected cards into an existing Google Doc.
  Dependency: `I-004`.

- [ ] `I-006` Implement optional figure insertion.
  Output: figures and captions appear in exported docs when available.
  Dependency: `D-004`, `I-004`.

- [ ] `I-007` Record export job results.
  Output: export success/failure log with Google Doc id.
  Dependency: `I-004`.

### Track J: Observability and Operations

- [ ] `J-001` Define system statuses across stages.
  Output: standard lifecycle statuses for runs, topic jobs, papers, and exports.
  Dependency: `A-001`.

- [ ] `J-002` Add run progress summary.
  Output: counts for discovered, accessible, parsed, carded, judged, reviewed, exported.
  Dependency: `J-001`.

- [ ] `J-003` Add failure logging and retry hooks.
  Output: operational visibility for debugging.
  Dependency: `J-001`.

- [ ] `J-004` Expose current stage and elapsed time for each topic job.
  Output: the UI shows whether a topic is in discovery, acquisition, parsing, card generation, or export preparation.
  Dependency: `J-001`.

- [ ] `J-005` Distinguish `running` from `stalled` and `waiting_for_access`.
  Output: long-running topic jobs do not look ambiguously active.
  Dependency: `J-004`.

- [ ] `J-006` Add timeout budgets for discovery and remote asset acquisition.
  Output: per-topic work can fail or degrade predictably instead of hanging invisibly.
  Dependency: `C-002`, `J-004`.

- [ ] `J-007` Add calibration and evaluation observability.
  Output: visible status for active calibration set, last evaluation run, metric deltas, and failed examples.
  Dependency: `F-010`.

### Track K: Calibration and Saturation

- [ ] `K-001` Introduce versioned rubric records.
  Output: judgement criteria can evolve traceably.
  Dependency: `F-001`.

- [ ] `K-002` Introduce versioned prompt records.
  Output: prompt changes are auditable.
  Dependency: `F-001`.

- [ ] `K-003` Track category growth and semantic duplication over time.
  Output: early saturation metrics.
  Dependency: `G-003`, `A-003`.

- [ ] `K-004` Compare retrieval strategies by incremental yield.
  Output: evidence for whether additional search strategies still add value.
  Dependency: `C-002`, `K-003`.

- [ ] `K-005` Add support for boundary-case sets and later calibration workflows.
  Output: structured path toward improved judgement quality.
  Dependency: `F-003`, `H-005`.

---

## 19. Recommended Build Order

Implementation order should be:

1. `A` Run intake and parallel execution basics
2. `B` Local PDF ingestion
3. `D` Parsing and evidence store
4. `E` Candidate card generation, including the interrupt-priority aha-quality calibration work
5. `F` Judgement, including positive/negative/boundary-case calibration and split extraction/judgement stages
6. `H` Minimal internal review list, extended to first-class excluded-content review
7. `J` Observability, including calibration/evaluation visibility
8. `I` Google Docs export
9. `C` Multi-source discovery and access queue
10. `G` Semantic neighbor retrieval
11. `K` Calibration and saturation

Reason:

- this order gets the shortest path to a working end-to-end loop
- it validates the most important user value early
- it delays complexity that is useful but not necessary for first proof of usefulness

---

## 20. Open Questions

- Which parser stack should be used first for PDF and HTML?
- Should the first internal review surface be a lightweight web page or a spreadsheet-like embedded grid?
- What exact Google Doc layout does the boss prefer?
- Should figure export be required in Phase 0 or allowed to land in Phase 1?
- What is the first bounded concurrency target for topic jobs and paper jobs?

---

## 21. Definition of Done for the First Real Milestone

The first meaningful milestone is complete when:

- a user can submit multiple topics in one action
- a user can also submit one or more local PDF paths
- the system runs the topic and local-PDF flows in parallel
- at least accessible papers are parsed into evidence sections
- candidate cards are generated
- cards receive an initial judgement
- cards appear in a simple internal review list
- selected cards can be exported into a readable Google Doc
- inaccessible but relevant papers are kept in a visible acquisition queue

At that point, the system is operationally real, even if later calibration and saturation layers are still incomplete.
