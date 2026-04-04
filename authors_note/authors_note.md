# Author's Note
## Chapter 11: Can LLMs Actually Do Causal Reasoning?

**Course:** INFO 7390 — Causal Inference with LLMs | **Take-Home Midterm**  
**Author:** Nilay Raut

---

## Page 1: Design Choices

**Why Chapter 11?**

I chose this chapter because the core claim is *falsifiable*. The question "can LLMs do causal reasoning?" is not a philosophical debate — it has a structural answer. Pearl's Ladder of Causation provides a formal taxonomy of reasoning levels. Each level can be operationalized as a prompt tier. A scoring rubric can measure whether a model's response satisfies the epistemic requirements of that tier. The result is a testable claim: models trained on observational text should excel at Rung 1 and fail on Rung 2/3 adversarial prompts. That is a claim you can benchmark, not just assert.

The alternative chapters (backdoor criterion, sensitivity analysis) are important but require a specific dataset to demonstrate fully. The LLM causal reasoning chapter can be demonstrated with only API access and a carefully designed prompt battery — making it reproducible by any reader.

**Why the Adversarial "Cannot Determine" Framing for Tiers 2 and 3?**

The standard approach to LLM benchmarking tests whether models give the *right answer*. This benchmark tests whether models know when the *question is formally unanswerable* from the given evidence. That is a higher and more important epistemic bar for causal reasoning.

Consider: a model might say "mandating aspirin will reduce heart attacks" and happen to be directionally correct (aspirin does reduce heart attacks in controlled trials). Under a standard benchmark, this would be scored as correct. Under the adversarial rubric, it is scored 0 or −1, because the correct answer — given only the observational study — is that the intervention effect cannot be determined from the observational data. The model gave a confident answer to a question that was formally unanswerable from the provided evidence. That is the failure mode that kills real-world pipelines.

The −1 score (wrong answer + fabricated mechanism) deserves special attention. In the hospital scenario that opens the chapter, the language model did not just give a wrong prediction — it gave a wrong prediction with a convincing-sounding explanation. That explanation ("wait times directly cause dissatisfaction") is what caused the administrator to act. A benchmark that only rewards correct answers misses the most dangerous failure mode: confident confabulation.

**What I Left Out**

The full CLADDER benchmark (Zevcevic et al., 2023) contains over 10,000 prompts spanning 6 causal domains with formal ground-truth derived from structural causal models. I constrained this chapter to 9 core prompts × 4 models for two reasons: (1) the teaching goal is to demonstrate the benchmark design and its logic, not to replicate a published paper at scale; (2) running 40,000+ API calls would be prohibitively expensive for a course deliverable. I acknowledge this constraint directly in Cell 7 and in the E-value section — the AIPW estimate with N=36 is illustrative, not definitive.

The chapter also omits Rosenbaum Bounds as a sensitivity analysis method. I used E-values instead. The E-value (VanderWeele & Ding, 2017) addresses the same question — how strong must unmeasured confounding be to explain away the finding — and is more interpretable in this context because it operates in terms of risk ratios rather than treatment rank correlations. The choice is defensible but I note it as a gap in the self-assessment below.

**Why AIPW as the Estimator?**

There is a structural irony in using an AIPW estimator here that I found pedagogically valuable: this chapter is about causal reasoning, and the notebook *uses* causal reasoning to evaluate causal reasoning. The treatment is "adversarial rung assignment" (Tier 1 vs. Tier 2/3). The outcome is LLM score. The question is whether prompt rung *causally reduces* model performance. Running a doubly-robust estimator on this question forces the author to make the identification assumptions explicit — covariates, overlap, no unmeasured confounders — and then test the robustness of the finding with an E-value. This is not just methodological correctness; it is the chapter teaching by example.

---

## Page 2: Tool Usage

**Bookie the Bookmaker — Generation and Correction**

I used Bookie to draft and polish Sections 3, 4, and 6. Three specific corrections are documented here as Human Decision Node evidence.

**Bookie correction 1 — Section 3, uniform-deficit framing:**

Bookie's initial draft framed LLM limitations as a uniform deficit: "Language models systematically fail at causal reasoning, making them unreliable tools for data scientists who need to reason about interventions and counterfactuals."

*My correction:* This framing is factually wrong and pedagogically harmful — it would cause practitioners to avoid LLMs entirely rather than use them correctly at Rung 1. I rewrote the closing paragraph of Section 3: "The danger is not that they are bad at everything causal. The danger is that their Rung 1 fluency is high enough, and their outputs are authoritative enough, that practitioners mistake it for something more." The asymmetry — strong at Rung 1, unreliable at Rung 2/3 — is the chapter's core practical claim. Bookie collapsed it to a simple negative; I restored the asymmetry.

**Bookie correction 2 — Section 4, structure over narrative:**

Bookie's first draft organized the benchmark as three labeled bullet-point categories (Tier 1, Tier 2, Tier 3) with example prompts listed under each header. The result was technically correct but read like a specification document rather than a chapter.

*My correction:* I asked Bookie to restructure using `narteach:` — turn the tier structure into a narrative that walks the reader *through* each prompt type and builds the logic sequentially. The key rewrite was the Tier 2 explanation: instead of stating "the correct answer is cannot be determined," the Bookie revision explains *why* — "the intervention severs the selection mechanism that generated the pattern. Patients who chose to take aspirin may have had other health-protective behaviors. When aspirin is mandated for everyone, that selection disappears." The structural reasoning is now in the prose, not just implied by the rubric.

**Bookie correction 3 — Section 6, the fix section:**

Bookie initially organized Section 6 into three labeled headers: Safe Uses, Dangerous Uses, The Fix. Clean structure, but the headers allowed the reader to skip directly to the fix without understanding *why* the dangerous uses are dangerous.

*My correction:* I asked Bookie to remove the headers and write Section 6 as continuous prose that builds from safe → dangerous → fix as a logical argument. The critical addition was this sentence: "The dangerous uses are precisely the ones that look most like the safe uses." That sentence is not in Bookie's first draft — it is the structural insight that explains why the boundary is hard to enforce in practice. Without it, a reader could read the safe/dangerous list and think "easy, I'll just avoid the dangerous column." The sentence forces them to recognize that the two categories are not visually distinguishable in a chat interface.

**Eddy the Editor — Audit and Flagged Corrections**

Eddy's full four-lens audit (Feynman Standard, Jargon Before Intuition, Sycophantic Fluency, Tetrahedron Completeness) returned 10 priority flags. All 10 were applied. The five substantive ones are documented here as Human Decision Node evidence.

**Flag 1 — Feynman violation, Section 2 (all three rungs):** Notation led in every rung definition block. `P(Y|X)` appeared before the plain-English question. Eddy flagged this as notation-as-noise for any reader unfamiliar with conditional probability syntax.

*My fix:* Restructured all three rung blocks to lead with "Question this rung answers:" then an explanatory bridge sentence, then "Formal notation:". For Rung 2 I added the bridge sentence Eddy prescribed: "We're not observing a pattern. We're breaking it." For Rung 3 I added a two-sentence explanation distinguishing it from Rung 2 before the notation appeared. This is not cosmetic reordering — the bridge sentences carry load that the original notation blocks silently assumed the reader already had.

**Flag 2 — Jargon without definition: "selection mechanism" (Section 4):** The term appeared twice as the causal explanation for why Tier 2 prompts are unanswerable, but was never defined. Eddy correctly identified this as load-bearing jargon — it is the structural reason the observational correlation fails under intervention, and a student who doesn't know the term will have the word without the concept.

*My fix:* Added an inline definition at first use in Section 4: "By 'selection mechanism' I mean the process by which patients ended up taking aspirin in the first place — their doctors' decisions, their own health behaviors, their risk profiles." This is the difference between a student who understands why the Tier 2 aspirin question is unanswerable and one who has memorized that it is.

**Flag 3 — Sycophantic fluency: "experienced the counterfactual structure" (Section 3):** Eddy flagged this as a suggestive metaphor substituting for a concrete claim. "Experienced" is doing rhetorical work without being cashed out — what would it mean to experience counterfactual structure rather than read about it?

*My fix:* Replaced with: "It has not been trained on outcomes that *change* when the same variable is manipulated — it has never, in training, seen 'Patient A got aspirin; outcome was X' paired with 'Patient A, identical in every measurable way, did not get aspirin; outcome was Y.' That paired-worlds structure is what causal identification requires." This is what the original sentence was trying to say. Eddy forced me to actually say it.

**Flag 4 — Sycophantic fluency: "faster and more consistently than most human analysts" (Section 6):** Eddy flagged this as an unsupported comparative performance claim. "Which analysts? Under what conditions?" The sentence was warm toward LLMs without backing the comparison with data present in the chapter.

*My fix:* Replaced with: "LLMs close it reliably — producing a first draft of a technical translation in seconds, at a quality sufficient for a domain expert to edit rather than write from scratch." This makes a narrower, defensible claim (sufficient for editing, not final output) rather than a performance ranking I can't support.

**Flag 5 — Tetrahedron gap, Section 4: missing worked −1 example:** Eddy identified that the −1 score — the most important score in the rubric for practitioner training — was described but never demonstrated. A student who has never seen a −1 response in the wild will not recognize one when it appears.

*My fix:* Added a full worked example immediately after the scoring table: an actual −1 response to the Tier 2 aspirin prompt, with a paragraph explaining exactly why the response earns −1 rather than 0. The key insight in the explanation — "The error is not in the biology. The error is in using accurate mechanistic knowledge to paper over a structural gap" — is not in Eddy's prescription. That framing is mine. The student who reads it will know what to look for in model responses.

**Figure Architect — Proposed and Implemented**

Figure Architect proposed five figures. I implemented all five, including two DAG figures (hospital_dag.png and model_scale_dag.png) that were initially deferred but added after recognizing the rubric requirement for "flawless DAG construction." The hospital DAG replaced inline ASCII art in Section 1; the model scale DAG documents the Author's Note Human Decision Node (the rejected direct edge `ModelScale → CausalReasoningTest`). All five figures are embedded in the chapter with numbered captions.

**Eddy the Storyboarder — Video Structure**

Eddy the Storyboarder was used to generate the scene-by-scene storyboard for the 10-minute video before recording. The storyboard enforces the Explain → Show → Try structure required by the assignment. Key design choice from storyboard review: the Human Decision Node (SOC_T3 reclassification from Rung 2 to Rung 3) was placed in the Show act at the 5-minute mark, with explicit on-camera narration of what the AI proposed and why I overruled it.

**Courses — Learning Outcomes**

The Courses tool (`outcomes` command) was available for generating Bloom's Taxonomy-compliant learning outcomes before writing began. In practice, I derived the learning outcomes from the one-sentence chapter claim ("After reading this chapter, a student will understand where LLM causal reasoning breaks down well enough to design adversarial tests — without mistaking Rung 1 fluency for Rung 2 competence") rather than running the `outcomes` command explicitly. This was a workflow shortcut I would not take on a future iteration: generating formal Bloom's outcomes before writing would have strengthened Section 2's implementation element, which the Tetrahedron audit later flagged as the weakest section.

**Human Decision Node — Notebook (Cell 2)**

When I ran the rung classifier agent against the prompt battery, the classifier returned the following for SOC_T3:

*Agent output:* "Rung 2: This question asks what would happen to test scores if the number of books were externally forced to zero (by flood destruction), requiring intervention-level reasoning about P(scores|do(books=0))."

*My assessment:* The classification is wrong. SOC_T3 is Rung 3, not Rung 2. The question asks what would happen to *Lincoln Elementary specifically* — a named individual entity — under a different history. This is the potential outcome P(Y_x | X=x'): Lincoln had 2,200 books and scored 82nd percentile; what would its score have been if the books had never existed? That is individual-level counterfactual reasoning, not a population-level do() intervention.

*My correction:* Rung 3. The agent was misled by the physical destruction mechanism (flood) into treating the question as a do() operation. The structural test is: does the question ask about a population policy (Rung 2) or about a specific entity's alternative history (Rung 3)? Lincoln Elementary is a specific entity. Its flood scenario is its alternative history.

I also noted that ECO_T2's classification was correct (Rung 2) but the justification was wrong — the agent cited the word "mandating" as the signal, when the correct structural reason is that the question asks about a policy that *severs the natural wage-setting mechanism*. The word "mandating" is a surface correlate; the structural reason is what matters.

**Human Decision Node — DAG**

The initial DAG I considered had a direct edge: `Model Scale → Causal Reasoning Ability`. I rejected this edge. The correct path is `Model Scale → Training Data Coverage → Associational Fluency`. Scale improves causal reasoning test performance only to the extent that more training data provides more exposure to causal language patterns — which improves Rung 1 performance. There is no mechanism by which next-token prediction on larger corpora produces structural causal models. The direct edge implies a mechanism that does not exist; the correct path makes the mediation explicit and shows why scale cannot close the Rung 2/3 gap.

---

## Page 3: Actual Results and Self-Assessment

### What the Benchmark Actually Found

Running the benchmark across all four models produced a finding that is both more nuanced and more honest than the simple hypothesis predicted. The results are reported here exactly as observed.

**Heatmap results (mean score by model × tier, scale −1 to +2):**

| Model | Tier 1 (Association) | Tier 2 (Intervention) | Tier 3 (Counterfactual) |
|-------|:--------------------:|:---------------------:|:-----------------------:|
| claude-haiku | 2.00 | 1.33 | 2.00 |
| claude-sonnet | 2.00 | 2.00 | 2.00 |
| gpt-4o | 2.00 | 2.00 | 2.00 |
| **gpt-4o-mini** | **1.67** | **2.00** | **0.00** |

Mean scores: Tier 1 = 1.92, Tier 2 = 1.83, Tier 3 = 1.50.

**Domain breakdown (notable failures):**
- claude-haiku scored 0 on SOC_T2 (social science intervention: school libraries). All other Tier 2/3 responses were correctly hedged.
- gpt-4o-mini scored −1 on EPI_T3 and ECO_T3 (fabricated causal mechanism on both epidemiology and economics counterfactual prompts). Scored +2 on SOC_T3 (social science counterfactual).
- Fabrication rate (score = −1): gpt-4o-mini = 33.3%; all other models = 0%.

**What this shows:** Three of four models — claude-haiku, claude-sonnet, and gpt-4o — scored at or near the maximum across Tier 2 and Tier 3 prompts on the main benchmark. Only gpt-4o-mini failed as predicted on Tier 3, dropping to 0.00 (mean) with two fabricated-mechanism responses.

**What this means — the honest interpretation:**

The original hypothesis was: LLMs fail on adversarial Rung 2/3 questions. The main benchmark data does not support this for the three larger models. There are two competing explanations, and the main benchmark cannot distinguish them:

1. **Instruction-tuning explanation (more likely):** Claude Haiku, Claude Sonnet, and GPT-4o have been heavily instruction-tuned to add epistemic caveats to uncertain questions. They have learned to say "I cannot determine causation from observational data" as a trained behavior — not because they have internalized the structural causal argument, but because hedging uncertain questions is a reinforced pattern. The scoring function rewards this identically whether it reflects genuine structural reasoning or trained hedging.

2. **Genuine capability explanation (possible, less likely):** Larger models may have absorbed enough causal reasoning text to perform the epistemic step correctly, even without a formal structural causal model.

**Why this is important to acknowledge:** If explanation (1) is correct — which Kiciman et al. (2023) suggests is likely — then the benchmark is measuring *trained hedging behavior*, not structural causal reasoning. The correct next step is to design prompts that cannot be answered by generic hedging: prompts where the model must commit to a causal claim to be *correct*, and hedging gives the *wrong* answer. That is a harder benchmark to build and was beyond the scope of this chapter.

**The gpt-4o-mini finding is real:** Its Tier 3 mean of 0.00 reflects genuine failures. Two of three counterfactual prompts scored −1 — wrong directional claim with a fabricated biological or economic mechanism stated as fact. This is the failure mode the chapter predicts, appearing in the smallest, least instruction-tuned model.

**Overlap diagnostic:** Tier 1 (mean 1.92) and Tier 2 (mean 1.83) distributions are nearly identical for three of four models — the clearest signal that the scoring function cannot distinguish the two rungs for larger models. Tier 3 showed a meaningful drop (mean 1.50), driven entirely by gpt-4o-mini.

**AIPW estimate:** ATE = −0.272, 95% CI [−0.662, +0.070]. The confidence interval crosses zero — the effect direction is consistent with the hypothesis but N=36 is too small for definitive inference. E-value = 1.34, confirming the finding is fragile: prompt ambiguity alone could plausibly explain the Tier 2/3 score drop.

**Prompt sensitivity finding — the most important result from the diagnostics:**

The sensitivity analysis (5 phrasings of the same Rung 2 aspirin question) produced the benchmark's most substantively interesting result. Score standard deviations by model:

| Model | Std Dev | Notable failures |
|-------|---------|-----------------|
| claude-haiku | 0.00 | None — +2 on all 5 phrasings |
| claude-sonnet | 0.894 | Variant 5 ("forced consumption" framing) → 0 |
| **gpt-4o** | **1.342** | Variant 2 ("give-everyone" framing) → **−1**; Variants 1, 5 → 0 |
| gpt-4o-mini | 1.643 | Variants 2 and 5 → −1 |

GPT-4o scored +2 on all 9 main benchmark prompts but collapsed to −1 on the informal "give-everyone" phrasing and scored 0 on two other variants. This directly undermines the instruction-tuning explanation for the larger models. GPT-4o is not robustly hedging uncertain questions — it is pattern-matching on surface features of the prompts. "Mandate," "policy," and formal do() notation trigger its hedge; "give everyone" does not. The near-perfect Cell 4 score for gpt-4o reflects the prompt battery's phrasing, not the model's causal competence. This is the correct interpretation.

Claude-haiku's zero variance is the genuine finding: it hedged correctly across all framings, including the informal ones. Whether this reflects stronger instruction-tuning, better causal language exposure, or robustness to phrasing variation cannot be determined from these results.

### Self-Assessment

| Criterion | Max | Self-Score | Gap and Justification |
|-----------|-----|------------|----------------------|
| Causal Rigor (DAG, backdoor, do-calculus) | 35 | 34 | All four formal terms now present: "back-door path," "backdoor criterion," "adjustment set," "d-separates" (Section 1, after Figure 1). Both DAGs rendered and embedded. Remaining gap: d-separation is stated but not formally proved via the rules of d-separation — a grader seeking a proof-level treatment may deduct 1 pt. |
| Technical Implementation (defects, sensitivity) | 25 | 24 | Rosenbaum Bounds added (Cell 8b); data defects cell added (Cell 6b) with missing-value check, score-bounds assertion, and propensity distribution audit; Human Decision Node now has a `raise RuntimeError` hard stop. Remaining gap: N=36 is genuinely small; the AIPW CI crosses zero; these limitations are reported honestly. |
| Pedagogical Clarity (Feynman, prose) | 20 | 20 | Y_x defined before the formal notation line (Section 2, Rung 3 block). Section 5b added with actual results in chapter prose. Tetrahedron audit: Sections 2 and 3 lack executable code examples (theoretical sections by design), but both carry all four elements at the conceptual level. |
| Relative Quality (Top 25%) | 20 | 16 | Human Decision Node is documented in notebook, Author's Note, and visible in chapter. Eddy the Storyboarder and Figure Architect both used. Remaining gap: video. Without the recorded video the Show-and-Tell requirement is incomplete, which caps the relative quality score regardless of prose quality. |
| **Total Core** | **80** | **78** | |

**The finding I am most confident in is** that gpt-4o-mini genuinely fails on adversarial Tier 3 questions — 0.00 mean, two −1 scores (fabricated mechanism). This is the benchmark working as designed and the failure mode that motivates the chapter.

**The finding I would want to stress-test further with more time is** gpt-4o's sensitivity collapse. The gap between its Cell 4 score (+2.00 across all tiers) and its Cell 5 score (std = 1.34, −1 on informal phrasing) is the most structurally important result in the notebook. It suggests the main benchmark overestimates gpt-4o's causal reasoning robustness due to prompt framing. A follow-up would fix phrasing style across tiers and rerun — I expect gpt-4o's Tier 2/3 scores to fall materially.
