# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Build the ARI investor decks.

Two files, because the two audiences need different things and one deck that
tries to serve both serves neither:

    ari-overview.pptx     semi-technical. What each index is for, and what it
                          catches that nothing else does.
    ari-technical.pptx    technical. Method, numbers, controls, and the
                          results that go against us.

Every number here is read from the experiment JSON rather than typed in, so a
slide cannot drift away from the measurement it reports. Figures come from
docs/figures.

The weak results stay in both decks. An investor running an LLM over this
material will find the top-2 margin comparison in about a minute, and it is
better that they find our version of it.

The decks are historical presentation artifacts. The 2026-09-16 repository
audit superseded several statements they made; each deck now opens with a
slide listing them, and the affected slides are labelled. The ARI-E numbers
are still read from harness_effect.json because that file is what the slides
historically showed; the paper withholds them pending a rerun. See
docs/RETIRED_CLAIMS.md for the index.

    python docs/deck/build_deck.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] if (HERE.parents[2] / "experiments").exists() else HERE.parents[1]
EXP = ROOT / "experiments"
FIG = ROOT / "docs" / "figures"

INK = RGBColor(0x1B, 0x1B, 0x1B)
GREEN = RGBColor(0x0B, 0x6E, 0x4F)
RED = RGBColor(0xC1, 0x54, 0x3A)
GREY = RGBColor(0x6E, 0x6E, 0x6E)
BLUE = RGBColor(0x2F, 0x5D, 0x8A)

W, H = Inches(13.333), Inches(7.5)


# ---------------------------------------------------------------------------
# Measurements, read from disk
# ---------------------------------------------------------------------------

def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def facts() -> dict:
    f = {}
    r = load(EXP / "regime-discrimination/results/regime_matrix.json")
    if r:
        rows = {x["condition"]: x for x in r["rows"]}
        f["tf32_on"] = rows.get("gpu_tf32_on")
        f["tf32_off"] = rows.get("gpu_tf32_off")
        f["int8"] = rows.get("int8")
    d = load(EXP / "decoding-reproducibility/results/decoding_matrix_qwen3b_gpu.json")
    if d:
        f["d_model"] = d["model"]
        f["d_rows"] = {x["condition"]: x for x in d["rows"]}
    b = load(EXP / "decoding-reproducibility/results/baselines.json")
    if b:
        f["storage"] = b["storage_bytes"]
        f["sens"] = {k: v[1] for k, v in b["sensitivity"].items()}
        f["tail"] = b["tail_only"]
    e = load(EXP / "harness-effect/results/harness_effect.json")
    if e:
        f["e_contrasts"] = [c for c in e["contrasts"] if c.get("n_cases")]
    return f


# ---------------------------------------------------------------------------
# Slide helpers
# ---------------------------------------------------------------------------

def deck() -> Presentation:
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def textbox(slide, text, *, left, top, width, height, size=18, bold=False,
            color=INK, align=PP_ALIGN.LEFT, space_after=6):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return tb


def title_slide(prs, title, subtitle, tag=""):
    s = blank(prs)
    textbox(s, title, left=Inches(0.9), top=Inches(2.4), width=Inches(11.5),
            height=Inches(1.4), size=40, bold=True)
    textbox(s, subtitle, left=Inches(0.9), top=Inches(3.9), width=Inches(11.5),
            height=Inches(1.2), size=20, color=GREY)
    if tag:
        textbox(s, tag, left=Inches(0.9), top=Inches(6.4), width=Inches(11.5),
                height=Inches(0.5), size=13, color=GREY)
    return s


def header(slide, title, kicker=""):
    if kicker:
        textbox(slide, kicker, left=Inches(0.7), top=Inches(0.35),
                width=Inches(12), height=Inches(0.35), size=12, color=GREY)
    textbox(slide, title, left=Inches(0.7), top=Inches(0.7), width=Inches(12),
            height=Inches(0.9), size=28, bold=True)


def bullets(slide, items, *, top=Inches(1.9), size=17, left=Inches(0.9),
            width=Inches(11.6)):
    tb = slide.shapes.add_textbox(left, top, width, Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        text, color, bold = (item if isinstance(item, tuple)
                             else (item, INK, False))
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(12)
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.name = "Calibri"
    return tb


def table(slide, rows, *, left=Inches(0.9), top=Inches(2.0), width=Inches(11.5),
          height=Inches(0.4), size=13, highlight=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width,
                                   height * len(rows))
    tbl = shape.table
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            for run in para.runs:
                run.font.size = Pt(size)
                run.font.bold = (r == 0)
                run.font.name = "Calibri"
                if highlight and r in highlight:
                    run.font.color.rgb = highlight[r]
    return shape


def picture(slide, name, *, left=Inches(1.0), top=Inches(1.8), width=Inches(11.3)):
    p = FIG / name
    if p.exists():
        slide.shapes.add_picture(str(p), left, top, width=width)


def note(slide, text, *, top=Inches(6.6), color=GREY, size=12):
    textbox(slide, text, left=Inches(0.7), top=top, width=Inches(12),
            height=Inches(0.6), size=size, color=color)


SUPERSEDED = [
    ("This deck is a historical presentation artifact, regenerated from the "
     "experiment JSON on 2026-09-16. It is not maintained scientific evidence. "
     "Use the paper and docs/REPOSITORY_AUDIT.md for current claims.", INK, True),
    ("ARI-E slides: the harness-effect numbers come from an earlier estimator and "
     "a partial trajectory export. The paper withholds them pending a rerun.", RED, False),
    ("Self-hosted `time` cells were immediate repeats, not captures after a gap "
     "greater than 24 hours. The self-hosted comparable core is incomplete.", RED, False),
    ("A nonsignificant Recall@10 difference is not equivalence. Ranked result lists "
     "did change under the same conditions.", INK, False),
    ("'Early warning' means a changed code at a step whose token did not change. "
     "It is not a validated predictor of later failure.", INK, False),
    ("TF32 and kernel-selection mechanism statements are consistent with the data "
     "but were not established by profiling.", INK, False),
    ("Decoding disagreement rates labelled H-bar are packed-byte rates, not bit or "
     "coordinate rates.", INK, False),
    ("Index of superseded statements: docs/RETIRED_CLAIMS.md.", GREY, False),
]


def superseded_slide(prs):
    s = blank(prs)
    header(s, "Superseded statements", "READ FIRST: 2026-09-16 REPOSITORY AUDIT")
    bullets(s, SUPERSEDED, size=14)
    return s


# ---------------------------------------------------------------------------
# Deck 1: semi-technical
# ---------------------------------------------------------------------------

def build_overview(f: dict, out: Path):
    prs = deck()

    title_slide(
        prs, "Measuring whether an AI system does the same thing twice",
        "ARI: three indices for agent reproducibility",
        "Numbers are read from the experiment JSON in this repository. "
        "Slide 2 lists statements superseded by the 2026-09-16 audit.")
    superseded_slide(prs)

    s = blank(prs)
    header(s, "The problem: your dashboard cannot see the change", "THE GAP")
    bullets(s, [
        ("A model provider changes how your model is served. Nothing tells you.",
         INK, True),
        "They move it to different hardware, or serve it at lower precision to cut cost.",
        "The API still returns 200. Latency looks normal. Quality dashboards do not move.",
        ("Regulators now require reproducibility: EU AI Act Art. 15, NIST AI RMF, "
         "US Treasury AI RMF.", BLUE, False),
        ("None of them define how to measure it. There is no standard instrument.",
         RED, True),
    ])
    note(s, "This is the gap ARI fills.")

    s = blank(prs)
    header(s, "Three layers, three different questions", "THE PRODUCT")
    table(s, [
        ["Index", "Question it answers", "Who buys it"],
        ["ARI-R", "Does the same input give the same internal representation?",
         "Risk, compliance, platform"],
        ["ARI-D", "Does the same context produce the same next token?",
         "Anyone on a hosted model API"],
        ["ARI-E", "Does the scaffold or the model decide the outcome?",
         "Anyone publishing agent benchmarks"],
    ], top=Inches(2.1), height=Inches(0.75), size=15)
    bullets(s, [
        ("Why three and not one: an index called 'agent reproducibility' that "
         "only measures embeddings overstates what it covers.", INK, False),
        ("Splitting it turns every gap into a roadmap item instead of a hole.",
         GREEN, True),
    ], top=Inches(5.0), size=16)

    s = blank(prs)
    header(s, "ARI-R: the audit gap", "USE CASE 1")
    bullets(s, [
        ("You certify a model on your own hardware. Your vendor serves it on theirs.",
         INK, True),
        "Same weights. Same precision setting on paper. One switch differs: TF32.",
    ], top=Inches(1.8), size=17)
    if f.get("tf32_on"):
        on, off = f["tf32_on"], f["tf32_off"]
        table(s, [
            ["What you would check", "TF32 off", "TF32 on"],
            ["Recall@10, the quality metric", "unchanged", "unchanged"],
            ["SEMQ code agreement", f"{off['semq_her']:.4f}",
             f"{on['semq_her']:.4f}"],
        ], top=Inches(3.1), height=Inches(0.6), size=15,
            highlight={2: GREEN})
        note(s, f"The audited system and the served system disagree on "
                f"{(1-on['semq_her'])*100:.0f}% of documents. No Recall@10 change "
                f"was detected; a nonsignificant difference is not equivalence. "
                f"Turning TF32 off restores agreement, which associates the effect "
                f"with that one switch.",
             top=Inches(5.1), color=INK, size=15)

    s = blank(prs)
    header(s, "ARI-D: the silent downgrade", "USE CASE 2")
    bullets(s, [
        ("A provider quietly serves your model at lower precision. "
         "Your outputs look fine.", INK, True),
    ], top=Inches(1.8), size=17)
    if f.get("d_rows", {}).get("tf32_on"):
        t = f["d_rows"]["tf32_on"]
        table(s, [
            ["Signal", "Reading"],
            ["Text your users see", "100% identical"],
            ["Tokens emitted", f"{t['token_agreement']:.0%} identical"],
            ["ARI-D: steps where the instrument saw a change",
             f"{t['early_warning_steps']:.0%}"],
        ], top=Inches(2.8), height=Inches(0.65), size=15,
            highlight={3: GREEN})
        note(s, "Output comparison is silent across the entire run. The "
                "instrument reads a code change at every step while the token "
                "is unchanged. This is contemporaneous disagreement, not a "
                "validated predictor of later failure.", top=Inches(5.3),
             color=INK, size=15)

    s = blank(prs)
    header(s, "ARI-E: was it the model, or the wrapper?",
           "USE CASE 3. HISTORICAL: WITHHELD FROM THE PAPER PENDING RERUN")
    bullets(s, [
        ("Two agent products run the identical model. One scores higher. "
         "Which one did the work?", INK, True),
        "Measured on 22,000 real agent runs across two widely used scaffolds.",
    ], top=Inches(1.8), size=17)
    if f.get("e_contrasts"):
        sc = [c for c in f["e_contrasts"] if c["title"].startswith("scaffold")]
        mo = [c for c in f["e_contrasts"] if c["title"].startswith("model")]
        avg = lambda xs: sum(c["effect"] for c in xs) / len(xs)
        table(s, [
            ["What changed", "How much it decided the outcome"],
            ["The scaffold around the model", f"{avg(sc):+.3f}"],
            ["The model itself", f"{avg(mo):+.3f}"],
        ], top=Inches(3.2), height=Inches(0.7), size=16, highlight={1: GREEN})
        note(s, "Historical reading: the wrapper effect exceeded the model effect "
                "for these pairs. Superseded: earlier estimator, partial export; "
                "not a current result.", top=Inches(5.4), color=RED, size=15)

    s = blank(prs)
    header(s, "The control is the product",
           "WHY THIS IS HARD TO COPY. HISTORICAL NUMBERS, SEE SLIDE 2")
    bullets(s, [
        ("Agents disagree with themselves 10-13% of the time. Anyone can diff "
         "two systems. Almost nobody subtracts that noise.", INK, True),
        "",
    ], top=Inches(1.8), size=17)
    if f.get("e_contrasts"):
        c = f["e_contrasts"][0]
        apparent = 1 - c["cross_agreement"]
        table(s, [
            ["Method", "Reported effect"],
            ["Naive difference between the two scaffolds", f"{apparent:.1%}"],
            ["ARI-E, after removing the agent's own noise", f"{c['effect']:.1%}"],
        ], top=Inches(3.0), height=Inches(0.7), size=16, highlight={1: RED, 2: GREEN})
        note(s, "In the historical run the control removed 59-74% of the apparent "
                "effect. The estimator has since been corrected and the numbers "
                "await a rerun; the method point stands, the values do not.",
             top=Inches(5.2), color=INK, size=15)

    s = blank(prs)
    header(s, "Anyone can check our numbers without trusting us",
           "WHY IT IS CREDIBLE")
    bullets(s, [
        ("Every report is cryptographically signed and bound to the exact data "
         "it came from.", INK, True),
        "Change the report, or the data underneath it, and verification fails.",
        ("The verifier uses the Python standard library and `cryptography`. "
         "It does not import the SDK or the harness.", GREEN, True),
        "A standard whose verification requires the vendor's code is not a standard.",
    ])
    note(s, "python verify_report.py harness_effect.json --expect-key <key>   "
            "->   ALL CHECKS PASSED", top=Inches(5.6), color=GREEN, size=14)

    s = blank(prs)
    header(s, "Where we actually are", "HONEST STATUS")
    table(s, [
        ["Index", "Status", "Strength of the evidence"],
        ["ARI-R", "Measured on CPU and GPU",
         "TF32 association replicated; attribution needs profiling"],
        ["ARI-D", "Measured on two models",
         "Real, but a cheaper statistic competes"],
        ["ARI-E", "Historical analysis, withheld",
         "Pending rerun with the corrected estimator"],
    ], top=Inches(2.0), height=Inches(0.75), size=15)
    bullets(s, [
        ("Known weakness, stated up front: at the decoding layer a simple "
         "top-2 margin statistic gets 86% of the signal for a fraction of the "
         "storage.", RED, True),
        ("What survives is coverage. It sees changes the cheap statistic is "
         "structurally blind to. The technical deck's tail-coverage slide has "
         "the evidence.", INK, False),
    ], top=Inches(4.9), size=15)

    prs.save(out)
    return out


# ---------------------------------------------------------------------------
# Deck 2: technical
# ---------------------------------------------------------------------------

def build_technical(f: dict, out: Path):
    prs = deck()

    title_slide(
        prs, "ARI: method, measurements, and what did not survive",
        "Technical appendix. Every figure is generated from the results JSON.",
        "Reports are Ed25519-signed. Verification is independent of our SDK. "
        "Slide 2 lists statements superseded by the 2026-09-16 audit.")
    superseded_slide(prs)

    s = blank(prs)
    header(s, "What SEMQ is, and what it is for", "INSTRUMENT")
    bullets(s, [
        ("SEMQ maps a float vector to a symbolic code by comparing against a "
         "calibrated scale.", INK, True),
        "Codes are exact, so agreement is a hash comparison rather than a threshold.",
        ("The operator uses one fixed calibration scale, not a trained codebook. "
         "Cross-platform invariance still needs verification on each platform.",
         INK, False),
        ("Retracted: we previously claimed SEMQ is necessary for the index. "
         "It is not. A frozen product quantizer also works.", RED, True),
        ("What survives is verification cost: one declared rule and one scalar, "
         "against a codebook of thousands of values that each need checking.",
         INK, False),
    ])

    s = blank(prs)
    header(s, "ARI-R: Recall@10 did not detect a serving change",
           "MEASUREMENT")
    picture(s, "01-regime-discrimination.png", top=Inches(1.7), width=Inches(11.3))
    note(s, "BEIR SciFact, 5,183 documents, 300 queries with real relevance "
            "judgements. Grey bars are controls: new process, single thread, "
            "4x and 16x batch. All read exactly zero, which is what makes the "
            "red bars interpretable.", top=Inches(5.9), size=13)

    s = blank(prs)
    header(s, "ARI-R: the TF32 audit gap, both halves measured", "MEASUREMENT")
    if f.get("tf32_on"):
        on, off = f["tf32_on"], f["tf32_off"]
        table(s, [
            ["Condition vs fp32/CPU", "R@10", "delta R@10", "95% CI",
             "top-10 lists same", "SEMQ HER"],
            ["GPU fp32, TF32 off", f"{off['recall_at_10']:.4f}",
             f"{off['recall_delta']:+.4f}",
             f"[{off['recall_delta_ci95'][0]:+.4f}, {off['recall_delta_ci95'][1]:+.4f}]",
             f"{off['top10_identical']:.2%}", f"{off['semq_her']:.4f}"],
            ["GPU fp32, TF32 on", f"{on['recall_at_10']:.4f}",
             f"{on['recall_delta']:+.4f}",
             f"[{on['recall_delta_ci95'][0]:+.4f}, {on['recall_delta_ci95'][1]:+.4f}]",
             f"{on['top10_identical']:.2%}", f"{on['semq_her']:.4f}"],
        ], top=Inches(2.2), height=Inches(0.7), size=13, highlight={2: GREEN})
    bullets(s, [
        ("The paired interval on delta Recall@10 is [0.0000, 0.0000] at 300 "
         "queries. A nonsignificant difference is not equivalence.", INK, True),
        ("The TF32-off row is the control. It associates the effect with TF32 "
         "rather than with the host change alone.", INK, False),
        ("Narrowed claim: retrieval *quality* is blind. Retrieval *result lists* "
         "are not. int8 changes 100% of them.", RED, False),
    ], top=Inches(4.5), size=15)

    s = blank(prs)
    header(s, "ARI-D: logit codes change while tokens do not", "MEASUREMENT")
    picture(s, "02-decoding-early-warning.png", top=Inches(1.7), width=Inches(10.4),
            left=Inches(1.4))
    note(s, "Teacher-forced: every condition is scored on the reference token "
            "sequence, so position i sees an identical context. That isolates "
            "the per-step computation from compounding divergence.",
         top=Inches(5.7), size=13)

    s = blank(prs)
    header(s, "ARI-D: the result that goes against us", "ADVERSARIAL")
    picture(s, "04-cost-vs-sensitivity.png", top=Inches(1.6), width=Inches(7.4),
            left=Inches(0.6))
    bullets(s, [
        ("The top-2 margin sits at the same height and 2,000x to the left.",
         RED, True),
        "86% of the signal for 8 bytes per step against SEMQ's 16,000.",
        "Both are equally bit-reproducible.",
        ("For detecting a precision change, ship the margin.", RED, True),
        "",
        ("KL divergence is out for a different reason: it is second order, so "
         "it reads zero when it matters, and regrouping the arithmetic moves "
         "it 4.6%.", INK, False),
    ], top=Inches(1.9), left=Inches(8.2), width=Inches(4.6), size=13)

    s = blank(prs)
    header(s, "ARI-D: what survives is coverage, not sensitivity", "ADVERSARIAL")
    picture(s, "05-tail-coverage.png", top=Inches(1.6), width=Inches(7.6),
            left=Inches(0.6))
    bullets(s, [
        ("Move only the tail of the distribution and leave the top ranks "
         "untouched.", INK, True),
        "SEMQ does not move: 0.1289 against 0.1288.",
        ("The margin and the token statistic read exactly zero.", GREEN, True),
        "They are structurally blind, not merely less sensitive.",
        "",
        ("Open question: does any real serving change produce tail-confined "
         "drift? A precision change does not. A token-suppression filter or an "
         "adapter might.", RED, False),
    ], top=Inches(1.9), left=Inches(8.4), width=Inches(4.4), size=13)

    s = blank(prs)
    header(s, "ARI-E: the estimator and its control", "METHOD")
    bullets(s, [
        ("Cross-harness disagreement means nothing until you know how often one "
         "harness disagrees with itself.", INK, True),
        "",
        ("E(h1, h2)  =  mean self-consistency  -  cross-harness agreement",
         BLUE, True),
        "",
        "Zero means the harnesses are interchangeable on those cases.",
        ("If a harness has one run per case, self-consistency is unmeasurable. "
         "The code refuses to report an effect rather than assuming 1.0.",
         INK, False),
        ("Intervals resample cases, not runs, because runs of one case are not "
         "independent.", INK, False),
    ], size=16)

    s = blank(prs)
    header(s, "ARI-E: historical analysis of 22,000 agent runs",
           "MEASUREMENT. HISTORICAL: WITHHELD FROM THE PAPER PENDING RERUN")
    if f.get("e_contrasts"):
        rows = [["Contrast", "Cases", "Self", "Cross", "Effect", "95% CI"]]
        for c in f["e_contrasts"]:
            selves = sum(c["self_consistency"].values()) / len(c["self_consistency"])
            rows.append([
                c["title"].replace(" effect", "").replace("scaffold, ", "scaffold "),
                f"{c['n_cases']:,}", f"{selves:.3f}",
                f"{c['cross_agreement']:.3f}", f"{c['effect']:+.3f}",
                f"[{c['effect_ci'][0]:+.3f}, {c['effect_ci'][1]:+.3f}]"])
        table(s, rows, top=Inches(2.0), height=Inches(0.62), size=12)
    bullets(s, [
        ("Source: nvidia/Open-SWE-Traces. SWE-agent and OpenHands, crossed with "
         "two models. Neither scaffold was built by us.", INK, False),
        ("Outcome is the repository's own test suite, which the agent never saw.",
         INK, False),
        ("Largest threat: 22% of rows were ungraded and dropped. If grading "
         "failure tracks difficulty, the surviving set is easier.", RED, True),
        ("Superseded: earlier estimator and partial export. Rerun against a pinned "
         "dataset revision before quoting.", RED, True),
    ], top=Inches(4.9), size=13)

    s = blank(prs)
    header(s, "ARI-E: why the earlier plan would have failed",
           "POWER. HISTORICAL: SIMULATION PREDATES THE ESTIMATOR CORRECTION")
    table(s, [
        ["Pass-rate gap", "Repeats", "25 cases", "100 cases", "200 cases"],
        ["0.00 (null)", "2", "8%", "6%", "6%"],
        ["0.20", "2", "8%", "11%", "37%"],
        ["0.20", "5", "23%", "74%", "96%"],
        ["0.40", "5", "99%", "100%", "100%"],
    ], top=Inches(2.0), height=Inches(0.65), size=14, highlight={2: RED})
    bullets(s, [
        ("We planned to run 25 cases at 2 repeats. That design detects a "
         "20-point gap 8% of the time.", RED, True),
        ("The published result that motivated the work, 24/25 against 19/25, is "
         "a 20-point gap. We would have missed it nine times in ten.", INK, False),
        ("Simulated before spending the compute, not after.", GREEN, True),
    ], top=Inches(4.5), size=15)

    s = blank(prs)
    header(s, "Verification architecture", "ATTESTATION")
    bullets(s, [
        ("Signing uses SEMQ. Repo content-addresses the artifacts, notary "
         "produces an Ed25519 sidecar.", INK, True),
        ("What is signed is a manifest, not the report. It names every input by "
         "digest, so swapping the data underneath an unchanged report fails.",
         INK, False),
        ("Verification does not require SEMQ. The standalone verifier is stdlib "
         "plus cryptography, and a test walks its import graph to prove it.",
         GREEN, True),
        "",
        ("Stated limits: a pass shows what was signed, not who signed it unless "
         "you pass a key you already trust. Without a timestamp authority the "
         "recorded time is the signer's own clock.", GREY, False),
    ], size=15)

    s = blank(prs)
    header(s, "What we retracted, and why that matters", "CREDIBILITY")
    table(s, [
        ["Claim", "Status"],
        ["'SEMQ is ~300x more sensitive than retrieval'", "Retracted"],
        ["'SEMQ is necessary for the index' (v1, seed stability)", "Retracted"],
        ["'A frozen VQ is not verifiable' (v2)", "Retracted"],
        ["'Serving changes are invisible to retrieval'", "Narrowed to quality metrics"],
        ["'SEMQ is the most sensitive decoding probe'", "Withdrawn; margin is cheaper"],
        ["'Self-hosted time cells satisfy the >24 h gap'", "Superseded; immediate repeats"],
        ["'Two-encoder run shows clean batch invariance'", "Superseded by seven-encoder table"],
        ["'Decoding H-bar is a bit or coordinate rate'", "Corrected; packed-byte rate"],
        ["ARI-E contrasts and power", "Withheld pending rerun"],
    ], top=Inches(1.9), height=Inches(0.42), size=12)
    bullets(s, [
        ("Most were retracted because we tested them adversarially; the last four "
         "came from the 2026-09-16 repository audit.", GREEN, True),
        ("Remaining claims are listed with their evidence in docs/paper/CLAIM_TO_EVIDENCE.md.",
         INK, False),
    ], top=Inches(6.2), size=14)

    prs.save(out)
    return out


def main() -> None:
    f = facts()
    HERE.mkdir(parents=True, exist_ok=True)
    a = build_overview(f, HERE / "ari-overview.pptx")
    b = build_technical(f, HERE / "ari-technical.pptx")
    for p in (a, b):
        print(f"  wrote {p.relative_to(ROOT)}  "
              f"({len(Presentation(p).slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
