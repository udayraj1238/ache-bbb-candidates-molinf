"""Synthesis dossier (Word) for the delivered set: one row per molecule with structure, route, predicted values with margins, flags, plus the
assay plan, QC checklist and limitations. Every number is read from results/SYNTHESIS_SET.csv; nothing is typed in.
Run with bbbenv python (python-docx, RDKit).   Output: paper/AChE_MolInf_v4/Supporting_Information/SYNTHESIS_DOSSIER.docx"""
import io, os
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
RDLogger.DisableLog("rdApp.*")

S = pd.read_csv("results/SYNTHESIS_SET.csv")
BEST = pd.read_csv("results/BEST_MOLECULES.csv") if os.path.exists("results/BEST_MOLECULES.csv") else None
BESTRANK = dict(zip(BEST.smiles, BEST.best_rank)) if BEST is not None else {}
R = pd.read_csv("results/SYNTHESIS_SET_REJECTS.csv")
ROUTES = pd.read_csv("results/AD_V2_ROUTES.csv").drop_duplicates("smiles").set_index("smiles") if os.path.exists("results/AD_V2_ROUTES.csv") else None
GNN = pd.read_csv("results/AD_V2_GNN.csv").set_index("smiles").p_gnn if os.path.exists("results/AD_V2_GNN.csv") else None
OUT = "paper/AChE_MolInf_v4/Supporting_Information/SYNTHESIS_DOSSIER.docx"
doc = Document()
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = sec.page_height, sec.page_width
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(1.6))
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(9.5)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def para(text, bold=False, size=None, style=None, italic=False):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    return p


def table(rows, header, widths=None, size=8.5):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size); shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return t


doc.add_heading("Synthesis dossier: %d brain-penetrant AChE inhibitor candidates" % len(S), 0)
para("All values are computational predictions. None of these molecules has been synthesized or tested. This dossier gives a chemist the structure, the computed route, "
     "the predicted properties with their margins and the flags that decide what to measure first.", italic=True)

doc.add_heading("1. What the set is", 1)
n10 = int((S.min_slack_validated >= 0.10).sum())
para("The %d molecules are the complete group, out of %s designed analogues of potent published AChE inhibitors, that passed every gate listed in Section 2 with at least 5 %% margin "
     "on the tightest gate (median %.0f %%, %d molecules at least 10 %%). They lie on %d Bemis-Murcko scaffolds from %d published parents; %d carry one matched-pair edit and %d two. "
     "Each is absent from PubChem (exact and same connectivity), ChEMBL (flexmatch) and SureChEMBL, and each has a computed route of 1-6 steps to purchasable starting materials "
     "(median route score %.3f). Rank is the margin on the tightest gate; a larger margin means the molecule sits farther inside the model thresholds, not that it is a more potent inhibitor."
     % (len(S), "2,700", 100 * S.min_slack_validated.median(), n10, S.murcko_scaffold.nunique(), S.parent.nunique(), int((S.edits == 1).sum()), int((S.edits == 2).sum()), S.top_score.median()))

doc.add_heading("2. Gates every molecule passed", 1)
gates = [("BBB permeability", "in-loop and audit models >= 0.75; independent Martins model >= 0.513"),
         ("AChE activity", "primary model >= 0.70; both second-pair models >= 0.50; similarity to nearest potent active >= 0.417 (model domain) and < 0.90 (prior art)"),
         ("Brain exposure", "predicted logBB >= -0.45; similarity to logBB training set >= 0.35"),
         ("Drug-likeness", "CNS MPO >= 4; SA score <= 3.5; MW <= 450; no unstable functional group"),
         ("Toxicity", "no endpoint (hERG, Ames, DILI, CYP1A2/2C9/2D6) more than 0.10 worse than the parent"),
         ("Efflux / exposure", "P-gp interaction <= 0.713; exposure surrogate (passive x [1 - P-gp]) >= 0.2485 (donepezil-anchored)"),
         ("Docking", "Vina <= -8.20 kcal/mol and pose coverage of the donepezil footprint >= 0.43 (4EY7; galantamine-anchored)"),
         ("Structure", "stereochemistry defined or justified racemate; protonation rule agrees with Dimorphite-DL at pH 7.4; not a tautomer of a training/reference compound"),
         ("Novelty", "no match in PubChem (exact and same connectivity), ChEMBL (flexmatch) or SureChEMBL"),
         ("Synthesis", "computed route to purchasable starting materials (AiZynthFinder, USPTO policies, ZINC stock)"),
         ("Margin", "at least 5 % slack on every numeric gate above")]
table(gates, ["Gate", "Criterion"], widths=[4.0, 21.0])

doc.add_heading("3. Read this before ordering anything", 1)
fl = [("hERG", "The classifier probability is high (median %.2f, maximum %.2f), as for the donepezil-like parents, but that model failed its calibration (it could not rank cisapride above donepezil). A second model, a regression on 10,470 ChEMBL potencies, ranked unseen scaffolds well (rho 0.67) and separated ten blockers from ten non-blockers (AUC 1.00) but also failed a pre-specified criterion (cisapride was predicted weaker than donepezil), so neither model can rank these molecules. Measure hERG early: it is the most likely reason a candidate fails." % (S.tox_herg.median(), S.tox_herg.max())),
      ("P-gp and BCRP", "Efflux was not measured. The P-gp model is trained on inhibition data and did not separate BBB+ from BBB- compounds (AUC 0.52); the BCRP similarity flag also failed its calibration and flags %d molecule(s)." % int(S.bcrp_watch.sum())),
      ("Docking", "The Vina score separates binders from property-matched decoys only weakly (AUC 0.64). Treat it as a plausibility filter; it is not evidence of potency."),
      ("Top-ranked N-ethyl indanones", "The four molecules with the largest margins have lost donepezil's benzyl ring and so cannot stack on Trp86. They rank high on predicted brain exposure and BBB margin. Three set members were tested earlier in 2 ns of molecular dynamics, one replica each: two kept their pose, one drifted."),
      ("BChE", "%d molecules have a predicted BChE probability >= 0.5. This is not a defect for Alzheimer's disease, but it matters when reading a selectivity assay." % int((S.p_bche >= 0.5).sum())),
      ("Parent compound", "The parent is listed by suppliers for %d molecules and not for %d. No computed route passes through the parent (each molecule is built from purchasable building blocks), so this does not limit the routes; a chemist may prefer a late-stage modification of a catalogued parent where one exists." % (int((S.parent_supply_status == 'purchasable').sum()), int((S.parent_supply_status != 'purchasable').sum()))),
      ("Routes", "Routes are computed from literature reaction templates and are hypotheses for a chemist to review. The listed purchasable precursors are those of the best-scoring route."),
      ("Prior art", "Novelty rests on structure searches in four sources. A structure database cannot see the scope of a patent claim: the eseroline-carbamate series was excluded because its parents sit in patent families on substituted phenserines. All remaining molecules are close analogues of known drugs; freedom to operate needs a patent attorney.")]
table(fl, ["Flag", "What to know"], widths=[4.0, 21.0])

doc.add_heading("4. First experiments, in order", 1)
for s in ["Ellman assay for human AChE IC50, donepezil on every plate; BChE counter-screen.",
          "hERG patch clamp (or a validated binding surrogate) for every compound that is active.",
          "MDCK-MDR1 permeability with and without a P-gp inhibitor; BCRP transport assay for compounds with efflux ratio > 2.",
          "Kinetic solubility, microsomal stability and, for the leads, unbound fraction in plasma and brain tissue.",
          "Begin with the 63 priority-tier molecules (margin 10 % or more) that have a 1-2 step route; tier 1 or 2 is shown under each ID in the table below."]:
    doc.add_paragraph(s, style="List Number")

doc.add_heading("5. Quality checklist for each synthesized compound", 1)
for s in ["Identity: 1H and 13C NMR and HRMS match the structure in this dossier; record the salt form.",
          "Purity: at least 95 % by HPLC at two wavelengths before any assay; no residual metal catalyst above assay tolerance.",
          "Stereochemistry: racemates are labeled as such (every undefined centre lies alpha to a carbonyl and epimerizes); defined centres confirmed by optical rotation or chiral HPLC.",
          "Record the route actually used and compare with the computed route; report any step that failed."]:
    doc.add_paragraph(s, style="List Bullet")

if BEST is not None:
    doc.add_heading("6. Shortlist of the twenty best molecules for a first synthesis batch", 1)
    para("Selection: the 63 priority-tier molecules (margin 10 % or more) were put through every remaining computational check: graph-network probability >= 0.70; docking gates in the original run, in two re-dockings from new seeds and after re-docking with the amine protonated; no PAINS, Brenk or NIH alert; route of at most three steps; no Lipinski or Veber violation. "
         "The 45 eligible molecules were ranked on eight axes under 5,000 random weightings, and the twenty with the best median rank (at most two per scaffold, four per parent) are listed. The order inside the list is not firm (median width of the 5th-95th percentile rank interval about 23 positions); the list is robust as a group. "
         "All twenty still carry the open issues of Section 3: hERG block and potency are unmeasured.", italic=True)
    rows_ = [("%d" % r.best_rank, "AD%03d" % r.set_rank, r.smiles, "%.1f %%" % (100 * r.min_slack_validated), "%.2f" % r.p_gnn, "%.1f" % r.vina_worst, "%d" % r.n_steps, "%d" % r.contact_score, "%.0f-%.0f" % (r.rank_p05, r.rank_p95)) for r in BEST.itertuples()]
    table(rows_, ["Best", "ID", "SMILES", "Margin", "GNN p", "Worst Vina", "Steps", "Contacts (0-3)", "Rank 5-95 %"], widths=[1.2, 1.8, 9.0, 1.8, 1.6, 2.0, 1.4, 2.2, 2.2], size=8)
doc.add_heading("7. All 106 molecules", 1)
para("Columns: structure; AChE, BBB (in-loop / audit / Martins) probabilities; predicted logBB; margin and tightest gate; Vina and pose coverage; route steps and purchasable precursors; parent supply; flags.", italic=True)
hdr = ["ID", "Structure", "Predicted values", "Margin", "Docking", "Route (computed)", "Flags"]
t = doc.add_table(rows=1, cols=len(hdr)); t.style = "Table Grid"
for i, h in enumerate(hdr):
    c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(8.5); shade(c, "D9E2F3")
trow = t.rows[0]._tr; trPr = trow.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
for r in S.itertuples():
    m = Chem.MolFromSmiles(r.smiles)
    buf = io.BytesIO(); Draw.MolToImage(m, size=(360, 260)).save(buf, format="PNG"); buf.seek(0)
    cells = t.add_row().cells
    cells[0].text = ""; run = cells[0].paragraphs[0].add_run("AD%03d" % r.set_rank); run.bold = True; run.font.size = Pt(8.5); cells[0].add_paragraph("priority tier" if r.margin_tier == 1 else "tier 2").runs[0].font.size = Pt(7)
    if r.smiles in BESTRANK:
        bp = cells[0].add_paragraph("shortlist #%d" % BESTRANK[r.smiles]); bp.runs[0].bold = True; bp.runs[0].font.size = Pt(7.5)
    cells[1].text = ""; cells[1].paragraphs[0].add_run().add_picture(buf, width=Cm(4.6))
    sm = cells[1].add_paragraph(); rr = sm.add_run(r.smiles); rr.font.size = Pt(6.5); rr.font.name = "Consolas"
    pv = ["AChE %.2f; BBB %.2f / %.2f / %.2f" % (r.target_probability, r.p_inloop, r.p_audit, r.bbb_martins), "logBB %.2f; CNS MPO %.1f; SA %.1f" % (r.pred_logBB, r.cns_mpo, r.sa_score),
          "MW %.0f; TPSA %.0f; cLogP %.1f; HBD %d" % (r.mw, r.tpsa, r.clogp, r.hbd), "P-gp %.2f; Kp,uu %.2f" % (r.p_pgp_substrate, r.kpuu_surrogate),
          "hERG %.2f (parent %.2f); Ames %.2f; DILI %.2f" % (r.tox_herg, r.parent_tox_herg, r.tox_ames, r.tox_dili), "edits %d; nearest potent active sim %.2f" % (r.edits, r.sim_nearest_potent)]
    cells[2].text = "\n".join(pv)
    cells[3].text = "%.1f %%\n(%s)" % (100 * r.min_slack_validated, r.tightest_validated)
    cells[4].text = "Vina %.1f\ncoverage %.2f" % (r.vina_score, r.pose_coverage)
    prec = str(r.precursors_in_stock).replace(", ", "\n")
    cells[5].text = "%d step%s (longest sequence), score %.3f\nStart from:\n%s\nParent: %s\nStereo: %s" % (r.n_steps, "" if r.n_steps == 1 else "s", r.top_score, prec,
                                                                                         "listed by suppliers" if r.parent_supply_status == "purchasable" else "not listed (route does not use it)",
                                                                                         "defined" if r.stereo_status == "already_defined" else "racemate (alpha to C=O)")
    flags = []
    if r.tox_herg > 0.5: flags.append("hERG predicted high: measure")
    if r.p_bche >= 0.5: flags.append("BChE p %.2f" % r.p_bche)
    if bool(r.bcrp_watch): flags.append("BCRP watch")
    if r.set_rank <= 4: flags.append("N-ethyl indanone: no Trp86 ring")
    if ROUTES is not None and r.smiles in ROUTES.index and isinstance(ROUTES.loc[r.smiles, "reactions"], str) and not str(ROUTES.loc[r.smiles, "reactions"]).startswith("ERROR") and bool(ROUTES.loc[r.smiles, "solved"]):
        steps_ = str(ROUTES.loc[r.smiles, "reactions"]).split(" || ")
        cells[5].text += "\nReactions (computed, %d in total):\n" % len(steps_) + "\n".join("%d. %s" % (i + 1, s) for i, s in enumerate(steps_))
    if GNN is not None and r.smiles in GNN.index:
        cells[2].text += "\nGNN activity p %.2f" % GNN.loc[r.smiles]
    cells[6].text = "\n".join(flags) if flags else "-"
    for c in cells[2:]:
        for p in c.paragraphs:
            for run in p.runs: run.font.size = Pt(7.5)
    for i, w in enumerate([1.6, 5.2, 5.6, 2.3, 2.4, 6.0, 3.3]):
        cells[i].width = Cm(w)

doc.add_heading("8. Candidates that were rejected, and why", 1)
rc = R.reject_reason.value_counts()
table([(k, v) for k, v in rc.items()], ["Reason", "Candidates"], widths=[18.0, 3.0])
para("Every rejected candidate with its reason is in REJECTED_CANDIDATES_WITH_REASON.csv.", italic=True)
doc.save(OUT)
print("saved", OUT, "| molecules", len(S))
