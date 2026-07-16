"""Generate examples/label_cooccurrence_explained.pdf.

A self-contained, plain-language walkthrough of the FSD50K label co-occurrence
analysis (scripts/label_cooccurrence_fsd50k.py) and how it motivates the
PLG/LLG label graphs in ATGNN. Rerun after regenerating the figure:

    python examples/make_label_analysis_pdf.py
"""

import os

from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIG = os.path.join(REPO, "results", "label_cooccurrence.png")
OUT = os.path.join(HERE, "label_cooccurrence_explained.pdf")

MARGIN = 18
BODY_W = 210 - 2 * MARGIN  # A4 width minus margins


class Doc(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(130)
            self.cell(0, 6, "ATGNN - FSD50K label co-occurrence analysis", align="R")
            self.ln(8)
            self.set_text_color(0)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(0, 6, f"{self.page_no()}", align="C")
        self.set_text_color(0)

    def h1(self, txt):
        self.set_font("helvetica", "B", 16)
        self.multi_cell(BODY_W, 8, txt)
        self.ln(2)

    def h2(self, txt):
        self.ln(2)
        self.set_font("helvetica", "B", 12.5)
        self.multi_cell(BODY_W, 7, txt)
        self.ln(1)

    def body(self, txt):
        self.set_font("helvetica", "", 10.5)
        self.multi_cell(BODY_W, 5.4, txt)
        self.ln(1.5)

    def bullet(self, txt):
        self.set_font("helvetica", "", 10.5)
        x = self.get_x()
        self.cell(5, 5.4, "-")
        self.multi_cell(BODY_W - 5, 5.4, txt)
        self.set_x(x)
        self.ln(0.8)


doc = Doc(format="A4")
doc.set_margins(MARGIN, 14, MARGIN)
doc.set_auto_page_break(True, margin=16)
doc.add_page()

doc.h1("Why label graphs? The FSD50K label co-occurrence analysis, explained")

doc.body(
    "ATGNN contains two components that go beyond a standard audio classifier: the PLG "
    "(Patch-Label Graph) block, where learnable label embeddings attach to regions of the "
    "spectrogram, and the LLG (Label-Label Graph) block, where a learned matrix lets each "
    "label's embedding influence other labels. A natural examiner question is: what is the "
    "evidence that modelling relationships between labels is worth doing at all?"
)
doc.body(
    "This document answers that question with data. The analysis script "
    "(scripts/label_cooccurrence_fsd50k.py) uses only the official FSD50K ground-truth "
    "annotations (dev.csv + eval.csv: 51,197 clips, 200 classes) and asks two things: "
    "(1) do clips genuinely carry multiple labels, and (2) do labels co-occur with strong, "
    "systematic structure that a model could learn?"
)
doc.body(
    "The logic: if labels mostly appeared alone, or co-occurred randomly, a label graph would "
    "be pointless - a classifier predicting each class independently would suffice. But if "
    "clips usually carry several labels, and certain labels systematically appear together, "
    "then a model that learns those relationships has real signal to exploit."
)

doc.h2("Headline numbers")
doc.bullet("84.3% of clips are multi-label; the mean is 2.99 labels per clip (median 3, max 22).")
doc.bullet("169 of the 200 classes have at least one partner label that co-occurs with them "
           "more than half the time - most of those near 100%.")
doc.bullet("The strongest pairs: Music + Musical_instrument (14,703 clips, P ~ 1.0), "
           "Musical_instrument + Percussion (3,977), Animal + Wild_animals (2,392, "
           "P(Wild_animals | Animal) = 0.55).")

doc.h2("The four-panel figure")
doc.body("The figure below is produced by the analysis script "
         "(results/label_cooccurrence.png). Each panel is explained on the next page.")
doc.image(FIG, x=MARGIN, w=BODY_W)

doc.add_page()
doc.h2("Top-left: labels per clip")
doc.body(
    "A histogram of how many labels each clip carries. The bulk of clips have 2-4 labels and "
    "84% have more than one. Takeaway: FSD50K is fundamentally a multi-label dataset, so "
    "predicting labels jointly - not independently - matches the data."
)

doc.h2("Top-right: class frequency (long tail)")
doc.body(
    "Classes ranked from most to least common, on a logarithmic scale. The most common class "
    "appears in roughly 15,000 clips, the rarest in about 100 - a steep 'long tail'. "
    "Takeaway: rare classes have very little training data of their own, which is exactly the "
    "situation where borrowing signal from correlated, more frequent labels helps."
)

doc.h2("Bottom-left: the co-occurrence heatmap")
doc.body(
    "The core panel. It shows the 30 most frequent classes; the cell at row i, column j is the "
    "conditional probability P(j | i): given that a clip contains label i, how likely is it to "
    "also contain label j? Dark purple means the pair almost never co-occurs; bright yellow "
    "means that if you see label i, label j is almost certainly present too. For example, the "
    "bright cell on the Guitar row under the Music column means virtually every clip tagged "
    "Guitar is also tagged Music."
)
doc.body(
    "The key visual point: the map is mostly dark with a few very bright cells. Label "
    "relationships are sparse and specific, not uniform noise. That specific, learnable "
    "pattern is what the LLG block's adjacency matrix can capture."
)

doc.h2("Bottom-right: how strong is each class's best partner?")
doc.body(
    "For each of the 200 classes we take the highest co-occurrence probability it has with "
    "any other class, and plot the distribution. 169 of 200 classes have a partner label that "
    "follows them more than half the time - and for most of those the relationship is nearly "
    "deterministic (the tall bar at 1.0). Takeaway: almost every class has at least one "
    "strongly predictive companion label."
)

doc.h2("Where the strong pairs come from")
doc.body("Two flavours of co-occurrence show up in results/top_label_pairs.csv:")
doc.bullet(
    "Ontology hierarchy. FSD50K labels come from the AudioSet ontology, so an Electric_guitar "
    "clip is also tagged Guitar and Music by construction. This gives near-deterministic "
    "pairs such as Music + Musical_instrument (P ~ 1.0)."
)
doc.bullet(
    "Real-world regularities. Pairs like Animal + Wild_animals or Bird + Wild_animals are not "
    "forced by the ontology but reflect how sounds actually co-occur in recordings "
    "(normalised PMI 0.80 for Animal + Wild_animals)."
)
doc.body(
    "Both kinds are learnable structure: if the model is fairly confident about "
    "Electric_guitar, the label graph lets it raise its confidence in Music essentially for "
    "free - and conversely, implausible label combinations can be suppressed."
)

doc.h2("How this maps to the ATGNN architecture")
doc.bullet(
    "PLG (patch-label): with ~3 labels per clip, different labels typically correspond to "
    "different regions/times of the spectrogram. Attaching each label embedding to its "
    "nearest patch nodes lets co-occurring labels bind to different evidence in the same clip."
)
doc.bullet(
    "LLG (label-label): the sparse, near-deterministic co-occurrence structure measured here "
    "is exactly the kind of latent correlation a learnable label-label adjacency matrix "
    "(L_hat = A L + L) can encode."
)

doc.h2("One-sentence summary for the thesis")
doc.body(
    "FSD50K clips carry ~3 labels on average, 84% are multi-label, and nearly every class has "
    "a companion label that co-occurs with it more than half the time - so architectures like "
    "ATGNN's PLG/LLG blocks, which explicitly model patch-label and label-label "
    "relationships, address measurable structure in the data rather than an assumed one."
)

doc.h2("Reproducing")
doc.body(
    "python scripts/label_cooccurrence_fsd50k.py --gt_dir $DATA_DIR/ground_truth "
    "--out_dir results\n"
    "Outputs: results/label_cooccurrence.png (the figure), results/label_stats.csv "
    "(per-class frequencies), results/top_label_pairs.csv (top-100 pairs with conditional "
    "probabilities and normalised PMI). This PDF: python examples/make_label_analysis_pdf.py."
)

doc.output(OUT)
print(f"written {OUT}")
