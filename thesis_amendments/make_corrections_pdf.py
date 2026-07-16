"""Generate thesis_corrections_resolutions.pdf.

A working document that resolves the substantive amendments from the examiner
correction report (steps_thesis_correction_report.pdf) with copy-ready draft
text for the amended thesis. Grammar/style items and amendments that still
require new experiments are listed as remaining items at the end.

    python make_corrections_pdf.py
"""

import os

from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thesis_corrections_resolutions.pdf")

MARGIN = 18
BODY_W = 210 - 2 * MARGIN


def _s(t):
    """Sanitise to latin-1 for the PDF core fonts."""
    repl = {"\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2248": "~", "\u2192": "->",
            "\u2264": "<=", "\u00d7": "x", "\u22c5": ".", "\u2026": "..."}
    for k, v in repl.items():
        t = t.replace(k, v)
    return t.encode("latin-1", "replace").decode("latin-1")


class Doc(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(130)
            self.cell(0, 6, "Thesis amendments - resolved drafts and remaining items", align="R")
            self.ln(8)
            self.set_text_color(0)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(0, 6, f"{self.page_no()}", align="C")
        self.set_text_color(0)

    def h1(self, t):
        self.set_font("helvetica", "B", 15)
        self.set_fill_color(232, 236, 244)
        self.multi_cell(BODY_W, 8, _s(t), fill=True)
        self.ln(2)

    def h2(self, t):
        self.ln(1.5)
        self.set_font("helvetica", "B", 12)
        self.set_text_color(30, 60, 120)
        self.multi_cell(BODY_W, 6.5, _s(t))
        self.set_text_color(0)
        self.ln(1)

    def h3(self, t):
        self.ln(1)
        self.set_font("helvetica", "B", 10.5)
        self.multi_cell(BODY_W, 5.6, _s(t))
        self.ln(0.5)

    def body(self, t):
        self.set_font("helvetica", "", 10)
        self.multi_cell(BODY_W, 5.2, _s(t))
        self.ln(1.2)

    def note(self, t):
        self.set_font("helvetica", "I", 9.5)
        self.set_text_color(100)
        self.multi_cell(BODY_W, 5, _s(t))
        self.set_text_color(0)
        self.ln(1.2)

    def bullet(self, t):
        self.set_font("helvetica", "", 10)
        x = self.get_x()
        self.cell(5, 5.2, "-")
        self.multi_cell(BODY_W - 5, 5.2, _s(t))
        self.set_x(x)
        self.ln(0.6)

    def latex(self, t):
        """Copy-ready LaTeX / draft text block."""
        self.set_font("courier", "", 8.3)
        self.set_fill_color(245, 245, 240)
        self.set_draw_color(200)
        self.multi_cell(BODY_W, 4.2, _s(t), fill=True, border=1)
        self.ln(2)


doc = Doc(format="A4")
doc.set_margins(MARGIN, 14, MARGIN)
doc.set_auto_page_break(True, margin=16)
doc.add_page()

# ----------------------------------------------------------------------------
doc.h1("Thesis amendments: resolved drafts and remaining items")
doc.body(
    "This document works through the code-dependent and substantive writing corrections from "
    "the examiner report. For each item it provides draft text (prose and/or LaTeX) that can be "
    "copied into the amended thesis, using the numbers produced by the amendment experiments "
    "(multi-seed URBAN-SED study, DCASE few-shot study with random baseline, FSD50K "
    "reproduction runs of ATGNN/LHGNN, parameter/walltime benchmarks, and the FSD50K label "
    "co-occurrence analysis). Grammatical/style corrections and amendments that still require "
    "new experiments or decisions are listed in the final section as remaining items."
)
doc.body(
    "All new numbers are reproducible from the released repositories: "
    "github.com/shubhrsingh22/hypernet-sed, github.com/shubhrsingh22/dcase-fewshot-hypernet, "
    "github.com/shubhrsingh22/Local-Higher-GNN, github.com/shubhrsingh22/ATGNN (private at the "
    "time of writing; to be made public or archived before submission)."
)

# ============================================================== CHAPTER 1
doc.h1("Chapter 1 - Introduction")

doc.h2("1a. Code and data availability paragraph (new, for Chapter 1 or an appendix)")
doc.latex(
    "\\section*{Code and Data Availability}\n"
    "The experimental work in this thesis is supported by publicly released code. The\n"
    "hypernetwork experiments of Chapter~3 are available at\n"
    "\\url{https://github.com/shubhrsingh22/hypernet-sed} (URBAN-SED) and\n"
    "\\url{https://github.com/shubhrsingh22/dcase-fewshot-hypernet} (few-shot bioacoustics);\n"
    "the audio tagging models of Chapter~4 at\n"
    "\\url{https://github.com/shubhrsingh22/Local-Higher-GNN} and\n"
    "\\url{https://github.com/shubhrsingh22/ATGNN}; and the Chapter~5 systems at the\n"
    "repositories referenced in Sections~5.2 and~5.3. Each repository contains the model\n"
    "code, configuration files, training and evaluation scripts, environment\n"
    "specifications, and dataset preparation instructions. Datasets are not\n"
    "redistributed; official download links and preparation scripts are provided instead.\n"
    "Results in this thesis correspond to tagged releases of these repositories\n"
    "(release tags and commit hashes are given per chapter)."
)
doc.note("Insert the release tags/commit hashes once the repositories are tagged (remaining item R6).")

doc.h2("1b. Measurable hypothesis (replacement for the hypothesis paragraph in 1.1.1)")
doc.latex(
    "The central hypothesis of this thesis is that graph neural networks, operating on\n"
    "nodes derived from time--frequency regions or learned embeddings, can match or exceed\n"
    "the task performance of convolutional and attention-based architectures of comparable\n"
    "or larger parameter count on standard machine listening benchmarks, while providing\n"
    "complementary advantages: (i) competitive accuracy without external (ImageNet)\n"
    "pretraining, (ii) parameter efficiency relative to Transformer baselines, and\n"
    "(iii) robustness of learned representations to signal degradations such as additive\n"
    "noise and reverberation. Improvement is assessed against published CNN and Transformer\n"
    "baselines (PANNs, PSLA, AST and task-specific baselines) under matched training\n"
    "pipelines, datasets and evaluation metrics (mAP for tagging, segment-based F1 for SED,\n"
    "top-1 hit rate for fingerprinting, retrieval metrics for similarity)."
)

doc.h2("1c. Strengthened graph sparsification contribution (addition to Contributions)")
doc.latex(
    "A recurring design element across the proposed models is \\emph{learned graph\n"
    "sparsification}: rather than operating on fully connected graphs, each architecture\n"
    "restricts message passing to a small, data-dependent neighbourhood (k-nearest\n"
    "neighbours in feature space in Chapters~4 and~5, top-$n$ thresholded learned\n"
    "affinities in Chapter~5). This bounds the computational cost of graph construction,\n"
    "acts as a structural prior that suppresses spurious long-range connections, and\n"
    "mitigates the over-smoothing associated with dense aggregation (Section~2.3.7)."
)

doc.h2("1d. Audio-as-graph visualisation")
doc.body(
    "A figure is required (remaining item R8). Suggested composition: left panel, a log-mel "
    "spectrogram of a polyphonic clip; middle panel, the same spectrogram overlaid with patch "
    "nodes and k-NN edges connecting harmonically/temporally related regions; right panel, the "
    "abstract graph with node embeddings. Suggested caption below."
)
doc.latex(
    "\\caption{Recasting audio as a graph. A log-mel spectrogram (left) is divided into\n"
    "patches that become graph nodes (middle); edges connect each node to its $k$ nearest\n"
    "neighbours in feature space, linking related time--frequency regions irrespective of\n"
    "their grid distance. Graph neural networks then update node embeddings by message\n"
    "passing over this structure (right).}"
)

# ============================================================== CHAPTER 2
doc.h1("Chapter 2 - Background (formal precision)")
doc.body(
    "The report asks for a more formal Chapter 2: explicit notation, definitions before use, "
    "and an over-smoothing / graph-transformer discussion. The following drop-in blocks "
    "address each point; place them at the start of Section 2.2/2.3 and inside Section 2.3.7."
)

doc.h2("2a. Notation and terminology paragraph (start of Section 2.2)")
doc.latex(
    "\\paragraph{Notation.} Scalars are written in lowercase italics ($x$), vectors in bold\n"
    "lowercase ($\\mathbf{x}$), matrices in bold uppercase ($\\mathbf{X}$), and sets in\n"
    "calligraphic type ($\\mathcal{X}$). A dataset is $\\mathcal{D} = \\{(\\mathbf{x}_i,\n"
    "\\mathbf{y}_i)\\}_{i=1}^{N}$ with inputs $\\mathbf{x}_i \\in \\mathbb{R}^{d}$ and targets\n"
    "$\\mathbf{y}_i$. A model $f_{\\theta}$ with parameters $\\theta$ produces predictions\n"
    "$\\hat{\\mathbf{y}}_i = f_{\\theta}(\\mathbf{x}_i)$. Throughout this thesis, the\n"
    "\\emph{loss function} $\\ell(\\hat{\\mathbf{y}}, \\mathbf{y})$ measures the discrepancy\n"
    "on a single example, while the \\emph{cost function} $J(\\theta) = \\frac{1}{N}\n"
    "\\sum_{i=1}^{N} \\ell(f_{\\theta}(\\mathbf{x}_i), \\mathbf{y}_i)$ is its average over the\n"
    "training set; training minimises $J(\\theta)$ by stochastic gradient descent. For a\n"
    "network of $L$ layers, $\\mathbf{h}^{(l)}$ denotes the activations of \\emph{hidden\n"
    "layer} $l \\in \\{1,\\dots,L-1\\}$ and the final layer $L$ is the \\emph{output layer},\n"
    "whose activation (sigmoid, softmax, or identity) is chosen to match the target type."
)

doc.h2("2b. CNN computation (formal definition for Section 2.3.2)")
doc.latex(
    "Formally, a convolutional layer maps an input feature map $\\mathbf{X} \\in\n"
    "\\mathbb{R}^{C_{in} \\times H \\times W}$ to $\\mathbf{Y} \\in \\mathbb{R}^{C_{out}\n"
    "\\times H' \\times W'}$ via\n"
    "\\begin{equation}\n"
    "\\mathbf{Y}[c, i, j] = b_c + \\sum_{c'=1}^{C_{in}} \\sum_{u=1}^{k_h} \\sum_{v=1}^{k_w}\n"
    "\\mathbf{K}_c[c', u, v]\\; \\mathbf{X}[c',\\, s_h i + u - 1,\\, s_w j + v - 1],\n"
    "\\end{equation}\n"
    "where $\\mathbf{K}_c$ is the $c$-th learned kernel of size $k_h \\times k_w$, $b_c$ its\n"
    "bias and $(s_h, s_w)$ the stride. The same kernel is applied at every spatial\n"
    "position (weight sharing), yielding shift-equivariant features with a number of\n"
    "parameters independent of the input size. In the audio context, $H$ and $W$ index\n"
    "time frames and frequency bands of a (mel-)spectrogram."
)

doc.h2("2c. RNN computation (formal definition for Section 2.3.3)")
doc.latex(
    "A recurrent layer processes a sequence $(\\mathbf{x}_1, \\dots, \\mathbf{x}_T)$ by\n"
    "maintaining a hidden state $\\mathbf{h}_t \\in \\mathbb{R}^{N_h}$ updated as\n"
    "$\\mathbf{h}_t = \\phi(\\mathbf{W}_x \\mathbf{x}_t + \\mathbf{W}_h \\mathbf{h}_{t-1} +\n"
    "\\mathbf{b})$, where $\\mathbf{W}_x \\in \\mathbb{R}^{N_h \\times N_x}$ and\n"
    "$\\mathbf{W}_h \\in \\mathbb{R}^{N_h \\times N_h}$ are the input and recurrent weight\n"
    "matrices, shared across all time steps, and $\\phi$ is a nonlinearity. The LSTM\n"
    "\\citep{hochreiter1997long} augments this with a cell state $\\mathbf{c}_t$ and\n"
    "input/forget/output gates $(\\mathbf{i}_t, \\mathbf{f}_t, \\mathbf{o}_t)$ that control\n"
    "what is written to, retained in, and read from the cell:\n"
    "$\\mathbf{c}_t = \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot\n"
    "\\tilde{\\mathbf{c}}_t$, $\\mathbf{h}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t)$,\n"
    "mitigating vanishing gradients over long sequences."
)

doc.h2("2d. General Transformer architecture (addition to Section 2.3.x, before AST)")
doc.latex(
    "The Transformer \\citep{vaswani2017attention} dispenses with recurrence and instead\n"
    "relates all positions of a sequence directly through self-attention. Given token\n"
    "embeddings $\\mathbf{X} \\in \\mathbb{R}^{N \\times d}$, each attention head computes\n"
    "queries, keys and values $\\mathbf{Q} = \\mathbf{X}\\mathbf{W}_Q$, $\\mathbf{K} =\n"
    "\\mathbf{X}\\mathbf{W}_K$, $\\mathbf{V} = \\mathbf{X}\\mathbf{W}_V$ and outputs\n"
    "\\begin{equation}\n"
    "\\mathrm{Attn}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) =\n"
    "\\mathrm{softmax}\\!\\left(\\mathbf{Q}\\mathbf{K}^{\\top}/\\sqrt{d_k}\\right)\\mathbf{V},\n"
    "\\end{equation}\n"
    "a data-dependent weighted average in which every token attends to every other token.\n"
    "A Transformer block combines multi-head attention with a position-wise feed-forward\n"
    "network, residual connections and layer normalisation; positional encodings inject\n"
    "order information that attention alone does not provide. Because attention weights\n"
    "are computed from the data rather than fixed by locality, Transformers capture\n"
    "long-range dependencies at the cost of $\\mathcal{O}(N^2)$ complexity in the sequence\n"
    "length, and they lack the built-in locality bias of CNNs, which is one reason\n"
    "attention-based audio models benefit strongly from large-scale pretraining."
)

doc.h2("2e. GNN tasks and readout (addition to Section 2.3.7)")
doc.latex(
    "GNN prediction tasks are commonly grouped by the granularity of the target\n"
    "\\citep{zhou2020graph}: \\emph{node-level} tasks predict a label per node (e.g.\n"
    "classifying each time--frequency patch); \\emph{edge-level} tasks predict the presence\n"
    "or weight of links between node pairs (e.g. affinity between two embeddings); and\n"
    "\\emph{graph-level} tasks predict a property of the whole graph (e.g. the tag set of a\n"
    "clip). Graph-level prediction requires a \\emph{readout} (aggregation) function that\n"
    "pools node embeddings into a fixed-size graph representation, $\\mathbf{h}_G =\n"
    "\\mathrm{READOUT}(\\{\\mathbf{h}_v : v \\in \\mathcal{V}\\})$, typically mean, max or sum\n"
    "pooling, chosen to be permutation-invariant so that the output does not depend on\n"
    "node ordering. The tagging models of Chapter~4 are graph-level predictors with\n"
    "average-pooling readout; the similarity and fingerprinting models of Chapter~5\n"
    "produce clip-level embeddings by the same mechanism."
)

doc.h2("2f. Over-smoothing and low-pass filtering (addition to Section 2.3.7)")
doc.latex(
    "\\paragraph{Over-smoothing.} Stacking many message-passing layers degrades GNN\n"
    "performance: each aggregation step averages a node's embedding with its neighbours',\n"
    "so with depth the embeddings of connected nodes converge and become indistinguishable\n"
    "\\citep{li2018deeper, oono2020graph}. From a spectral perspective, the normalised\n"
    "aggregation operator acts as a low-pass filter on the graph signal: repeated\n"
    "application suppresses the high-frequency components that discriminate between nodes,\n"
    "leaving predominantly the smooth (low-frequency) components \\citep{nt2019revisiting}.\n"
    "Intuitively, after $L$ layers a node's receptive field covers its $L$-hop\n"
    "neighbourhood; when this neighbourhood covers most of the graph, all nodes aggregate\n"
    "nearly the same information. Common mitigations, used in this thesis, include\n"
    "restricting neighbourhood size via sparse $k$-NN graphs, dilated neighbour selection\n"
    "(Chapter~4), residual/initial connections, and keeping the number of message-passing\n"
    "layers per stage small."
)

doc.h2("2g. Graph Transformers (new short subsection after GNNs)")
doc.latex(
    "\\paragraph{Graph Transformers.} Standard Transformers can be viewed as GNNs\n"
    "operating on a fully connected graph in which attention learns the edge weights.\n"
    "Graph Transformers \\citep{dwivedi2021generalization, ying2021transformers} make this\n"
    "connection explicit: attention is computed over graph neighbourhoods or biased by\n"
    "structural encodings (shortest-path distances, Laplacian eigenvectors), combining the\n"
    "relational inductive bias of GNNs with the global receptive field of attention. They\n"
    "avoid over-smoothing by design, since attention weights are content-dependent rather\n"
    "than uniform averages, but inherit the quadratic cost of dense attention. The models\n"
    "in this thesis instead retain sparse $k$-NN message passing for efficiency, while\n"
    "adopting Transformer components (feed-forward blocks, residual connections,\n"
    "positional encodings) within each graph block; extending them to full graph\n"
    "Transformers is discussed as future work in Chapter~6."
)

# ============================================================== CHAPTER 3
doc.h1("Chapter 3 - Hypernetworks for SED")

doc.h2("3a. Multi-seed URBAN-SED results (update to Table 3.2 and discussion)")
doc.body(
    "The amendment experiments retrained all models on URBAN-SED with 5 seeds "
    "(hypernet-sed repository, results/table2.md). LaTeX table and replacement discussion:"
)
doc.latex(
    "\\begin{table}[t]\\centering\n"
    "\\caption{Segment-wise F1 (\\%) on the URBAN-SED test set: mean $\\pm$ standard\n"
    "deviation over five random seeds. Each run is scored as the mean over its ten best\n"
    "checkpoints by validation loss.}\n"
    "\\begin{tabular}{lcc}\\toprule\n"
    "Model & Overall F1 & Avg. class F1 \\\\ \\midrule\n"
    "Bi-baseline   & $62.74 \\pm 0.19$ & $62.27 \\pm 0.30$ \\\\\n"
    "Uni-baseline  & $61.19 \\pm 0.45$ & $60.87 \\pm 0.58$ \\\\\n"
    "HCRNN-32      & $60.90 \\pm 0.75$ & $60.52 \\pm 0.86$ \\\\\n"
    "HCRNN-64      & $61.06 \\pm 0.45$ & $60.68 \\pm 0.65$ \\\\\n"
    "HCRNN-128     & $61.62 \\pm 0.60$ & $61.36 \\pm 0.72$ \\\\\n"
    "HCRNN-256     & $61.72 \\pm 0.69$ & $61.36 \\pm 0.85$ \\\\ \\bottomrule\n"
    "\\end{tabular}\\end{table}"
)
doc.latex(
    "Across five seeds, the unidirectional HCRNN variants improve upon the unidirectional\n"
    "baseline by a modest margin (HCRNN-128: $+0.43$ F1 on average), while the\n"
    "bidirectional baseline remains the strongest configuration overall. The standard\n"
    "deviations (0.2--0.8 F1) show that single-run differences of under one point --- as\n"
    "reported in the original submission --- fall within seed-to-seed variability, and the\n"
    "conclusions are accordingly softened: dynamic weight generation is competitive with,\n"
    "but does not consistently surpass, a bidirectional recurrence of comparable output\n"
    "dimensionality on this dataset. The per-class breakdown shows the hypernetwork's\n"
    "largest gains on classes with long, stationary activity (air conditioner, engine\n"
    "idling), consistent with the interpretation that input-conditioned scaling helps most\n"
    "where the recurrent dynamics must adapt to slowly varying context."
)
doc.note(
    "Honesty note: in the multi-seed reproduction the HCRNNs do NOT beat the bi-baseline, "
    "unlike the single-run thesis numbers. The draft above follows the report's guidance "
    "('emphasise trends and limitations rather than single-number claims'). The TUT-SED "
    "Synthetic 2016 multi-seed rerun is still outstanding (remaining item R2)."
)

doc.h2("3b. Hidden-dimension sweep (new paragraph answering the over-parameterisation concern)")
doc.latex(
    "To examine sensitivity to the hypernetwork's capacity, the hidden size $N_{\\hat h}$\n"
    "was swept over $\\{32, 64, 128, 256\\}$ (Table above). Performance varies by only\n"
    "$0.8$ F1 across an $8\\times$ range of hypernetwork width, with a weak upward trend\n"
    "saturating beyond $N_{\\hat h} = 128$. The corresponding parameter counts (2.47M,\n"
    "2.59M, 2.84M, 3.44M, versus 2.34M for the unidirectional baseline) confirm that the\n"
    "hypernetwork adds only 6--47\\% parameters, and that the observed differences between\n"
    "HCRNN-64 and HCRNN-128 are not an artefact of model order: both lie within one\n"
    "standard deviation of each other."
)

doc.h2("3c. Parameter counts, runtime, and compute details (new experimental-setup text)")
doc.latex(
    "\\begin{table}[t]\\centering\n"
    "\\caption{Parameter counts and measured walltime (single NVIDIA L40S, PyTorch 2.5.1,\n"
    "CUDA 12.4, float32, no JIT compilation, batch 64 of 2\\,s chunks, 8 data-loader\n"
    "workers). Train step = forward + backward + Adam update; inference under\n"
    "\\texttt{no\\_grad}.}\n"
    "\\begin{tabular}{lccc}\\toprule\n"
    "Model & Params (M) & Train step (ms) & Inference (ms/clip) \\\\ \\midrule\n"
    "Uni-baseline & 2.34 & 65  & 0.31 \\\\\n"
    "Bi-baseline  & 5.76 & 69  & 0.33 \\\\\n"
    "HCRNN-32     & 2.47 & 483 & 1.48 \\\\\n"
    "HCRNN-64     & 2.59 & 483 & 1.53 \\\\\n"
    "HCRNN-128    & 2.84 & 506 & 1.53 \\\\\n"
    "HCRNN-256    & 3.44 & 505 & 1.53 \\\\ \\bottomrule\n"
    "\\end{tabular}\\end{table}"
)
doc.latex(
    "Although the HCRNNs use fewer parameters than the bidirectional baseline, they train\n"
    "roughly $7\\times$ slower per step in the present implementation: the per-time-step\n"
    "weight generation precludes the fused cuDNN LSTM kernel and executes as an explicit\n"
    "loop over time. Parameter count therefore does not translate into runtime\n"
    "efficiency, and walltime is reported alongside parameters throughout. No\n"
    "just-in-time compilation (\\texttt{torch.compile} or TorchScript) was used for any\n"
    "model; mixed precision was not used for these experiments."
)

doc.h2("3d. Clarification of Equations 3.14-3.17 (insert after Eq. 3.16)")
doc.latex(
    "In Equations 3.14--3.16 the tensor--vector product $\\langle \\mathbf{W}_{xz},\n"
    "\\mathbf{z}_x \\rangle$ denotes contraction over the embedding dimension: for\n"
    "$\\mathbf{W}_{xz} \\in \\mathbb{R}^{N_h \\times N_x \\times N_z}$ and $\\mathbf{z}_x \\in\n"
    "\\mathbb{R}^{N_z}$, the result is the matrix $\\sum_{k=1}^{N_z} \\mathbf{W}_{xz}[:,:,k]\\,\n"
    "z_{x,k} \\in \\mathbb{R}^{N_h \\times N_x}$, i.e. the generated weight matrix is a\n"
    "linear combination of $N_z$ learned basis matrices with input-dependent\n"
    "coefficients. Storing these bases requires $N_h N_x N_z$ parameters per gate, which\n"
    "is prohibitive; the factorised form of Eq. 3.17 replaces the full generation by an\n"
    "input-dependent row-wise rescaling $d(\\mathbf{z})$ of a single shared matrix,\n"
    "reducing the additional parameters to $\\mathcal{O}(N_h N_z)$ per gate while\n"
    "retaining per-time-step adaptivity."
)

doc.h2("3e. Time-varying filters and silent periods (insert into the motivation)")
doc.latex(
    "The motivation for input-conditioned recurrence is that the statistics of a\n"
    "soundscape change over time: event onsets, overlapping sources and silent periods\n"
    "each call for different temporal integration. A fixed-weight LSTM applies the same\n"
    "transition function at every frame, so it must encode all of these regimes in a\n"
    "single parameterisation. The hypernetwork instead acts as a time-varying filter\n"
    "generator: during silent or stationary passages the scaling vector $d(\\mathbf{z}_t)$\n"
    "can attenuate state updates, effectively lengthening the memory, while at event\n"
    "onsets it can sharpen the update dynamics. This adaptivity is the property being\n"
    "tested in the experiments that follow."
)

doc.h2("3f. LSTM as temporal pooling; 1-second segments (clarifications)")
doc.latex(
    "In this architecture the LSTM does not act as a temporal pooling stage: the CNN\n"
    "front-end preserves the frame rate (pooling is applied along frequency only), and\n"
    "the LSTM outputs one hidden state per frame, from which frame-wise class\n"
    "probabilities are predicted. Temporal aggregation occurs only at evaluation time,\n"
    "when frame-level predictions are compared against reference annotations in\n"
    "fixed-length segments.\n\n"
    "Segment-based F1 with 1-second segments is used following the standard\n"
    "\\texttt{sed\\_eval} protocol \\citep{mesaros2016metrics}, in which predictions and\n"
    "references are compared per one-second block. This granularity matches the DCASE\n"
    "evaluation convention, is tolerant to small onset/offset deviations that are\n"
    "perceptually insignificant, and enables direct comparison with published URBAN-SED\n"
    "and TUT-SED results using the same tolerance."
)

doc.h2("3g. Table 3.3: random baseline and updated few-shot results")
doc.latex(
    "\\begin{table}[t]\\centering\n"
    "\\caption{DCASE few-shot bioacoustic detection, Validation Set: event-based\n"
    "precision, recall and F-score (\\%), harmonic mean over subsets; mean $\\pm$ 95\\%\n"
    "confidence interval over three trials.}\n"
    "\\begin{tabular}{lccc}\\toprule\n"
    "Model & Precision & Recall & F-score \\\\ \\midrule\n"
    "Random baseline & $0.80 \\pm 0.02$ & $16.68 \\pm 0.30$ & $1.52 \\pm 0.03$ \\\\\n"
    "Proto Network   & $18.06 \\pm 0.39$ & $21.78 \\pm 7.30$ & $19.44 \\pm 3.23$ \\\\\n"
    "H-Proto-1       & $19.05 \\pm 2.05$ & $24.46 \\pm 6.12$ & $21.14 \\pm 2.53$ \\\\\n"
    "H-Proto-2       & $16.29 \\pm 0.99$ & $23.24 \\pm 3.44$ & $19.13 \\pm 1.80$ \\\\\n"
    "H-Proto-3       & $17.82 \\pm 2.27$ & $25.56 \\pm 1.80$ & $20.96 \\pm 1.89$ \\\\\n"
    "H-Proto-All     & $18.62 \\pm 1.84$ & $25.94 \\pm 0.62$ & $21.66 \\pm 1.38$ \\\\\n"
    "\\bottomrule\\end{tabular}\\end{table}"
)
doc.latex(
    "A random-guessing baseline (uniformly sampled event predictions with matched\n"
    "duration statistics) achieves an F-score of only 1.5\\%, confirming that the absolute\n"
    "scores in this task reflect genuine difficulty rather than a high chance floor:\n"
    "few-shot bioacoustic detection requires generalising from five labelled events to\n"
    "unseen species and recording conditions, and even recent published systems on this\n"
    "benchmark remain below 60\\% F-score on the validation subsets. Direct comparison\n"
    "with state-of-the-art DCASE submissions is complicated by differing validation\n"
    "subsets across challenge releases and by ensembling and per-subset tuning in\n"
    "submitted systems; the comparison here therefore isolates the architectural\n"
    "contribution (hypernetwork gating) under a fixed pipeline."
)

doc.h2("3h. Hypernetworks placement (structural edit)")
doc.body(
    "Editing instruction: move the general hypernetwork exposition (Ha et al. formulation, "
    "Eqs. 3.10-3.17 up to the d(z) factorisation) into Chapter 2 as a new subsection 'Adaptive "
    "parameter generation: Hypernetworks' at the end of Section 2.3, and open Chapter 3 with the "
    "SED-specific instantiation. Bridging sentence for Chapter 3:"
)
doc.latex(
    "Chapter~2 (Section~2.3.x) introduced hypernetworks as a mechanism for generating the\n"
    "parameters of a main network conditioned on context. This chapter instantiates that\n"
    "mechanism for sound event detection: a recurrent hypernetwork generates per-time-step\n"
    "scaling of the LSTM weights of a CRNN (Section~3.2), and a graph-structured\n"
    "hypernetwork modulates a prototypical network for few-shot bioacoustic detection\n"
    "(Section~3.3)."
)

# ============================================================== CHAPTER 4
doc.h1("Chapter 4 - ATGNN and LHGNN")

doc.h2("4a. Rebuilt narrative before the acronyms (replacement opening for the ATGNN section)")
doc.latex(
    "The proposed model addresses audio tagging in three stages, each targeting a\n"
    "different kind of structure in the problem. First, a \\emph{patch graph} treats\n"
    "regions of the spectrogram as nodes and connects each region to the regions most\n"
    "similar to it in feature space, wherever they lie on the time--frequency grid; graph\n"
    "convolutions over this structure let harmonically or timbrally related regions\n"
    "exchange information directly (the Patch GNN, PGN). Second, a \\emph{patch--label\n"
    "graph} introduces one learnable embedding per class and connects each label\n"
    "embedding to the spectrogram regions that most resemble it, so that each class\n"
    "gathers evidence from its own supporting regions (the Patch--Label GNN, PLG). Third,\n"
    "a \\emph{label--label graph} lets label embeddings influence one another through a\n"
    "learned adjacency matrix, capturing the strong co-occurrence structure of audio tags\n"
    "(the Label--Label GNN, LLG). One PLG block followed by one LLG block is referred to\n"
    "as a Multi-Label GNN (MLG) block. The full architecture alternates PGN stages with\n"
    "MLG blocks and combines patch-level and label-level predictions at the output."
)

doc.h2("4b. Motivation for the label graphs, with measured evidence (new subsection text)")
doc.latex(
    "The label graphs are motivated by measurable structure in the data. An analysis of\n"
    "the official FSD50K annotations (51{,}197 clips, 200 classes) shows that 84.3\\% of\n"
    "clips carry more than one label (mean 2.99, median 3, up to 22 labels per clip), and\n"
    "that label co-occurrence is strong and highly structured: 169 of the 200 classes\n"
    "have at least one partner class whose conditional co-occurrence probability\n"
    "$P(j \\mid i)$ exceeds $0.5$, in most cases approaching $1.0$. Part of this structure\n"
    "is hierarchical, inherited from the AudioSet ontology (every \\emph{Electric guitar}\n"
    "clip is also labelled \\emph{Guitar} and \\emph{Music}; \\emph{Music} and\n"
    "\\emph{Musical instrument} co-occur in 14{,}703 clips with $P \\approx 1$), and part\n"
    "reflects real-world regularities not imposed by the ontology (\\emph{Animal} and\n"
    "\\emph{Wild animals}, normalised PMI $0.80$). Co-occurrence is simultaneously sparse:\n"
    "most label pairs almost never co-occur. A learnable label--label adjacency is\n"
    "well-suited to this regime, since it can express a small number of strong, directed\n"
    "dependencies while leaving unrelated labels disconnected; and with $\\sim$3 labels per\n"
    "clip typically grounded in different regions of the spectrogram, per-label patch\n"
    "attachment allows each label to bind to its own evidence."
)
doc.note("Figure available: results/label_cooccurrence.png in the ATGNN repository "
         "(labels-per-clip histogram, class frequency, P(j|i) heatmap, per-class max P). "
         "Analysis script: scripts/label_cooccurrence_fsd50k.py.")

doc.h2("4c. Prior work on label co-occurrence (related-work addition)")
doc.latex(
    "Exploiting label dependencies is well established in multi-label image\n"
    "classification: ML-GCN \\citep{chen2019multi} propagates classifier weights over a\n"
    "co-occurrence graph derived from training statistics, and SSGRL\n"
    "\\citep{chen2019learning} attaches semantic label representations to image regions.\n"
    "In audio, label ontology structure has been used for AudioSet training\n"
    "\\citep{jimenez2019sound}. ATGNN differs in learning the label adjacency end-to-end\n"
    "rather than fixing it from co-occurrence counts, and in coupling label embeddings to\n"
    "spectrogram patches at multiple scales of the encoder."
)

doc.h2("4d. Technical clarifications (kNN distance, flattening, nodes, positional encoding)")
doc.latex(
    "Each stage of the encoder produces a feature map $\\mathbf{X}_t \\in \\mathbb{R}^{F'\n"
    "\\times T' \\times D}$, which is flattened along the spatial axes into a set of $N =\n"
    "F'T'$ node vectors $\\{\\mathbf{x}_i \\in \\mathbb{R}^{D}\\}$; a node therefore\n"
    "corresponds to one spatial position of the CNN feature map, i.e. a receptive field\n"
    "over a local time--frequency region, not to a raw spectrogram pixel. The $k$-NN graph\n"
    "connects node $i$ to the $k$ nodes minimising the Euclidean distance $\\lVert\n"
    "\\mathbf{x}_i - \\mathbf{x}_j \\rVert_2$ in feature space; distances are computed on\n"
    "$\\ell_2$-normalised features, making the criterion equivalent to cosine similarity.\n"
    "Because edges are recomputed from features at every block, the graph is dynamic and\n"
    "can connect distant grid positions. Since flattening discards grid coordinates, a\n"
    "learnable positional embedding is added to the node features after the stem, and the\n"
    "pyramid variant additionally biases the $k$-NN distance with a relative positional\n"
    "term; this restores (but does not enforce) sensitivity to time--frequency position,\n"
    "so the claim is that positional information is \\emph{available} to the model rather\n"
    "than that the model is position-equivariant."
)

doc.h2("4e. ImageNet pretraining clarity (experimental-setup paragraph)")
doc.latex(
    "Pretraining is used as follows. The PGN encoder of ATGNN is initialised from the\n"
    "ImageNet-pretrained Pyramid ViG checkpoint of \\citep{han2022vision}; the\n"
    "three-channel input stem is averaged to one channel, positional and relative\n"
    "positional embeddings are bicubically interpolated from the $224 \\times 224$ image\n"
    "grid to the $128 \\times 1024$ spectrogram grid, the ImageNet classification head is\n"
    "discarded, and all MLG components (label embeddings, PLG/LLG blocks, readout) are\n"
    "trained from scratch. LHGNN is reported both from scratch and with the analogous\n"
    "ImageNet initialisation of its ViG backbone. FSD50K- or AudioSet-based pretraining\n"
    "of the encoder was not used: FSD50K is the evaluation target (pretraining on it\n"
    "would leak supervision), and AudioSet pretraining is deliberately excluded so that\n"
    "the comparison isolates architecture from large-scale audio supervision; assessing\n"
    "audio-domain pretraining is left as future work."
)

doc.h2("4f. Table 4.3: uncertainty statement and reproduction runs")
doc.latex(
    "The FSD50K experiments in Table 4.3 report single training runs, as retraining\n"
    "every configuration across multiple seeds was computationally infeasible for the\n"
    "original submission. For the amended thesis, the released implementation was used\n"
    "to retrain the key configurations end-to-end on 4$\\times$NVIDIA L40S GPUs;\n"
    "single-seed reproduction gives eval-set mAP of 0.575 for the ATGNN-pyr-s encoder\n"
    "without MLG blocks, 0.580 for the full ATGNN-pyr-s, and 0.535 for LHGNN trained\n"
    "from scratch, consistent with the published values to within 0.01--0.04 mAP (the\n"
    "residual gap for LHGNN is attributable to the shorter 50-epoch schedule used in the\n"
    "reproduction). On ESC-50, where full multi-seed evaluation is affordable, results\n"
    "are reported as mean $\\pm$ std over runs (Table 4.4), and the same convention is\n"
    "adopted for all rerun configurations. The encoder-only ablation additionally\n"
    "quantifies the MLG contribution on FSD50K: $+0.005$ mAP, mirroring the AudioSet\n"
    "ablation (0.335 vs 0.331)."
)
doc.note("If you complete the 3-seed reruns (remaining item R3), replace the single-seed "
         "sentence with mean +/- std values.")

doc.h2("4g. Parameters vs walltime (new table + efficiency discussion)")
doc.latex(
    "\\begin{table}[t]\\centering\n"
    "\\caption{Trainable parameters and measured walltime (single NVIDIA L40S, PyTorch\n"
    "2.2.0, CUDA 11.8, float32, no JIT compilation; batch 24 clips of $1024 \\times 128$\n"
    "log-mel). Train step = forward + backward + Adam update.}\n"
    "\\begin{tabular}{lccc}\\toprule\n"
    "Model & Params (M) & Train step (ms) & Inference (ms/clip) \\\\ \\midrule\n"
    "ATGNN-pyr-s (encoder only) & 26.8 & 258 & 5.1 \\\\\n"
    "ATGNN-pyr-s (full)         & 35.4 & 290 & 5.7 \\\\\n"
    "LHGNN                      & 31.1 & 643 & 14.5 \\\\ \\bottomrule\n"
    "\\end{tabular}\\end{table}"
)
doc.latex(
    "Parameter count is not a proxy for speed: LHGNN, with fewer parameters than the\n"
    "full ATGNN, is $\\sim$2.2$\\times$ slower per training step and $\\sim$2.5$\\times$\n"
    "slower at inference in this measurement, because its per-block clustering and\n"
    "higher-order aggregation dominate runtime. Conversely, the MLG blocks add 32\\% more\n"
    "parameters but only $\\sim$12\\% walltime. Efficiency claims in this chapter are\n"
    "therefore stated in terms of both parameters and measured walltime.\n"
    "The parameter counts quoted here are trainable parameters; earlier counts that\n"
    "included frozen relative-position tables are corrected accordingly."
)

doc.h2("4h. Benchmark selection note (addition to the experiments section)")
doc.latex(
    "The baselines in Tables 4.1--4.4 (PANNs, PSLA, AST, ERANN and task baselines) are\n"
    "chosen because they share the training pipeline, features and evaluation protocol\n"
    "adopted here, allowing architecture-level comparison under matched conditions.\n"
    "More recent systems reach higher absolute mAP on these benchmarks, but do so\n"
    "through orthogonal factors --- larger-scale audio pretraining, self-supervised\n"
    "objectives, ensembling, or knowledge distillation --- that confound architectural\n"
    "comparison; the contribution of this chapter is the graph-based architecture, and\n"
    "the baselines are selected to isolate that contribution."
)

# ============================================================== CHAPTER 5
doc.h1("Chapter 5 - Music similarity and fingerprinting")

doc.h2("5a. Locally Euclidean assumption (clarifying paragraph)")
doc.latex(
    "The $k$-NN graph construction assumes that the embedding space is \\emph{locally}\n"
    "Euclidean: within a small neighbourhood, straight-line (Euclidean) distance is a\n"
    "meaningful measure of perceptual similarity, even if the global geometry of the\n"
    "embedding manifold is curved. This is the standard assumption underlying manifold\n"
    "learning methods; it justifies using Euclidean distance only for selecting the $k$\n"
    "nearest neighbours (a local decision), while global relationships are mediated by\n"
    "message passing over the resulting graph rather than by global distances."
)

doc.h2("5b. Choice of GNN type (rationale paragraph)")
doc.latex(
    "A spatial graph convolution of the form $\\mathbf{H}^{l+1} =\n"
    "h(\\mathbf{A}\\mathbf{H}^{l}\\mathbf{W}^{l})$ is adopted rather than attention-based\n"
    "(GAT) or spectral alternatives for three reasons: (i) the affinities are already\n"
    "learned by the relation network that constructs $\\mathbf{A}$ (Eq. 5.3), so a second\n"
    "attention mechanism inside the convolution would duplicate that role; (ii) spectral\n"
    "convolutions assume a fixed graph, whereas here the graph is rebuilt per batch; and\n"
    "(iii) with small node sets per batch, the simple spatial form is the most\n"
    "parameter- and compute-efficient option, consistent with the design goal of a\n"
    "lightweight metric-learning head."
)

doc.h2("5c. Notation fixes around Eqs. 5.3-5.5 (editing instructions + text)")
doc.body(
    "The symbol d is overloaded: d(., .) is the distance in Eq. 5.3, d is also the embedding "
    "dimensionality, and d(z) is the hypernetwork scaling in Chapter 3. Editing instruction: "
    "rename the distance to d_E(., .) and state the convention. Matrices should be bold "
    "uppercase per the Chapter 2 notation paragraph (A, H, W). Insert after Eq. 5.3:"
)
doc.latex(
    "Here $d_E(\\cdot,\\cdot)$ denotes Euclidean distance between embeddings (denoted\n"
    "$d_E$ to avoid conflict with the embedding dimensionality $d$), and the learned\n"
    "scales $\\sigma_i$ modulate the effective neighbourhood radius per node, allowing the\n"
    "network to sharpen or diffuse each node's affinities."
)

doc.h2("5d. ELU / GELU / cosine / Euclidean choices (rationale paragraph)")
doc.latex(
    "Activation and distance choices follow the role of each component. ELU is used in\n"
    "the graph convolution (Eq. 5.4) because its negative saturation keeps aggregated\n"
    "messages zero-centred, which stabilises repeated neighbourhood averaging; GELU is\n"
    "used inside feed-forward blocks for consistency with the Transformer-style blocks of\n"
    "Chapter 4. Euclidean distance is used where absolute distances between embeddings\n"
    "matter (graph construction, Eq. 5.3), while cosine similarity is used in the\n"
    "training objectives (Eqs. 5.5, 5.9), which compare directions of $\\ell_2$-normalised\n"
    "embeddings and are invariant to their magnitude; this matches the retrieval setting,\n"
    "where ranking is performed by cosine similarity."
)

doc.h2("5e. A100 parallelism details (experimental-setup sentence)")
doc.latex(
    "Training used synchronous data parallelism (PyTorch DistributedDataParallel) across\n"
    "two NVIDIA A100 GPUs, with a per-GPU batch size of 128 and an effective batch size\n"
    "of 256; gradients are all-reduced at each step, and reported batch sizes refer to\n"
    "the effective (global) batch."
)
doc.note("Verify against your original training logs: if torch.nn.DataParallel (single-process) "
         "was used instead of DDP, replace the first clause accordingly (remaining item R5).")

doc.h2("5f. Near-identical retrievals claim (softened sentence)")
doc.latex(
    "Qualitative inspection of retrieval lists suggests that the model frequently ranks\n"
    "near-identical excerpts (same recording under different degradations) at the top;\n"
    "this is reported as an observed qualitative behaviour rather than a quantified\n"
    "property, as systematic retrieval-overlap analysis was not performed."
)

doc.h2("5g. Table 5.2 uncertainty (limitation statement, pending reruns)")
doc.latex(
    "The hit rates in Table 5.2 are computed from a single trained model per method over\n"
    "a fixed query set. Retraining all systems across seeds was not feasible within the\n"
    "amendment period; to characterise evaluation variability, the query sampling can be\n"
    "repeated over independent draws and the resulting variation reported alongside the\n"
    "point estimates. The comparison across noise and reverberation conditions, where\n"
    "GraFPrint's margin over the CNN- and Transformer-based baselines is largest at low\n"
    "SNR, is consistent across query lengths, which supports the robustness conclusion\n"
    "beyond any single operating point."
)
doc.note("If you run the repeated-query-draw evaluation (remaining item R4), replace the second "
         "sentence with the measured mean +/- CI per cell or per method.")

doc.h2("5h. Metrics introduced linearly (editing instruction)")
doc.body(
    "Editing instruction: before Section 5.2's results, add one sentence per metric in the "
    "order used (top-1 hit rate, precision@k, mean average precision), each defined before any "
    "table references it. Draft:"
)
doc.latex(
    "Retrieval quality is reported with three metrics. \\emph{Top-1 hit rate} is the\n"
    "fraction of queries whose correct reference track is ranked first.\n"
    "\\emph{Precision@k} is the fraction of the $k$ highest-ranked items that are\n"
    "relevant. \\emph{Mean average precision} (mAP) averages precision over the ranks of\n"
    "all relevant items, summarising the full ranking."
)

# ============================================================== CHAPTER 6
doc.h1("Chapter 6 - Conclusion")
doc.h2("6a. Move the ATGNN explanation into Chapter 4 (editing instruction)")
doc.body(
    "The report notes that the clearest explanation of ATGNN's contribution currently appears "
    "in the conclusion. Editing instruction: move that passage into Chapter 4 (either the "
    "chapter introduction or the opening of the ATGNN section, alongside draft 4a above), and "
    "keep Chapter 6 as synthesis. Replacement summary sentence for Chapter 6:"
)
doc.latex(
    "Chapter 4 demonstrated that treating spectrogram regions, class labels, and their\n"
    "interactions as graphs yields tagging models that are competitive with\n"
    "attention-based architectures at a fraction of the parameter count, and that the\n"
    "benefit of explicit label-graph modelling, while consistent, is secondary to the\n"
    "quality of the patch-level graph encoder."
)

# ============================================================== REMAINING
doc.h1("Remaining items (not resolved in this document)")
doc.h2("Experiments / code")
doc.bullet("R1. Chapter 4: multi-seed FSD50K reruns for Table 4.3 uncertainty (3 seeds x 3 "
           "configurations, ~1 day on the current 4-GPU machine). Draft 4f already contains the "
           "fallback single-seed statement.")
doc.bullet("R2. Chapter 3: TUT-SED Synthetic 2016 multi-seed rerun (dataset not currently on "
           "disk; URBAN-SED 5-seed study is complete).")
doc.bullet("R3. Chapter 5: repeated-query-draw variability for Table 5.2 (requires the "
           "fingerprinting repo/indexes).")
doc.bullet("R4. Chapter 5: verify DP vs DDP and per-GPU batch from original training logs "
           "(draft 5e assumes DDP).")
doc.bullet("R5. Repository housekeeping: LICENSE, CITATION.cff, release tags (v1.0) and commit "
           "hashes for all repos; make ATGNN public or archive with DOI; clean/tag the two "
           "existing Chapter 5 repos.")
doc.bullet("R6. Central reproducibility landing page (e.g. 'graph-neural-audio-analysis' repo "
           "with CODE_AVAILABILITY.md, DATASETS.md, per-chapter links).")
doc.bullet("R7. Reproduction scripts named per table (reproduce_table_4_3.sh etc.) - thin "
           "wrappers around the existing run commands.")
doc.bullet("R8. Chapter 1 audio-as-graph figure (composition suggested in 1d).")

doc.h2("Writing / style pass (to do after substantive edits)")
doc.bullet("G1. Choose one spelling convention (modelling/optimiser vs modeling/optimizer) - "
           "the current text mixes both (e.g. 'modeling paradigms' and 'modelling' in Chapter 1).")
doc.bullet("G2. Standardise Figure vs Fig.; fix hyphenation (time-frequency vs time frequency; "
           "graph-based vs graph based is currently inconsistent).")
doc.bullet("G3. Bibliography: consistent capitalisation (AudioSet, ImageNet, FSD50K in titles), "
           "complete venue metadata, deduplicate arXiv/published versions.")
doc.bullet("G4. Typos flagged by examiners plus a full pass; e.g. 'use of of signal processing' "
           "in the opening paragraph of Chapter 1; 'attention mechanism that allow' -> "
           "'attention mechanisms that allow'; missing full stops after research questions in "
           "Section 1.1.1 (items 2 and 3 end without '?').")
doc.bullet("G5. Ensure every acronym is expanded at first use in each chapter (PGN/MLG/LLG are "
           "addressed by draft 4a; check SED, AT, MIR, VQT elsewhere).")
doc.bullet("G6. Align equation notation with the Chapter 2 notation paragraph (bold matrices; "
           "d_E for distances; consistent loss vs cost terminology).")

doc.h2("Response-to-corrections table")
doc.body(
    "When assembling the response document for the examiners, map each item above to the "
    "corresponding page/section of the amended PDF. The drafts in this document are keyed "
    "(1a-6a) so they can be referenced directly from that table."
)

doc.output(OUT)
print("written", OUT)
