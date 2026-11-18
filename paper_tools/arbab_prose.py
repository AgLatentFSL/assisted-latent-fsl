"""Rewrites the prose of the appendices in the voice of the AgEval and Assisted Few-Shot Learning papers
(skill ~/.claude/skills/write-like-arbab). Run it after the generators (make_appendix.py, make_layer_vlm.py,
make_layer_vlm_allk.py); appendix_tests.tex is written in this voice by make_layer_tests.py itself.
It changes words only: the set of numbers of each file must be the same before and after, or the file is not written.
A rule that is already applied is skipped; a rule that matches nothing is reported.
Usage (from the paper root): <python> tools/arbab_prose.py
"""
import os, re
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda s: s          # readability
LINES = {                # file -> [(prefix of the line, new line)]
"appendix.tex": [
 ("This appendix reports the retrieval analysis", R(r"This appendix provides the full retrieval analysis behind Sec.~\ref{sec:latent} and Sec.~\ref{sec:discussion}, encompassing every layer of nine encoders, the five datasets, and shot counts of 1, 2, 4 and 8. The analysis measures retrieval only, and every layer is compared with the pooled output of its own encoder. Appendix~\ref{app:vlm} provides the evaluation with the VLM.")),
 (r"\textbf{Disjoint queries and references.}", R(r"\textbf{Disjoint queries and references.} For each of the five datasets, we sampled 100 test queries and a separate, class-balanced reference bank of 100 images (seed 42), resulting in 500 queries in total. The queries are the AgEval pool of Sec.~\ref{sec:setup}, completed to 100 images, while the bank is drawn from the rest of the dataset. This ensures that no image is in both sets and that no bank image is byte-identical to a query. Note that this split differs from the leave-one-out pools of the supporting experiments (Appendix~\ref{app:support}), so the numbers of this appendix are not the numbers of Fig.~\ref{fig:layers}. Queries and references are not separated by capture date; Sec.~\ref{app:holdout} adds a date-disjoint check for the ViT.")),
 (r"\textbf{Encoders and layers.}", R(r"\textbf{Encoders and layers.} Table~\ref{tab:app-enc} lists the nine encoders. We read a layer at the CLS token where the model has one, at the mean of the patch tokens otherwise, and at the averaged feature map for ResNet-50, applying the final layer normalization of each transformer before the readout. Together, the encoders have 125 layers; each encoder also has a \emph{final output}, the pooled vector that it exposes for retrieval. The pooled ViT output is the retriever employed by assisted few-shot learning~\cite{arshad2025assisted}. The baseline of a layer is always the final output of its own encoder; outputs of different encoders are not compared. All vectors are normalized to unit length and ranked using cosine similarity.")),
 (r"\textbf{Measures.}", R(r"\textbf{Measures.} For a shot count $K$, \emph{slots matching} is the share of the top-$K$ reference slots filled by a reference of the query's class, and \emph{coverage} is the share of queries with at least one such reference. \emph{Replaced} is the share of the top-$K$ slots that hold a reference that the previous layer did not retrieve. To quantify the difference from the pooled output of the same encoder at the same $K$, we consider a query \emph{gained} if the layer supplies a same-class reference where the pooled list has none, and \emph{lost} if the layer loses every same-class reference of the pooled list. The \emph{net coverage change} is the gained rate minus the lost rate over all paired queries. Intervals are 95\% paired bootstrap intervals over the 500 queries (10{,}000 resamples).")),
 ("Between adjacent layers 35.0", R(r"Adjacent layers retrieve different references: 35.0\% ($K=1$), 31.2\% ($K=2$), 27.1\% ($K=4$), and 23.2\% ($K=8$) of the reference slots are replaced between adjacent layers (Fig.~\ref{fig:app-adj}), and the ordered list is identical for 65.0\%, 36.8\%, 10.4\%, and 0.7\% of the queries, respectively. The lists do not reach a plateau: at $K=4$, the ordered list takes its last-layer form only at the last layer for 89.6\% to 99.8\% of the queries, depending on the encoder (Fig.~\ref{fig:app-settle}a). Note that this analysis describes agreement between layers and not robustness to changes of the input image.")),
 (r"Figures~\ref{fig:app-ds} and~\ref{fig:app-dsnet} repeat", R(r"Dataset-specific results are provided in Figures~\ref{fig:app-ds} and~\ref{fig:app-dsnet}, which repeat the depth maps for each dataset, as the share of matching slots and as the net coverage change against the pooled output of the same encoder. Figure~\ref{fig:app-settle}b provides the final outputs by dataset. Notably, the first layers of every encoder are below the pooled output of that encoder on every dataset.")),
 ("The split above does not separate capture dates.", R(r"The split above does not separate capture dates. For DeepWeeds, the only dataset with a capture date, we conducted a separate check with date-disjoint banks for the ViT of the original pipeline. On a development split with 180 queries and ten banks of 99 images, we compared every hidden layer of the ViT with its pooled output at every shot count. The largest net coverage change of a ViT layer is $+$3.9\pp{} (block 11, $K=8$). This suggests that the gain of a hidden layer of the ViT over its pooled output is smaller under date separation than on the split above, where block 10 gains $+$7.8\pp{} at $K=1$. The check covers one dataset and is exploratory.")),
],
"appendix_more.tex": [
 ("The main text reports six encoders.", R(r"The main text reports six encoders. DINOv2, BioCLIP, and DINOv3 are reported in the appendix only: Appendix~\ref{app:atlas} provides their layers in retrieval and Appendix~\ref{app:vlm} with the VLM. This appendix provides the earlier experiments that employed the final CLS token of DINOv2 ViT-B/14~\cite{oquab2024dinov2} as the retriever. These results are not compared with the results of other encoders.")),
 (r"\textbf{AgEval protocol.}", R(r"\textbf{AgEval protocol.} On the 493 queries of Sec.~\ref{sec:main} with four references, assisted few-shot learning with DINOv2 retrieval achieves 60.0\% accuracy, compared to 52.7\% for random examples (72 wins, 36 losses, $p<0.001$). The references carry the label of the query on 2.68 of four slots.")),
 (r"\textbf{Split audit on DeepWeeds.}", R(r"\textbf{Split audit on DeepWeeds.} With the banks of Sec.~\ref{sec:shift}, the same-date bank increases the same-label references of four from 1.88 to 2.25 (+0.36, 95\% CI 0.28 to 0.45), in all 20 seeds. Same-date images take 23.0\% of the retrieved slots. Of the same-date references, 96\% carry the label of the query, compared to 44\% of the references from other dates.")),
 (r"\textbf{Group-disjoint banks.}", R(r"\textbf{Group-disjoint banks.} Table~\ref{tab:shift} provides two evaluations. On 63 held-out DeepWeeds queries with a date-disjoint bank, assisted few-shot learning is correct on 26 and zero-shot prompting on 21 (11 wins, 6 losses, $p=0.332$). Paddy Doctor~\cite{petchiammal2023paddy} has ten rice classes and the rice variety of each image. We sampled 50 queries from two varieties and a bank of 110 images from the eight other varieties. Assisted few-shot learning increases the accuracy of Claude Sonnet 5.5 from 22\% to 38\% (10 wins, 2 losses, $p=0.039$), while Claude Opus 5.5 remains at 44\% (3 wins, 3 losses). On a prespecified subset of 20 Paddy Doctor queries, all 20 repeated requests with the assisted prompt returned the same label for both VLMs.")),
],
"appendix_support.tex": [
 ("This appendix reports the experiments that preceded", R(r"This appendix provides the experiments that preceded the layer evaluation of the main text. These experiments follow the original protocol of AgEval and assisted few-shot learning, with four references and the pooled ViT output as the retriever. They demonstrate that retrieved examples help, that the improvement depends on references with the label of the query, and that same-date references inflate retrieval.")),
 (r"\textbf{AgEval protocol.}", R(r"\textbf{AgEval protocol.} We follow the evaluation of AgEval and assisted few-shot learning~\cite{arshad2024ageval,arshad2025assisted}. For each dataset, the AgEval loader samples a class-balanced pool of about 100 images with seed 42. Every pool image is a query and retrieves from the rest of the pool, resulting in 493 queries across the five datasets. Each query is evaluated with three prompts: zero-shot; four random examples; and assisted few-shot learning of Arshad \etal with retrieval by the pooled ViT output. Paired comparisons of accuracy employ the exact McNemar test, and paired comparisons of same-label references employ the Wilcoxon signed-rank test~\cite{wilcoxon1945individual}. Macro-F1 uses the fixed label set of the dataset.")),
 (r"\textbf{Retrieval quality.}", R(r"\textbf{Retrieval quality.} To quantify retrieval quality, we count the number of the four references that carry the label of the query (\emph{same-label references}, 0 to 4). Two retrievers are compared on one query by the \emph{turnover} of their references, the number of the four references of one retriever that the other does not return (0 to 4).")),
 (r"\finding{Finding S1}", R(r"\finding{Finding S1}{Retrieved examples outperform random examples and zero-shot prompting, and random examples do not help.}")),
 (r"Figure~\ref{fig:main} and Table~\ref{tab:main} give the result.", R(r"Figure~\ref{fig:main} and Table~\ref{tab:main} provide the result. Across the 493 queries, assisted few-shot learning is correct on 61.9\%, random examples on 52.7\%, and zero-shot prompting on 51.3\%. Assisted few-shot learning has 78 wins and 33 losses against random examples and 82 wins and 30 losses against zero-shot prompting ($p<0.001$ for both). The improvement over random examples is most pronounced on DeepWeeds (30 wins, 5 losses, $p<0.001$) and Soybean Diseases (14 wins, 1 loss, $p=0.001$), while it is absent on Bean Leaf Lesions (7 wins, 12 losses, $p=0.359$), where zero-shot prompting already reaches 75\%. Random examples do not help over zero-shot prompting (51 wins, 44 losses, $p=0.538$).")),
 (r"\finding{Finding S2}", R(r"\finding{Finding S2}{The accuracy of the VLM increases with the number of same-label references.}")),
 (r"Figure~\ref{fig:dose} groups", R(r"Figure~\ref{fig:dose} groups the 493 assisted prompts by the number of references that carry the label of the query. The accuracy is 40\% with no same-label reference, 41\% with one, 57\% with two, 68\% with three, and 78\% with four. Part of this trend reflects the difficulty of the query, because zero-shot prompting on the same queries increases from 38\% to 66\%. The improvement of the examples over zero-shot prompting is 2\pp{} with no same-label reference, 10\pp{} with one and with two, 14\pp{} with three, and 13\pp{} with four. These results indicate that examples without the label of the query do not help, and that the improvement appears from one same-label reference on.")),
 ("The pooled ViT output returns 2.49", R(r"The pooled ViT output retrieves 2.49 same-label references of four on average. For 58 of the 493 queries (12\%), none of the four references carries the label of the query, and the VLM gains 2\pp{} on these queries.")),
 (r"\textbf{Insight.} Same-label references are necessary", R(r"\textbf{Insight.} Same-label references are necessary for an improvement. This motivates the question of the main text: whether a hidden layer retrieves a same-class reference where the pooled output of its encoder retrieves none.")),
 ("The AgEval protocol draws queries", R(r"The AgEval protocol samples queries and references from one pool and does not separate source groups. However, field datasets are collected in sessions, and a query can then retrieve images of its own session.")),
 (r"\finding{Finding S7}", R(r"\finding{Finding S7}{With date-disjoint banks the improvement from the examples is smaller, and a hidden layer of the ViT does not change the accuracy.}")),
],
"appendix_vlm.tex": [
 ("This appendix gives every condition of the first downstream test", R(r"This appendix provides every condition of the first downstream evaluation, which covers nine encoders in the 2-shot setting. Its answers for the six encoders of the main text are the 2-shot results of Sec.~\ref{sec:vlm}, and Appendix~\ref{app:vlmk} provides the other shot counts. The evaluation employs 500 queries with the disjoint reference bank of Appendix~\ref{app:atlas}, two references per query, the standard assisted prompt, and Claude Sonnet 5.5.")),
],
"appendix_main_tables.tex": [], "appendix_vlm_allk.tex": [],
}
FRAGS = {                # file -> [(old fragment, new fragment)], each old fragment once
"appendix.tex": [
 (r"Figure~\ref{fig:app-depth} maps the share of matching slots over encoders and depth.", r"Figure~\ref{fig:app-depth} illustrates the share of matching slots across encoders and depth."),
 ("The hidden layers above the pooled output of their encoder with an interval that excludes zero are, as the best layer of an encoder:", "Our analysis reveals the following hidden layers above the pooled output of their encoder with an interval that excludes zero, as the best layer of an encoder:"),
 ("The last block of DINOv2 and of DINOv3 returns the references of the final output, and no earlier block", "Conversely, the last block of DINOv2 and of DINOv3 returns the references of the final output, and no earlier block"),
 ("give the share of matching slots and the replacement at every layer", "provide the share of matching slots and the replacement at every layer"),
 ("The ranking is computed on the queries it describes, so the leading layers are candidates and not validated choices.", r"It is important to note that the ranking is computed on the queries that it describes, so the leading layers are candidates and not validated choices. This ranking is the screening step that selects the five layers evaluated with the VLM (Sec.~\ref{sec:method})."),
 ("against its pooled output. at $K=1$", "against its pooled output: at $K=1$"),
 ("A small net change is the balance of a much larger exchange of queries, and the room for a gain shrinks as the budget grows.", "These counts highlight that a small net change is the balance of a much larger exchange of queries, and that the room for improvement shrinks as the shot count grows."),
 (r"Figure~\ref{fig:app-trf} gives the gained", r"Figure~\ref{fig:app-trf} provides the gained"),
],
"appendix_main_tables.tex": [
 (r"Table~\ref{tab:mean5} gives the values", r"Table~\ref{tab:mean5} provides the values"), (r"and Table~\ref{tab:latent} gives the values", r"and Table~\ref{tab:latent} provides the values"),
],
"appendix_support.tex": [
 ("This section measures retrieval only, on the AgEval pools, and it compares", "This analysis measures retrieval only, on the AgEval pools, and compares"),
 ("A choice of the layer therefore changes which images the VLM receives.", "These findings underscore that the choice of the layer changes which images the VLM receives."),
 (r"Figure~\ref{fig:layers} gives the same-label references", r"Figure~\ref{fig:layers} provides the same-label references"),
 ("The pooled ViT output that assisted few-shot learning uses returns 2.49.", "The pooled ViT output employed by assisted few-shot learning retrieves 2.49."),
 ("both return 2.76", "both retrieve 2.76"),
 ("The ViT is therefore best before its last block.", "This suggests that the ViT retrieves best before its last block."),
 (r"Figure~\ref{fig:latent} shows the patch tokens", r"Figure~\ref{fig:latent} illustrates the patch tokens"),
 ("The same-date bank raises the same-label references", "The same-date bank increases the same-label references"),
 (r"against 27\% of the references from other dates", r"compared to 27\% of the references from other dates"),
 (r"Figure~\ref{fig:heat} shows that two classes", r"Figure~\ref{fig:heat} reveals that two classes"),
 (r"against 1.98 in the AgEval pool, where the gain over", "compared to 1.98 in the AgEval pool, where the improvement over"),
 ("A second test compares a hidden layer", "A second evaluation compares a hidden layer"),
],
"appendix_vlm.tex": [
 ("The test has 55 conditions:", "The evaluation comprises 55 conditions:"),
 ("Each layer is compared with the pooled output of its encoder on the same queries: accuracy change", "We compare each layer with the pooled output of its encoder on the same queries, using the accuracy change"),
 (r"Table~\ref{tab:app-vlm} and Fig.~\ref{fig:app-vlm} give all layers.", r"Table~\ref{tab:app-vlm} and Fig.~\ref{fig:app-vlm} provide the results for all layers:"),
 ("The comparison at the level of the encoder does not depend on the choice of one layer:", "Notably, the comparison at the level of the encoder does not depend on the choice of one layer:"),
],
"appendix_vlm_allk.tex": [
 ("This appendix lists every condition of the test of", "This appendix provides every condition of the evaluation of"),
 ("over the pooled output of the same encoder at that budget. The test has 145 conditions", r"over the pooled output of the same encoder at that budget (the screening step of Sec.~\ref{sec:method}). The evaluation comprises 145 conditions"),
 (r"Figure~\ref{fig:vlmall} shows every condition, and Fig.~\ref{fig:vlm} shows the pooled output", r"Figure~\ref{fig:vlmall} illustrates every condition, and Fig.~\ref{fig:vlm} compares the pooled output"),
],
"appendix_more.tex": [],
}
GLOBAL = [("budgets of 1, 2, 4 and 8 references", "shot counts of 1, 2, 4 and 8"), ("budget of 1, 2, 4 and 8 references", "shot count of 1, 2, 4 and 8"), ("budgets", "shot counts"), ("budget", "shot count"),
          ("What the demonstrations add", "What the Examples Add"), ("demonstrations", "examples"), ("published pipeline", "original pipeline")]
nums = lambda s: set(re.findall(r"\d+(?:\{,\}\d+)*(?:\.\d+)?", re.sub(r"\b[1248]-shot", "", s)))
for f in LINES:
    p = os.path.join(HERE, f); old = open(p).read(); L = old.split("\n"); miss = []
    for pre, new in LINES[f]:
        hit = [i for i, l in enumerate(L) if l.startswith(pre)]
        if len(hit) == 1: L[hit[0]] = new
        else: miss.append(pre[:50])
    s = "\n".join(L)
    for a, b in FRAGS[f]:
        if s.count(a) == 1: s = s.replace(a, b)
        elif b not in s and re.sub("budget", "shot count", b) not in s: miss.append(a[:50])
    for a, b in GLOBAL: s = s.replace(a, b)
    if nums(s) != nums(old): print(f, "NOT WRITTEN, numbers differ:", sorted(nums(old) ^ nums(s))); continue
    open(p, "w").write(s); print(f, "ok" if s != old else "unchanged", "| no match:", miss)

# Second stage (2026-11-16): sentence-by-sentence pass against writing-style-guide.md, applied to the rewritten text.
FRAGS2 = {
"appendix.tex": [
 ("This ensures that no image is in both sets", "This sampling ensures that no image is in both sets"),
 (r"To quantify the difference from the pooled output of the same encoder at the same $K$, we consider a query \emph{gained} if the layer supplies", r"We consider a query \emph{gained} against the pooled output of the same encoder at the same $K$ if the layer supplies"),
 ("the coverage of the hidden layer of the same encoder that has the largest net coverage change. Our analysis reveals the following hidden layers above the pooled output of their encoder with an interval that excludes zero, as the best layer of an encoder:", "the coverage of the hidden layer with the largest net coverage change in the same encoder. The best hidden layers that are above the pooled output of their encoder with an interval that excludes zero are the following:"),
 ("At $K=1$, 19 of them are above that output with an interval that excludes zero and 52 are below it; at $K=2$, 13 and 48; at $K=4$, 3 and 44; at $K=8$, 4 and 40.", "The numbers of layers above and below that output with an interval that excludes zero are 19 and 52 at $K=1$, 13 and 48 at $K=2$, 3 and 44 at $K=4$, and 4 and 40 at $K=8$."),
 (r"This ranking is the screening step that selects the five layers evaluated with the VLM (Sec.~\ref{sec:method}).", r"The five layers evaluated with the VLM are selected from this ranking (Sec.~\ref{sec:method})."),
 ("This suggests that the gain of a hidden layer of the ViT", "This result suggests that the gain of a hidden layer of the ViT"),
],
"appendix_vlm.tex": [
 ("Its answers for the six encoders of the main text are", "The answers of this evaluation for the six encoders of the main text are"),
 ("wins, losses and the exact McNemar test.", "wins, losses, and the exact McNemar test."),
 ("the mean accuracy of the five tested layers against the pooled output is", "the change in the mean accuracy of the five tested layers against the pooled output is"),
],
"appendix_support.tex": [
 ("Every pool image is a query and retrieves from the rest of the pool, resulting in", "Every pool image is a query, and its references are retrieved from the rest of the pool, resulting in"),
 ("To quantify retrieval quality, we count the number of the four references that have the label of the query", "We quantify retrieval quality by the number of the four references that have the label of the query"),
 ("This motivates the question of the main text:", "This finding motivates the question of the main text:"),
 ("1.08 for the CLS token of the ViT and 1.39 for the CLS token of CLIP.", "1.08 for the CLS token of the ViT, and 1.39 for the CLS token of CLIP."),
 ("This suggests that the ViT retrieves best before its last block.", "This comparison suggests that the ViT retrieves best before its last block."),
 ("The same-date bank increases the same-label references of four of the pooled ViT output from", "The same-date bank increases the same-label references (of four) retrieved by the pooled ViT output from"),
 ("carry most of the effect", "account for most of the effect"),
],
"appendix_more.tex": [
 (r"Appendix~\ref{app:atlas} provides their layers in retrieval and Appendix~\ref{app:vlm} with the VLM.", r"Appendix~\ref{app:atlas} provides their layers in retrieval, and Appendix~\ref{app:vlm} provides them with the VLM."),
],
}
GLOBAL2 = [("carry the label", "have the label"), ("carries the label", "has the label")]
for f in LINES:
    p = os.path.join(HERE, f); old = open(p).read(); s = old; miss = []
    for a, b in GLOBAL2: s = s.replace(a, b)
    for a, b in FRAGS2.get(f, []):
        if s.count(a) == 1: s = s.replace(a, b)
        elif b not in s: miss.append(a[:50])
    if nums(s) != nums(old): print(f, "stage 2 NOT WRITTEN, numbers differ:", sorted(nums(old) ^ nums(s))); continue
    open(p, "w").write(s); print(f, "stage 2", "ok" if s != old else "unchanged", "| no match:", miss)
