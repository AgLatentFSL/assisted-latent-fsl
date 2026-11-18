"""Shared figure style matched to bioMoR Figure 5 (injection-site ablation).

Observed from the vector data of papers/2026_biomor_AAAI/figs/ablation_bar.pdf (2026-11-07):
  fills #D9D9D9, #A9D3EB, #DCC3E8, #BFDFA6, #F7B79B; white bar edges and white cross-hatching ("xx");
  value labels rotated 90 degrees above each bar; frameless legend above the axes; dashed light-gray horizontal
  gridlines only; gray (#909090) left and bottom spines, none on top or right; DejaVu Sans at 5.5-6.5 pt.
Recommended mapping onto CorpusAgent semantics (hue families of the Okabe-Ito palette used elsewhere):
  baselines gray, zero-shot agent blue, Agent + SFT lavender, Agent + SFT + GRPO green, RAG-SFT salmon,
  own box blue, learned box amber. Line and scatter plots take the same font, spines and grid without hatching.
"""
import matplotlib.pyplot as plt

SPINE, GRID, INK = "#909090", "#D9D9D9", "#222222"
P = {"ref": "#D9D9D9", "ref2": "#EDEDED", "blue": "#A9D3EB", "blue_lt": "#D6EAF5", "blue_mid": "#C2DFF0",
     "lavender": "#DCC3E8", "green": "#BFDFA6", "green_lt": "#E0F0D4", "salmon": "#F7B79B", "amber": "#F9D58B",
     "dark": "#9A9A9A"}
SYSTEM = {"B0": "#EDEDED", "B0_SFT": "#E2E2E2", "B1": P["ref"], "B1_SFT": P["salmon"], "ragsft": P["salmon"],
          "zs": P["blue"], "sft": P["lavender"], "grpo": P["green"], "B2s": P["blue_lt"], "B2d": P["blue_mid"],
          "own": P["blue"], "learned": P["amber"], "random": P["ref"], "nocrop": P["dark"]}
# saturated partners for lines and markers (same hue families)
LINE = {"zs": "#9DCBEB", "sft": "#A38FCB", "grpo": "#5DB28C", "ragsft": "#E0906B", "B1": "#7F7F7F", "own": "#5E9CCF",
        "learned": "#EDB55E"}

RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 6.5, "axes.titlesize": 6.8,
      "axes.labelsize": 6.8, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
      "axes.edgecolor": SPINE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
      "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.6,
      "axes.axisbelow": True, "xtick.color": SPINE, "ytick.color": SPINE, "xtick.labelcolor": INK,
      "ytick.labelcolor": INK, "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False,
      "hatch.linewidth": 0.5, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"}


def apply():
    plt.rcParams.update(RC)


def bar(ax, x, h, w, color, hatch="xx", **kw):
    """A bioMoR bar: pastel fill, white edge, white cross-hatch."""
    kw.setdefault("edgecolor", "white")
    kw.setdefault("linewidth", 0.6)
    return ax.bar(x, h, w, color=color, hatch=hatch, **kw)


def vlabel(ax, x, y, text, fs=5.0, pad=0.6):
    """Value label rotated 90 degrees just above a bar, as in bioMoR."""
    ax.text(x, y + pad, text, rotation=90, ha="center", va="bottom", fontsize=fs, color=INK)


def top_legend(ax, handles, ncol=3, y=1.02):
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.0)


def patch(color, label, hatch="xx"):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, edgecolor="white", hatch=hatch, label=label)


# ---- panel letters, as in bioMoR Figure 6 ("a.", "b.", ...), added to every multi-panel figure at save time ----
def _letter_axes(fig):
    axs = [a for a in fig.axes if a.get_label() != "<colorbar>" and a.get_visible() and a.has_data()
           and a.get_position().width > 0.12 and a.get_position().height > 0.12]
    if len(axs) < 2 or getattr(fig, "_ca_lettered", False):
        return
    axs.sort(key=lambda a: (round(-a.get_position().y1, 1), a.get_position().x0))
    for i, a in enumerate(axs):
        a.text(-0.02, 1.0, f"{'abcdefgh'[i]}.", transform=a.transAxes, ha="right", va="bottom",
               fontsize=7.5, fontweight="bold", color=INK, clip_on=False)
    fig._ca_lettered = True


def _patch_savefig():
    import matplotlib.figure as mf
    if getattr(mf.Figure.savefig, "_ca", False):
        return
    orig = mf.Figure.savefig

    def savefig(self, *a, **k):
        _letter_axes(self)
        return orig(self, *a, **k)
    savefig._ca = True
    mf.Figure.savefig = savefig


_orig_apply = apply


def apply():  # noqa: F811
    _orig_apply()
    _patch_savefig()
