# requires: pip install datasets
from IPython.display import display
import os, numpy as np, pandas as pd, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from scipy.stats import entropy

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.max_colwidth", 110); pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
print(DEV, torch.__version__)

from transformers import AutoTokenizer, AutoModelForMaskedLM
import transformers

MODEL_ID = "google-bert/bert-base-multilingual-cased"
REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"   # pinned checkpoint
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=REVISION).eval().to(DEV)
torch.set_grad_enabled(False)
assert len(tok) == model.config.vocab_size
print(transformers.__version__, model.config.hidden_act, sum(p.numel() for p in model.parameters()))

from huggingface_hub import login
login()
from datasets import load_dataset

def load_flores(code, split):
    "One FLORES+ language variety as a DataFrame sorted by sentence id."
    df = load_dataset("openlanguagedata/flores_plus", code, split=split).to_pandas()
    df["id"] = df["id"].astype(int)
    return df.sort_values("id").reset_index(drop=True)

LANGS = ["eng_Latn", "fra_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hans", "yor_Latn"]
EXTRA = ["kaz_Cyrl"]
ALL = LANGS + EXTRA
dev = {c: load_flores(c, "dev") for c in ALL}
devtest = {c: load_flores(c, "devtest") for c in ALL}
assert all((dev[c].id == dev["eng_Latn"].id).all() and (devtest[c].id == devtest["eng_Latn"].id).all() for c in ALL)
print({c: (len(dev[c]), len(devtest[c])) for c in ALL})

def encode(sentences, max_length=256):
    enc = tok(sentences, padding=True, truncation=True, max_length=max_length,
              return_tensors="pt", return_special_tokens_mask=True)
    special = enc.pop("special_tokens_mask").bool()
    keep = enc["attention_mask"].bool() & ~special
    return enc, keep

def sentence_reps(sentences, batch_size=32):
    "Mean-pooled hidden states of all layers: (13, N, 768)."
    chunks = []
    for i in range(0, len(sentences), batch_size):
        enc, keep = encode(sentences[i:i + batch_size])
        enc, keep = enc.to(DEV), keep.to(DEV)
        H = torch.stack(model.bert(**enc, output_hidden_states=True).hidden_states)
        w = keep.unsqueeze(-1).float()
        chunks.append(((H * w).sum(2) / w.sum(1)).cpu())
    return torch.cat(chunks, dim=1).numpy()

class FFNHooks:
    "Counts positive GELU activations and/or zeroes selected FFN neurons."
    def __init__(self, model, ablate=None):
        self.layers = model.bert.encoder.layer
        self.counts = torch.zeros(len(self.layers), model.config.intermediate_size, device=DEV)
        self.n_tokens = 0
        self.keep = None
        self.ablate = {i: torch.as_tensor(v, dtype=torch.long, device=DEV) for i, v in (ablate or {}).items()}

    def _hook(self, i):
        def hook(module, inputs, output):            # output: (B, T, 3072), after GELU
            if self.keep is not None:
                self.counts[i] += ((output > 0) & self.keep.unsqueeze(-1)).sum(dim=(0, 1))
            idx = self.ablate.get(i)
            if idx is not None and len(idx) > 0:
                output = output.clone()
                output[..., idx] = 0.0
            return output
        return hook

    def __enter__(self):
        self.handles = [l.intermediate.register_forward_hook(self._hook(i)) for i, l in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

def activation_probabilities(sentences, batch_size=32):
    "p[l, j] = P(neuron j of layer l is active | token)."
    with FFNHooks(model) as hooks:
        for i in range(0, len(sentences), batch_size):
            enc, keep = encode(sentences[i:i + batch_size])
            hooks.keep, hooks.n_tokens = keep.to(DEV), hooks.n_tokens + int(keep.sum())
            model.bert(**enc.to(DEV))
    return (hooks.counts / hooks.n_tokens).cpu().numpy()

def masked_batches(sentences, p=0.15, seed=0, batch_size=32):
    "Masks drawn once (on CPU): the same positions are masked in every condition."
    g = torch.Generator().manual_seed(seed)
    batches = []
    for i in range(0, len(sentences), batch_size):
        enc, keep = encode(sentences[i:i + batch_size])
        m = (torch.rand(keep.shape, generator=g) < p) & keep
        labels = enc["input_ids"].masked_fill(~m, -100)
        enc["input_ids"] = enc["input_ids"].masked_fill(m, tok.mask_token_id)
        batches.append((enc, labels))
    return batches

def mlm_token_losses(batches, ablate=None):
    "Cross-entropy (nats) of every masked position, optionally with ablation."
    out = []
    with FFNHooks(model, ablate=ablate):
        for enc, labels in batches:
            enc, labels = enc.to(DEV), labels.to(DEV)
            h = model.bert(**enc).last_hidden_state
            m = labels != -100
            out.append(F.cross_entropy(model.cls(h[m]), labels[m], reduction="none").cpu())
    return torch.cat(out).numpy()

if os.path.exists(f"{OUT}/P_all.npy"):
    P_all = np.load(f"{OUT}/P_all.npy")
else:
    P_all = np.stack([activation_probabilities(dev[c].text.tolist()) for c in ALL])
    np.save(f"{OUT}/P_all.npy", P_all)
P = P_all[:len(LANGS)]
n_tok = pd.Series({c: dev[c].text.map(lambda s: len(tok.tokenize(s)[:254])).sum() for c in ALL}, name="tokens (dev)")
print(P_all.shape); n_tok

# Q54: sign(GELU(x)) == sign(x)
x = torch.linspace(-20, 20, 400_001)
assert ((F.gelu(x) > 0) == (x > 0)).all()
print("hidden_act =", model.config.hidden_act, "| min GELU =", F.gelu(x).min().item())

fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=True)
for l, ax in enumerate(axes.flat):
    ax.hist(P[:, l].ravel(), bins=50, color="#3b6ea5")
    ax.set_title(f"layer {l}", fontsize=9); ax.set_yscale("log")
fig.supxlabel("activation probability (pooled over the 6 languages)"); fig.supylabel("neurons (log)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig31_act_prob_hist.png")

t31 = pd.DataFrame({
    "never active (all langs)": (P.max(0) == 0).sum(1),
    "p < 0.001 (all langs)": (P.max(0) < 1e-3).sum(1),
    "p > 0.99 (all langs)": (P.min(0) > .99).sum(1),
    "median p": np.median(P, axis=(0, 2)).round(3),
}).rename_axis("layer")
t31.to_csv(f"{OUT}/t31.csv"); t31

# Q55: standard error sqrt(p(1-p)/n), ours vs Tang et al. (1e8 tokens)
se = (pd.DataFrame({"n": pd.concat([n_tok[LANGS], pd.Series({"Tang et al.": 10**8})])})
      .assign(**{f"SE(p={p})": lambda d, p=p: np.sqrt(p * (1 - p) / d.n) for p in (0.5, 0.05, 0.005)}))
se.to_csv(f"{OUT}/t31_se.csv"); se

# Q56-57
configs = {"main": (LANGS, False), "no_deu": ([c for c in LANGS if c != "deu_Latn"], False),
           "tang_code": (LANGS, True), "tang_no_deu": ([c for c in LANGS if c != "deu_Latn"], True), "with_kaz": (ALL, True)}
sel = {}
for name, (langs, prefilter) in configs.items():
    Pc = P_all[[ALL.index(c) for c in langs]]
    tau = np.quantile(Pc, .95)
    with np.errstate(invalid="ignore", divide="ignore"):
        lape = np.where(Pc.max(0) > 0, entropy(Pc, axis=0), np.inf)
    lape = np.where((Pc > tau).any(0) | (not prefilter), lape, np.inf)
    keep = np.zeros(lape.size, bool); keep[np.argsort(lape, axis=None, kind="stable")[:round(.01 * lape.size)]] = True
    keep = keep.reshape(lape.shape)
    sel[name] = dict(langs=langs, P=Pc, tau=tau, lape=lape, keep=keep, assign=(Pc > tau) & keep)

S = sel["main"]; A = S["assign"]
print(f"tau = {S['tau']:.3f} | max LAPE = ln 6 = {np.log(6):.3f} | dead neurons = {(P.max(0) == 0).sum()} | kept = {S['keep'].sum()}")
t32 = pd.DataFrame({cfg: pd.Series(sel[cfg]["assign"].sum((1, 2)), index=sel[cfg]["langs"]) for cfg in sel}).fillna(0).astype(int)
t32.loc["several languages"] = [(sel[c]["assign"].sum(0) > 1).sum() for c in sel]
t32.loc["kept but unassigned"] = [(sel[c]["keep"] & ~sel[c]["assign"].any(0)).sum() for c in sel]
t32.to_csv(f"{OUT}/t32_counts.csv"); t32

# Q58: selected neurons per layer
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
per_layer = {}
for ax, s_ in zip(axes, ["main", "tang_code"]):
    per_layer[s_] = pd.DataFrame(sel[s_]["assign"].sum(2).T, columns=LANGS).rename_axis("layer")
    per_layer[s_].plot(ax=ax, marker="o", title=f"LAPE neurons per layer ({s_})", ylabel="neurons assigned")
    ax.set_xticks(range(12))
fig.tight_layout(); fig.savefig(f"{OUT}/fig32_per_layer.png")
pd.concat(per_layer, names=["selection"]).to_csv(f"{OUT}/t32_per_layer.csv")
display(pd.concat(per_layer, axis=1))

# Q64: neurons shared by two languages
shared = pd.DataFrame(np.einsum("aln,bln->ab", A.astype(int), A.astype(int)), index=LANGS, columns=LANGS)
shared.to_csv(f"{OUT}/t32_shared.csv"); shared

def tok_stats(sents):
    enc = tok(sents, add_special_tokens=False)
    s = pd.Series([t for ids in enc["input_ids"] for t in tok.convert_ids_to_tokens(ids)])
    n_words = sum(len({w for w in enc.word_ids(i) if w is not None}) for i in range(len(sents)))
    return {"pieces": len(s), "fertility": len(s) / n_words, "## share": s.str.startswith("##").mean(),
            "single-char share": s.str.removeprefix("##").str.len().eq(1).mean(), "[UNK] rate": (s == tok.unk_token).mean()}

tokdiag = (pd.DataFrame({c: tok_stats(dev[c].text.tolist()) for c in ALL}).T
           .assign(**{"premium vs eng": lambda d: d.pieces / d.pieces["eng_Latn"],
                      "neurons (main)": t32["main"], "neurons (tang_code)": t32["tang_code"], "neurons (with_kaz)": t32["with_kaz"]})
           .round(3))
tokdiag.to_csv(f"{OUT}/t64_tokenisation.csv"); tokdiag

# Q59 / Q65
jac = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
B = sel["no_deu"]["assign"]; T = sel["tang_code"]["assign"]
t59 = pd.DataFrame({
    "J(main, no_deu)": {c: jac(A[LANGS.index(c)], B[sel["no_deu"]["langs"].index(c)]) for c in sel["no_deu"]["langs"]},
    "J(main, tang_code)": {c: jac(A[i], T[i]) for i, c in enumerate(LANGS)},
    "J(tang_code, tang_no_deu)": {c: jac(T[i], sel["tang_no_deu"]["assign"][sel["tang_no_deu"]["langs"].index(c)])
                                  for i, c in enumerate(LANGS) if c != "deu_Latn"},
}).round(3)
t59.to_csv(f"{OUT}/t59_jaccard.csv"); t59

# Q62: kept neurons assigned to no language (shown as activation counts)
lost = np.argwhere(S["keep"] & ~A.any(0))
q62 = (pd.DataFrame((P[:, lost[:, 0], lost[:, 1]].T * n_tok[LANGS].values).round().astype(int), columns=LANGS)
       .assign(layer=lost[:, 0], neuron=lost[:, 1], LAPE=S["lape"][lost[:, 0], lost[:, 1]].round(3)))
print(f"{len(lost)} of {S['keep'].sum()} kept neurons are assigned to no language; their max activation count:",
      q62[LANGS].max(1).describe()[["min", "50%", "max"]].to_dict())
q62 = q62.sort_values(["LAPE", "layer"]).head(8); q62

# Q66: baseline loss on the first 300 devtest sentences, masks drawn once
N_EVAL = 300
batches = {c: masked_batches(devtest[c].text[:N_EVAL].tolist(), seed=SEED) for c in ALL}
base = {c: mlm_token_losses(batches[c]) for c in ALL}
base_tab = pd.DataFrame({"masked tokens": {c: len(base[c]) for c in ALL}, "baseline loss (nats)": {c: base[c].mean() for c in ALL}}).round(3)
base_tab.to_csv(f"{OUT}/t33_baseline.csv"); base_tab

# Q67-68
ABL = ["main", "tang_code"]
rng = np.random.default_rng(SEED)
sets = {}
for sname in ABL:
    As = sel[sname]["assign"]; pool = ~As.any(0)
    sets |= {(sname, "LAPE", k, 0): {l: np.flatnonzero(As[i, l]) for l in range(12)} for i, k in enumerate(LANGS)}
    sets |= {(sname, "random", k, r): {l: rng.choice(np.flatnonzero(pool[l]), As[i, l].sum(), replace=False) for l in range(12)}
             for r in range(3) for i, k in enumerate(LANGS)}
assert all(len(sets[(s_, "random", k, r)][l]) == len(sets[(s_, "LAPE", k, 0)][l]) for s_ in ABL for k in LANGS for r in range(3) for l in range(12))

if os.path.exists(f"{OUT}/t33_ablation.csv"):
    res = pd.read_csv(f"{OUT}/t33_ablation.csv")
else:
    rows = []
    for (sname, kind, k, r), ab in sets.items():
        for c in LANGS:
            d = mlm_token_losses(batches[c], ablate=ab) - base[c]
            rows.append(dict(selection=sname, kind=kind, ablated=k, draw=r, evaluated=c, delta=d.mean(),
                             se=d.std(ddof=1) / np.sqrt(len(d)), rel=d.mean() / base[c].mean(), n_neurons=sum(map(len, ab.values()))))
        print(sname, kind, k, r, "done")
    res = pd.DataFrame(rows); res.to_csv(f"{OUT}/t33_ablation.csv", index=False)

lape_res, rand_res = res.query("kind == 'LAPE'"), res.query("kind == 'random'")
M = {s_: lape_res.query("selection == @s_").pivot(index="ablated", columns="evaluated", values="delta").loc[LANGS, LANGS] for s_ in ABL}
M_se = {s_: lape_res.query("selection == @s_").pivot(index="ablated", columns="evaluated", values="se").loc[LANGS, LANGS] for s_ in ABL}
M_rel = {s_: lape_res.query("selection == @s_").pivot(index="ablated", columns="evaluated", values="rel").loc[LANGS, LANGS] for s_ in ABL}
R = {s_: rand_res.query("selection == @s_").groupby(["ablated", "evaluated"]).delta.mean().unstack().loc[LANGS, LANGS] for s_ in ABL}
R_sd = {s_: rand_res.query("selection == @s_").groupby(["ablated", "evaluated"]).delta.std().unstack().loc[LANGS, LANGS] for s_ in ABL}
pd.concat(M, names=["selection"]).round(3)

fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
vmax = max(max(t.values.max() for t in M.values()), max(t.values.max() for t in R.values()), 1e-3)
for row, s_ in zip(axes, ABL):
    for ax, (title, t) in zip(row, {f"LAPE neurons zeroed ({s_})": M[s_], f"random control ({s_}, mean of 3)": R[s_]}.items()):
        im = ax.imshow(t.values, cmap="Reds", vmin=0, vmax=vmax)
        ax.set_xticks(range(6), [c[:3] for c in LANGS]); ax.set_yticks(range(6), [c[:3] for c in LANGS])
        ax.set_xlabel("evaluated language"); ax.set_ylabel("deactivated language"); ax.set_title(title, fontsize=10); ax.grid(False)
        [ax.text(j, i, f"{t.values[i, j]:.2f}", ha="center", va="center", fontsize=8) for i in range(6) for j in range(6)]
fig.colorbar(im, ax=axes, label="Δ MLM loss (nats)"); fig.savefig(f"{OUT}/fig33_ablation.png", bbox_inches="tight")

# diagonal vs mean off-diagonal
spec = pd.concat({s_: pd.DataFrame({
    "n neurons": lape_res.query("selection == @s_").groupby("ablated").n_neurons.first().loc[LANGS],
    "diag (LAPE)": np.diag(M[s_]), "off-diag mean (LAPE)": (M[s_].values.sum(1) - np.diag(M[s_])) / 5,
    "diag (random)": np.diag(R[s_]), "off-diag mean (random)": (R[s_].values.sum(1) - np.diag(R[s_])) / 5,
}, index=LANGS) for s_ in ABL}, names=["selection"]).round(3)
spec.to_csv(f"{OUT}/t33_specificity.csv"); spec

fmt = lambda m, e: m.round(3).astype(str) + " ± " + e.round(3).astype(str)
display(pd.concat({s_: fmt(M[s_], M_se[s_]) for s_ in ABL}, names=["LAPE, selection"]))
display(pd.concat({s_: fmt(R[s_], R_sd[s_]) for s_ in ABL}, names=["random, selection"]))
Z = pd.concat({s_: (M[s_] / M_se[s_]).round(1) for s_ in ABL}, names=["z, selection"]); display(Z)
n_abl = lape_res.groupby(["selection", "ablated"]).n_neurons.first()
diag_tab = pd.concat({s_: pd.DataFrame({
    "n neurons": n_abl[s_].loc[LANGS], "Δ diag": np.diag(M[s_]), "z diag": np.diag(M[s_] / M_se[s_]),
    "ppl +%": 100 * (np.exp(np.diag(M[s_])) - 1), "Δ per neuron (×1000)": 1000 * np.diag(M[s_]) / n_abl[s_].loc[LANGS].values,
    "diag is max of row": [M[s_].loc[k].idxmax() == k for k in LANGS], "diag is max of column": [M[s_][k].idxmax() == k for k in LANGS],
    "|off-diag| max": [M[s_].loc[k].drop(k).abs().max() for k in LANGS], "z of that": [(M[s_] / M_se[s_]).loc[k, M[s_].loc[k].drop(k).abs().idxmax()] for k in LANGS],
}, index=LANGS) for s_ in ABL}, names=["selection"]).round(3)
diag_tab.to_csv(f"{OUT}/t33_diag.csv"); diag_tab

if os.path.exists(f"{OUT}/t33_active_control.csv"):
    act = pd.read_csv(f"{OUT}/t33_active_control.csv")
else:
    rng = np.random.default_rng(SEED + 1); rows = []
    for s_ in ABL:
        As = sel[s_]["assign"]
        for i, k in enumerate(LANGS):
            for r in range(3):
                ab = {}
                for l in range(12):
                    cand = np.flatnonzero(~As.any(0)[l] & (P[i, l] > S["tau"]))
                    ab[l] = rng.choice(cand, min(As[i, l].sum(), len(cand)), replace=False)
                short = int(As[i].sum() - sum(map(len, ab.values())))
                for c in LANGS:
                    d = mlm_token_losses(batches[c], ablate=ab) - base[c]
                    rows.append(dict(selection=s_, ablated=k, draw=r, evaluated=c, delta=d.mean(), n_neurons=sum(map(len, ab.values())), shortfall=short))
            print(s_, k, "done")
    act = pd.DataFrame(rows); act.to_csv(f"{OUT}/t33_active_control.csv", index=False)

C = {s_: act.query("selection == @s_").groupby(["ablated", "evaluated"]).delta.mean().unstack().loc[LANGS, LANGS] for s_ in ABL}
C_sd = {s_: act.query("selection == @s_").groupby(["ablated", "evaluated"]).delta.std().unstack().loc[LANGS, LANGS] for s_ in ABL}
display(pd.concat({s_: fmt(C[s_], C_sd[s_]) for s_ in ABL}, names=["active control, selection"]))
print("neurons missing because the pool was too small:", act.groupby(["selection", "ablated"]).shortfall.first().to_dict())
cmp_tab = pd.concat({s_: pd.DataFrame({
    "diag LAPE": np.diag(M[s_]), "diag active ctrl": np.diag(C[s_]), "diag random": np.diag(R[s_]),
    "off-diag mean LAPE": (M[s_].values.sum(1) - np.diag(M[s_])) / 5, "off-diag mean active ctrl": (C[s_].values.sum(1) - np.diag(C[s_])) / 5,
    "diag − off LAPE": np.diag(M[s_]) - (M[s_].values.sum(1) - np.diag(M[s_])) / 5,
    "diag − off active ctrl": np.diag(C[s_]) - (C[s_].values.sum(1) - np.diag(C[s_])) / 5,
}, index=LANGS) for s_ in ABL}, names=["selection"]).round(3)
cmp_tab.to_csv(f"{OUT}/t33_active_compare.csv"); cmp_tab

K = sel["with_kaz"]; AK = K["assign"]
shared_k = pd.DataFrame(np.einsum("aln,bln->ab", AK.astype(int), AK.astype(int)), index=ALL, columns=ALL)
shared_k.to_csv(f"{OUT}/bonus_shared.csv")

if os.path.exists(f"{OUT}/bonus_ablation.csv"):
    bonus = pd.read_csv(f"{OUT}/bonus_ablation.csv", index_col=0)
else:
    bonus = pd.DataFrame({k: {c: (mlm_token_losses(batches[c], {l: np.flatnonzero(AK[ALL.index(k), l]) for l in range(12)}) - base[c]).mean()
                              for c in ALL} for k in ["kaz_Cyrl", "rus_Cyrl"]}).T
    bonus.to_csv(f"{OUT}/bonus_ablation.csv")
display(shared_k.loc[["rus_Cyrl", "kaz_Cyrl"]]); bonus.round(3)

TT = sel["tang_code"]
picks = [(l, j, f"{k[:3]} #{n + 1}") for i, k in enumerate(LANGS)
         for n, (l, j) in enumerate(sorted(zip(*np.nonzero(TT["assign"][i])), key=lambda lj: -P[i][lj])[:3])]
for a, b in [("kaz_Cyrl", "yor_Latn"), ("kaz_Cyrl", "rus_Cyrl")]:
    both = AK[ALL.index(a)] & AK[ALL.index(b)]
    picks += [(l, j, f"{a[:3]}+{b[:3]}") for l, j in list(zip(*np.nonzero(both)))[:4]]
picks = pd.DataFrame(picks, columns=["layer", "neuron", "label"]).groupby(["layer", "neuron"]).label.agg(", ".join).reset_index()

if os.path.exists(f"{OUT}/t73_top_tokens.csv"):
    top_tab = pd.read_csv(f"{OUT}/t73_top_tokens.csv")
else:
    by_layer = picks.groupby("layer").neuron.apply(list).to_dict()
    store, state, toks, langs = {l: [] for l in by_layer}, {}, [], []
    rec = lambda l: (lambda m, i, o: store[l].append(o[..., by_layer[l]][state["keep"]].float().cpu()))
    handles = [model.bert.encoder.layer[l].intermediate.register_forward_hook(rec(l)) for l in by_layer]
    for c in ALL:
        for i in range(0, len(dev[c]), 32):
            enc, keep = encode(dev[c].text[i:i + 32].tolist())
            state["keep"] = keep.to(DEV); model.bert(**enc.to(DEV))
            toks += tok.convert_ids_to_tokens(enc["input_ids"].cpu()[keep].tolist()); langs += [c[:3]] * int(keep.sum())
    [h.remove() for h in handles]
    acts = {l: torch.cat(store[l]).numpy() for l in store}
    tokens, lang_s = pd.Series(toks), pd.Series(langs)
    rows = []
    for _, r in picks.iterrows():
        top = pd.DataFrame({"tok": tokens, "lang": lang_s, "a": acts[r.layer][:, by_layer[r.layer].index(r.neuron)]}).nlargest(200, "a")
        rows.append({"label": r.label, "layer": r.layer, "neuron": r.neuron,
                     "top tokens": " ".join(top.drop_duplicates("tok").tok.head(12)),
                     "langs (top-200)": ", ".join(f"{k}:{v}" for k, v in top.lang.value_counts().head(3).items()),
                     "## share": top.tok.str.startswith("##").mean().round(2),
                     "1-char share": top.tok.str.removeprefix("##").str.len().eq(1).mean().round(2)})
    top_tab = pd.DataFrame(rows); top_tab.to_csv(f"{OUT}/t73_top_tokens.csv", index=False)
top_tab

if os.path.exists(f"{OUT}/lang_means.npy"):
    means = np.load(f"{OUT}/lang_means.npy")
else:
    means = np.stack([sentence_reps(dev[c].text.tolist()).mean(1) for c in LANGS])   # (6, 13, 768)
    np.save(f"{OUT}/lang_means.npy", means)

W = torch.stack([l.output.dense.weight for l in model.bert.encoder.layer]).cpu().numpy()      # (12, 768, 3072)
G = torch.stack([l.output.LayerNorm.weight for l in model.bert.encoder.layer]).cpu().numpy()  # (12, 768)
Wt = G[:, :, None] * (W - W.mean(1, keepdims=True))
Mc = means[:, 1:] - means[:, 1:].mean(0)                                                      # (6, 12, 768), centred
Q = np.linalg.svd(Mc.transpose(1, 2, 0), full_matrices=False)[0][:, :, :len(LANGS) - 1]      # (12, 768, 5)
r_raw = (np.einsum("ldk,ldn->lkn", Q, W) ** 2).sum(1) / (W ** 2).sum(1)                      # (12, 3072)
r_ln = (np.einsum("ldk,ldn->lkn", Q, Wt) ** 2).sum(1) / (Wt ** 2).sum(1)

# Q74: random directions, E[r] = k/d
u = np.random.default_rng(SEED).standard_normal((768, 20_000))
print("k/d =", 5 / 768, "| random directions:", ((np.einsum("dk,dn->kn", Q[6], u) ** 2).sum(0) / (u ** 2).sum(0)).mean())

rand_mask = {s_: np.zeros_like(A[0]) for s_ in ABL}
for (s_, kind, k, r), ab in sets.items():
    for l, idx in ab.items():
        rand_mask[s_][l, idx] |= kind == "random"
t34 = pd.concat({s_: pd.DataFrame({
    **{f"LAPE {c[:3]}": (r_raw[As[i]].mean(), r_ln[As[i]].mean()) for i, c in enumerate(LANGS) if As[i].any()},
    "LAPE (all)": (r_raw[As.any(0)].mean(), r_ln[As.any(0)].mean()),
    "random (same layers)": (r_raw[rand_mask[s_]].mean(), r_ln[rand_mask[s_]].mean()),
    "all neurons": (r_raw.mean(), r_ln.mean()), "k/d": (5 / 768, 5 / 768),
}, index=["r (raw w)", "r (LN-corrected w)"]).T for s_, As in ((s_, sel[s_]["assign"]) for s_ in ABL)}, names=["selection"]).round(4)
t34.to_csv(f"{OUT}/t34.csv"); display(t34)

# per layer (tang_code)
At = sel["tang_code"]["assign"].any(0)
t34_layer = pd.DataFrame({"n LAPE": At.sum(1), "r LAPE": [r_raw[l][At[l]].mean() if At[l].any() else np.nan for l in range(12)],
                          "r all neurons": r_raw.mean(1), "k/d": 5 / 768}).rename_axis("layer").round(4)
t34_layer.to_csv(f"{OUT}/t34_layer.csv"); t34_layer
