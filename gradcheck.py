"""
Numerical gradient check for tinygpt.py

Checks that the hand-derived backward formulas in tinygpt.py are correct
by comparing them against a central-difference numerical gradient.

    numerical:  dL/dθ_i ≈ [ L(θ + h e_i) - L(θ - h e_i) ] / (2h)

    rel_error = ||G_analytic - G_numerical|| / (||G_analytic|| + ||G_numerical||)

Checks:

    1. GELU            gelu_grad
    2. LayerNorm       layernorm_backward        (R, gamma, beta)
    3. Attention head  one_head_backward         (X, WQ, WK, WV)
    4. Block           block_backward            (X + block parameters)
    5. Full model      loss_and_backward         (all parameters, including weight tying)

Component checks (1-4) build a scalar loss with a random constant matrix C.

    L = sum(out ⊙ C)   ->   dL/d(out) = C

Results are printed to the console, and an HTML report with charts
(gradcheck_report.html) is written and opened in the browser.

Usage:

    python gradcheck.py
    python gradcheck.py --init original     # keep tinygpt's default initialization
    python gradcheck.py --h 1e-6 --no-open
"""

import argparse
import html
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinygpt import TinyGPT


# rel_error thresholds
#
#   < 1e-6          PASS  (float64 central differences usually give 1e-8 ~ 1e-10)
#   1e-6 ~ 1e-4     WARN  (could be numerical noise or a subtle bug)
#   >= 1e-4         FAIL  (the formula is most likely wrong)

PASS_TOL = 1e-6
FAIL_TOL = 1e-4


# ====================================================================
# Numerical gradient
# ====================================================================

def numerical_grad(f, x, h, indices=None):

    # f : no-argument function returning the scalar loss at the current x
    # x : array perturbed in place (if it is in model.p, the model sees it directly)

    g = np.zeros_like(x)

    if indices is None:
        indices = np.ndindex(x.shape)

    for i in indices:

        old = x[i]

        x[i] = old + h
        f_plus = f()

        x[i] = old - h
        f_minus = f()

        x[i] = old

        g[i] = (f_plus - f_minus) / (2.0 * h)

    return g


def rel_error(a, n):

    return float(
        np.linalg.norm(a - n)
        /
        max(np.linalg.norm(a) + np.linalg.norm(n), 1e-30)
    )


def status_of(rel):

    if rel < PASS_TOL:
        return "PASS"

    if rel < FAIL_TOL:
        return "WARN"

    return "FAIL"


@dataclass
class Result:

    group: str
    name: str
    analytic: np.ndarray = field(repr=False)
    numeric: np.ndarray = field(repr=False)

    @property
    def rel(self):
        return rel_error(self.analytic, self.numeric)

    @property
    def max_abs(self):
        return float(np.max(np.abs(self.analytic - self.numeric)))

    @property
    def norm(self):
        return float(np.linalg.norm(self.analytic))

    @property
    def status(self):
        return status_of(self.rel)


# ====================================================================
# Test model
# ====================================================================

def make_model(init, seed):

    model = TinyGPT(
        vocab_size=11,
        max_context=6,
        d_model=8,
        n_heads=2,
        d_ff=12,
        n_layers=2,
        seed=seed
    )

    rng = np.random.default_rng(seed + 1)

    # ----------------------------------------------------------------
    # With gamma = 1, beta = 0, b = 0, bugs like "forgot to multiply by gamma"
    # stay hidden, so move them away from their defaults on purpose.
    # ----------------------------------------------------------------

    for k in model.p:
        if k.startswith(("gamma", "beta", "b1", "b2")):
            model.p[k] = model.p[k] + rng.normal(0.0, 0.3, model.p[k].shape)

    # ----------------------------------------------------------------
    # With the default init (E, P std 0.02) the layer-0 WQ/WK gradients are
    # nearly zero, so central-difference round-off noise dominates and WARNs appear.
    # "scaled" raises E, P to std 1 so only the formulas themselves are tested.
    # ----------------------------------------------------------------

    if init == "scaled":
        model.p["E"] = rng.normal(0.0, 1.0, model.p["E"].shape)
        model.p["P"] = rng.normal(0.0, 1.0, model.p["P"].shape)

    return model


# ====================================================================
# 1~4. Component checks
# ====================================================================

def check_gelu(model, rng, h):

    x = rng.normal(0.0, 2.0, size=(5, 7))
    C = rng.normal(size=x.shape)

    analytic = model.gelu_grad(x) * C

    numeric = numerical_grad(
        lambda: np.sum(model.gelu(x) * C), x, h
    )

    return [Result("GELU", "x", analytic, numeric)]


def check_layernorm(model, rng, h):

    T, d = 5, model.d

    R = rng.normal(0.0, 1.5, size=(T, d))
    gamma = rng.normal(1.0, 0.5, size=d)
    beta = rng.normal(0.0, 0.5, size=d)
    C = rng.normal(size=(T, d))

    def loss():
        Y, _ = model.layernorm_forward(R, gamma, beta)
        return np.sum(Y * C)

    _, cache = model.layernorm_forward(R, gamma, beta)
    GR, Ggamma, Gbeta = model.layernorm_backward(C, cache)

    return [
        Result("LayerNorm", "R", GR, numerical_grad(loss, R, h)),
        Result("LayerNorm", "gamma", Ggamma, numerical_grad(loss, gamma, h)),
        Result("LayerNorm", "beta", Gbeta, numerical_grad(loss, beta, h)),
    ]


def check_attention_head(model, rng, h):

    T, d, dh = 5, model.d, model.dh

    X = rng.normal(size=(T, d))
    WQ = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, dh))
    WK = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, dh))
    WV = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, dh))
    C = rng.normal(size=(T, dh))

    def loss():
        O, _ = model.one_head_forward(X, WQ, WK, WV)
        return np.sum(O * C)

    _, cache = model.one_head_forward(X, WQ, WK, WV)
    GX, GWQ, GWK, GWV = model.one_head_backward(C, cache, WQ, WK, WV)

    return [
        Result("Attention head", "X", GX, numerical_grad(loss, X, h)),
        Result("Attention head", "WQ", GWQ, numerical_grad(loss, WQ, h)),
        Result("Attention head", "WK", GWK, numerical_grad(loss, WK, h)),
        Result("Attention head", "WV", GWV, numerical_grad(loss, WV, h)),
    ]


def check_block(model, rng, h, n=0):

    T, d = 5, model.d

    X = rng.normal(size=(T, d))
    C = rng.normal(size=(T, d))

    def loss():
        X_next, _ = model.block_forward(X, n)
        return np.sum(X_next * C)

    _, cache = model.block_forward(X, n)
    GX, grads = model.block_backward(C, cache, n)

    results = [
        Result("Block", "X", GX, numerical_grad(loss, X, h))
    ]

    # parameters returned by block_backward are checked by perturbing model.p directly
    for k in grads:
        results.append(
            Result("Block", k, grads[k], numerical_grad(loss, model.p[k], h))
        )

    return results


# ====================================================================
# 5. Full model check
# ====================================================================

# token 3 appears twice so the np.add.at accumulation is tested too.
INPUT_IDS = [0, 3, 5, 3, 7]
TARGET_IDS = [3, 5, 3, 7, 1]


def full_loss(model):

    loss, _ = model.loss_and_backward(INPUT_IDS, TARGET_IDS)

    return loss


def check_full_model(model, h):

    _, grads = model.loss_and_backward(INPUT_IDS, TARGET_IDS)

    results = []

    for k, P in model.p.items():
        results.append(
            Result(
                "Full model",
                k,
                grads[k],
                numerical_grad(lambda: full_loss(model), P, h)
            )
        )

    return results


# ====================================================================
# h sweep
#
# If h is too large, truncation error (O(h^2)) grows;
# if too small, floating-point round-off error (O(eps/h)) grows.
# With correct formulas the error forms a V whose bottom goes below 1e-8.
# With wrong formulas the floor stays high no matter what h is.
# ====================================================================

SWEEP_PARAMS = ["E", "WQ_0_0", "gamma1_0", "W1_1"]
SWEEP_HS = 10.0 ** np.arange(-1.0, -10.01, -0.5)


def h_sweep(model, rng, n_elements=16):

    _, grads = model.loss_and_backward(INPUT_IDS, TARGET_IDS)

    curves = {}

    for k in SWEEP_PARAMS:

        P = model.p[k]

        flat = rng.choice(P.size, size=min(n_elements, P.size), replace=False)
        indices = [np.unravel_index(i, P.shape) for i in flat]

        analytic = np.array([grads[k][i] for i in indices])

        errors = []

        for h in SWEEP_HS:
            g = numerical_grad(lambda: full_loss(model), P, h, indices)
            numeric = np.array([g[i] for i in indices])
            errors.append(rel_error(analytic, numeric))

        curves[k] = errors

    return curves


# ====================================================================
# HTML report (inline SVG, no external libraries)
# ====================================================================

def esc(s):
    return html.escape(str(s))


def fmt(v):
    return f"{v:.2e}"


def log_ticks(lo, hi, step):
    return list(range(lo, hi + 1, step))


def svg_dot_plot(results, h):

    # One dot per row: its rel_error on a log axis.

    left, right, top, row = 230, 24, 44, 20
    W = 820
    H = top + row * len(results) + 34
    xmin, xmax = -14, 0

    def X(v):
        lv = np.log10(max(v, 10.0 ** xmin))
        return left + (lv - xmin) / (xmax - xmin) * (W - left - right)

    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="relative error per check">'
    ]

    # verdict band backgrounds
    out.append(
        f'<rect x="{X(PASS_TOL):.1f}" y="{top - 8}" '
        f'width="{X(FAIL_TOL) - X(PASS_TOL):.1f}" '
        f'height="{row * len(results) + 8}" class="band-warn"/>'
    )
    out.append(
        f'<rect x="{X(FAIL_TOL):.1f}" y="{top - 8}" '
        f'width="{X(10.0 ** xmax) - X(FAIL_TOL):.1f}" '
        f'height="{row * len(results) + 8}" class="band-fail"/>'
    )

    # grid + x ticks
    for t in log_ticks(xmin, xmax, 2):
        x = X(10.0 ** t)
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 8}" '
            f'y2="{H - 30}" class="grid"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{H - 12}" class="tick" '
            f'text-anchor="middle">1e{t}</text>'
        )

    # threshold lines
    for tol, label in ((PASS_TOL, "PASS limit 1e-6"), (FAIL_TOL, "FAIL limit 1e-4")):
        x = X(tol)
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 14}" '
            f'y2="{H - 30}" class="threshold"/>'
        )
        out.append(
            f'<text x="{x + 4:.1f}" y="{top - 18}" class="tick">{label}</text>'
        )

    prev_group = None

    for r_i, r in enumerate(results):

        y = top + row * r_i + row / 2

        if prev_group is not None and r.group != prev_group:
            out.append(
                f'<line x1="8" x2="{W - right}" y1="{y - row / 2:.1f}" '
                f'y2="{y - row / 2:.1f}" class="sep"/>'
            )

        prev_group = r.group

        out.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" class="label" '
            f'text-anchor="end"><tspan class="muted">{esc(r.group)} · </tspan>'
            f'{esc(r.name)}</text>'
        )

        out.append(
            f'<line x1="{left}" x2="{X(r.rel):.1f}" y1="{y:.1f}" '
            f'y2="{y:.1f}" class="stem"/>'
        )

        out.append(
            f'<g class="pt"><circle cx="{X(r.rel):.1f}" cy="{y:.1f}" r="12" '
            f'class="hit"/><circle cx="{X(r.rel):.1f}" cy="{y:.1f}" r="5" '
            f'class="dot {r.status.lower()}"/>'
            f'<title>{esc(r.group)} · {esc(r.name)} {r.analytic.shape}\n'
            f'rel error  {fmt(r.rel)}\nmax |diff| {fmt(r.max_abs)}\n'
            f'||grad||   {fmt(r.norm)}\n{r.status}</title></g>'
        )

    out.append("</svg>")

    return "\n".join(out)


def svg_scatter(results):

    # For every element: x = numerical, y = analytic.
    # With correct formulas every point lies on the y = x diagonal.

    W = H = 460
    pad_l, pad_b, pad_t, pad_r = 64, 50, 16, 16

    a = np.concatenate([r.analytic.ravel() for r in results])
    n = np.concatenate([r.numeric.ravel() for r in results])

    m = float(np.max(np.abs(np.concatenate([a, n])))) * 1.05

    def X(v):
        return pad_l + (v + m) / (2 * m) * (W - pad_l - pad_r)

    def Y(v):
        return H - pad_b - (v + m) / (2 * m) * (H - pad_b - pad_t)

    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="analytic vs numerical gradient scatter plot">'
    ]

    for t in np.linspace(-m, m, 5):
        out.append(
            f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pad_t}" '
            f'y2="{H - pad_b}" class="grid"/>'
        )
        out.append(
            f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{Y(t):.1f}" '
            f'y2="{Y(t):.1f}" class="grid"/>'
        )
        out.append(
            f'<text x="{X(t):.1f}" y="{H - pad_b + 18}" class="tick" '
            f'text-anchor="middle">{t:.2g}</text>'
        )
        out.append(
            f'<text x="{pad_l - 8}" y="{Y(t) + 4:.1f}" class="tick" '
            f'text-anchor="end">{t:.2g}</text>'
        )

    out.append(
        f'<line x1="{X(-m):.1f}" y1="{Y(-m):.1f}" x2="{X(m):.1f}" '
        f'y2="{Y(m):.1f}" class="diag"/>'
    )
    out.append(
        f'<text x="{X(m) - 6:.1f}" y="{Y(m) + 16:.1f}" class="tick" '
        f'text-anchor="end">y = x</text>'
    )

    for r in results:

        for idx in np.ndindex(r.analytic.shape):

            av = r.analytic[idx]
            nv = r.numeric[idx]
            bad = status_of(rel_error(np.array([av]), np.array([nv]))) == "FAIL" \
                and abs(av - nv) > 1e-6

            out.append(
                f'<circle cx="{X(nv):.1f}" cy="{Y(av):.1f}" r="3.5" '
                f'class="sc{" fail" if bad else ""}">'
                f'<title>{esc(r.name)}{list(idx)}\nanalytic  {av:.6e}\n'
                f'numerical {nv:.6e}</title></circle>'
            )

    out.append(
        f'<text x="{(pad_l + W - pad_r) / 2}" y="{H - 8}" class="axis-label" '
        f'text-anchor="middle">numerical gradient</text>'
    )
    out.append(
        f'<text x="14" y="{(pad_t + H - pad_b) / 2}" class="axis-label" '
        f'text-anchor="middle" transform="rotate(-90 14 '
        f'{(pad_t + H - pad_b) / 2})">analytic gradient</text>'
    )

    out.append("</svg>")

    return "\n".join(out)


def svg_h_sweep(curves, h_used):

    W, H = 820, 380
    pad_l, pad_r, pad_t, pad_b = 64, 96, 20, 48

    xs = np.log10(SWEEP_HS)
    xlo, xhi = xs.min(), xs.max()
    ylo, yhi = -14, 0

    def X(lh):
        return pad_l + (lh - xlo) / (xhi - xlo) * (W - pad_l - pad_r)

    def Y(v):
        lv = np.clip(np.log10(max(v, 1e-30)), ylo, yhi)
        return pad_t + (yhi - lv) / (yhi - ylo) * (H - pad_t - pad_b)

    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="relative error vs h">'
    ]

    for t in log_ticks(ylo, yhi, 2):
        y = Y(10.0 ** t)
        out.append(
            f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'class="grid"/>'
        )
        out.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="tick" '
            f'text-anchor="end">1e{t}</text>'
        )

    for t in range(int(xlo), int(xhi) + 1):
        out.append(
            f'<text x="{X(t):.1f}" y="{H - pad_b + 18}" class="tick" '
            f'text-anchor="middle">1e{t}</text>'
        )

    # PASS threshold line and the h in use
    y = Y(PASS_TOL)
    out.append(
        f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
        f'class="threshold"/>'
    )
    out.append(
        f'<text x="{W - pad_r + 6}" y="{y + 4:.1f}" class="tick">PASS limit</text>'
    )

    x = X(np.log10(h_used))
    out.append(
        f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{H - pad_b}" '
        f'class="threshold"/>'
    )
    out.append(
        f'<text x="{x + 4:.1f}" y="{pad_t + 12}" class="tick">h used</text>'
    )

    # spread the end-label y positions so direct labels do not overlap
    ends = sorted(
        ((Y(errs[0]), k) for k, errs in curves.items()),
        key=lambda e: e[0]
    )
    label_y = {}
    last = -1e9
    for y0, k in ends:
        y1 = max(y0, last + 14)
        label_y[k] = y1
        last = y1

    for s_i, (k, errs) in enumerate(curves.items()):

        cls = f"s{s_i + 1}"
        pts = " ".join(f"{X(lx):.1f},{Y(e):.1f}" for lx, e in zip(xs, errs))

        out.append(f'<polyline points="{pts}" class="line {cls}"/>')

        for lx, e in zip(xs, errs):
            out.append(
                f'<g class="pt"><circle cx="{X(lx):.1f}" cy="{Y(e):.1f}" r="10" '
                f'class="hit"/><circle cx="{X(lx):.1f}" cy="{Y(e):.1f}" r="4" '
                f'class="mk {cls}"/><title>{esc(k)}\nh = {10 ** lx:.0e}\n'
                f'rel error = {fmt(e)}</title></g>'
            )

        out.append(
            f'<text x="{W - pad_r + 6}" y="{label_y[k] + 4:.1f}" '
            f'class="label">{esc(k)}</text>'
        )

    out.append(
        f'<text x="{(pad_l + W - pad_r) / 2}" y="{H - 8}" class="axis-label" '
        f'text-anchor="middle">h (step size) — smaller to the right</text>'
    )

    out.append("</svg>")

    return "\n".join(out)


STATUS_ICON = {"PASS": "✓", "WARN": "!", "FAIL": "✕"}


def results_table(results):

    rows = []

    for r in results:
        rows.append(
            f"<tr><td>{esc(r.group)}</td><td><code>{esc(r.name)}</code></td>"
            f"<td>{esc(r.analytic.shape)}</td><td class='num'>{fmt(r.norm)}</td>"
            f"<td class='num'>{fmt(r.rel)}</td><td class='num'>{fmt(r.max_abs)}</td>"
            f"<td><span class='badge {r.status.lower()}'>"
            f"{STATUS_ICON[r.status]} {r.status}</span></td></tr>"
        )

    return (
        "<table><thead><tr><th>Group</th><th>Target</th><th>shape</th>"
        "<th class='num'>||analytic||</th><th class='num'>rel error</th>"
        "<th class='num'>max |diff|</th><th>Verdict</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --good: #0ca30c; --good-ink: #006300; --warning: #fab219; --critical: #d03b3b;
  --band-warn: rgba(250,178,25,0.10); --band-fail: rgba(208,59,59,0.08);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --good-ink: #0ca30c;
    --band-warn: rgba(250,178,25,0.10); --band-fail: rgba(208,59,59,0.14);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --good-ink: #0ca30c;
  --band-warn: rgba(250,178,25,0.10); --band-fail: rgba(208,59,59,0.14);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.55 system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
}
main { max-width: 900px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 4px; }
.sub, .desc { color: var(--ink-2); margin: 0 0 16px; }
.desc { font-size: 13px; }
section {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; margin-top: 20px; overflow-x: auto;
}
svg { width: 100%; height: auto; display: block; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.tile .k { color: var(--ink-2); font-size: 12px; }
.tile .v { font-size: 26px; font-weight: 600; margin-top: 2px; }
.verdict { font-size: 15px; font-weight: 600; margin-top: 16px; }
.verdict.pass { color: var(--good-ink); }
.verdict.fail { color: var(--critical); }
code { font-family: ui-monospace, Consolas, monospace; font-size: 12.5px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { font-weight: 600; font-size: 12px; }
.badge.pass { color: var(--good-ink); }
.badge.warn { color: var(--ink); }
.badge.warn::first-letter { color: var(--warning); }
.badge.fail { color: var(--critical); }
.grid { stroke: var(--grid); stroke-width: 1; }
.sep { stroke: var(--axis); stroke-width: 1; }
.threshold { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }
.band-warn { fill: var(--band-warn); }
.band-fail { fill: var(--band-fail); }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.label { fill: var(--ink); font-size: 12px; }
.muted { fill: var(--muted); }
.axis-label { fill: var(--ink-2); font-size: 12px; }
.stem { stroke: var(--axis); stroke-width: 1; }
.dot { stroke: var(--surface); stroke-width: 2; }
.dot.pass { fill: var(--s1); }
.dot.warn { fill: var(--warning); }
.dot.fail { fill: var(--critical); }
.hit { fill: transparent; }
.pt:hover .dot, .pt:hover .mk { stroke: var(--ink); }
.diag { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }
.sc { fill: var(--s1); fill-opacity: 0.55; }
.sc.fail { fill: var(--critical); fill-opacity: 1; }
.sc:hover { fill-opacity: 1; stroke: var(--ink); }
.line { fill: none; stroke-width: 2; }
.mk { stroke: var(--surface); stroke-width: 2; }
.line.s1 { stroke: var(--s1); } .mk.s1 { fill: var(--s1); }
.line.s2 { stroke: var(--s2); } .mk.s2 { fill: var(--s2); }
.line.s3 { stroke: var(--s3); } .mk.s3 { fill: var(--s3); }
.line.s4 { stroke: var(--s4); } .mk.s4 { fill: var(--s4); }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--ink-2); margin-bottom: 8px; }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
"""


def build_report(results, curves, args, elapsed):

    n_pass = sum(r.status == "PASS" for r in results)
    n_warn = sum(r.status == "WARN" for r in results)
    n_fail = sum(r.status == "FAIL" for r in results)
    worst = max(results, key=lambda r: r.rel)
    n_elems = sum(r.analytic.size for r in results)

    if n_fail == 0 and n_warn == 0:
        verdict = ("pass", "✓ All backward formulas match the numerical gradient.")
    elif n_fail == 0:
        verdict = ("pass", "! No FAILs, but some checks are WARN. "
                           "If those gradients are very small, it may be numerical noise.")
    else:
        verdict = ("fail", f"✕ {n_fail} checks do not match. See the FAIL rows in the table.")

    full = [r for r in results if r.group == "Full model"]

    legend = "".join(
        f'<span><i style="background: var(--s{i + 1})"></i>{esc(k)}</span>'
        for i, k in enumerate(curves)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TinyGPT Gradient Check</title>
<style>{CSS}</style>
</head>
<body>
<main>

<h1>TinyGPT Gradient Check</h1>
<p class="sub">Analytic gradients in tinygpt.py vs. central-difference numerical gradients ·
h = {args.h:g} · init = {esc(args.init)} · seed = {args.seed} ·
{time.strftime('%Y-%m-%d %H:%M')} · {elapsed:.1f}s</p>

<div class="tiles">
  <div class="tile"><div class="k">Checks</div><div class="v">{len(results)}</div></div>
  <div class="tile"><div class="k">Elements compared</div><div class="v">{n_elems:,}</div></div>
  <div class="tile"><div class="k">PASS / WARN / FAIL</div><div class="v">{n_pass} / {n_warn} / {n_fail}</div></div>
  <div class="tile"><div class="k">Worst rel error ({esc(worst.name)})</div><div class="v">{fmt(worst.rel)}</div></div>
</div>
<p class="verdict {verdict[0]}">{verdict[1]}</p>

<section>
<h2>Relative error per check</h2>
<p class="desc">rel = ‖G<sub>analytic</sub> − G<sub>numerical</sub>‖ / (‖G<sub>analytic</sub>‖ + ‖G<sub>numerical</sub>‖).
Further left is more accurate. The yellow band is WARN, the red band is FAIL. Hover a dot for details.</p>
{svg_dot_plot(results, args.h)}
</section>

<section>
<h2>Analytic vs numerical (Full model, every element)</h2>
<p class="desc">Each dot is one parameter element. With correct formulas every dot lies on the dashed y = x line.
Dots off the diagonal are shown in red.</p>
{svg_scatter(full)}
</section>

<section>
<h2>Error vs h (Full model)</h2>
<p class="desc">Large h increases truncation error (∝h²) and small h increases round-off error (∝ε/h), giving a V shape.
With correct formulas the bottom drops well below the PASS limit; with wrong ones it stays high regardless of h.
Each curve uses 16 random elements of that parameter.</p>
<div class="legend">{legend}</div>
{svg_h_sweep(curves, args.h)}
</section>

<section>
<h2>Detailed results</h2>
{results_table(results)}
</section>

</main>
</body>
</html>
"""


# ====================================================================
# Main
# ====================================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])

    parser.add_argument("--h", type=float, default=1e-5,
                        help="central-difference step size (default 1e-5)")
    parser.add_argument("--init", choices=["scaled", "original"], default="scaled",
                        help="scaled: raise E, P to std 1 / original: tinygpt default init")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("gradcheck_report.html"))
    parser.add_argument("--no-open", action="store_true",
                        help="do not open the report in the browser")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = make_model(args.init, args.seed)

    start = time.time()

    results = []

    checks = [
        ("GELU", lambda: check_gelu(model, rng, args.h)),
        ("LayerNorm", lambda: check_layernorm(model, rng, args.h)),
        ("Attention head", lambda: check_attention_head(model, rng, args.h)),
        ("Block", lambda: check_block(model, rng, args.h)),
        ("Full model", lambda: check_full_model(model, args.h)),
    ]

    for name, run in checks:

        print(f"\n[{name}]")

        for r in run():
            results.append(r)
            print(
                f"  {r.name:12s} rel {fmt(r.rel)}   "
                f"max|diff| {fmt(r.max_abs)}   {r.status}"
            )

    print("\n[h sweep]")
    curves = h_sweep(model, rng)
    for k, errs in curves.items():
        print(f"  {k:12s} min rel {fmt(min(errs))} at h = {SWEEP_HS[int(np.argmin(errs))]:.0e}")

    elapsed = time.time() - start

    n_fail = sum(r.status == "FAIL" for r in results)
    n_warn = sum(r.status == "WARN" for r in results)
    worst = max(results, key=lambda r: r.rel)

    print(
        f"\n{len(results)} checks | PASS {len(results) - n_fail - n_warn} "
        f"| WARN {n_warn} | FAIL {n_fail} "
        f"| worst {worst.group}/{worst.name} {fmt(worst.rel)} | {elapsed:.1f}s"
    )

    args.out.write_text(build_report(results, curves, args, elapsed), encoding="utf-8")
    print(f"report: {args.out}")

    if not args.no_open:
        webbrowser.open(args.out.resolve().as_uri())

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
