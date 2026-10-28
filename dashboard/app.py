"""
Foundermatcha - Key Findings Dashboard
======================================
    streamlit run dashboard/app.py

Needs users_clean.csv / matches_clean.csv / events_clean.csv. Enter their folder
in the sidebar (defaults to the folder you run the command from).

Shows only the six findings the project defends. Every live figure recomputes
from the CSVs; with filters cleared they reproduce the verified numbers.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Foundermatcha - Key Findings", layout="wide")

BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d03b3b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = [[0.0, "#f4f7fb"], [0.33, "#9ec5f4"], [0.66, "#3987e5"], [1.0, "#184f95"]]
STAGES = ["signup", "email_verified", "profile_updated", "match_suggested",
          "match_viewed", "intro_requested", "intro_sent", "meeting_booked"]
LABELS = ["Signed up", "Email verified", "Profile completed", "Match suggested",
          "Match viewed", "Intro requested", "Intro sent", "Meeting booked"]


def fx(f, h=380, leg=True):
    f.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), showlegend=leg,
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="system-ui, sans-serif", size=12, color=INK2),
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                    hoverlabel=dict(bgcolor="white", font_size=12))
    f.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    f.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return f


@st.cache_data(show_spinner="Loading...")
def load(folder):
    import pathlib
    d = pathlib.Path(folder)
    need = ["users_clean.csv", "matches_clean.csv", "events_clean.csv"]
    miss = [f for f in need if not (d / f).exists()]
    if miss:
        return None, None, None, miss
    u = pd.read_csv(d / need[0])
    m = pd.read_csv(d / need[1])
    e = pd.read_csv(d / need[2])
    u["signup_date"] = pd.to_datetime(u["signup_date"])
    e["event_timestamp"] = pd.to_datetime(e["event_timestamp"])
    return u, m, e, []


def cramers_v(x, y):
    """Bias-corrected Cramer's V (Bergsma 2013). 0 = nothing beyond chance."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if d.empty:
        return 0.0
    ct = pd.crosstab(d.x, d.y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    c = stats.chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    r, k = ct.shape
    p2 = max(0.0, c / n - ((k - 1) * (r - 1)) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(p2 / den)) if den > 0 else 0.0


def q5(s):
    try:
        return pd.qcut(s, 5, duplicates="drop").astype(str)
    except Exception:
        return s.astype(str)


def claim(text):
    """The finding being demonstrated, stated before the evidence."""
    st.markdown(
        "<div style='border-left:3px solid #2a78d6;padding:10px 16px;margin:4px 0 14px;"
        "background:#fbfcfe'><b>Claim.</b> " + text + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("Foundermatcha")
st.sidebar.caption("Key findings")
folder = st.sidebar.text_input("Data folder", "data")
U, M, E, miss = load(folder)
if miss:
    st.error("Missing: " + ", ".join(miss))
    st.info("Run Objective 1 first, or point 'Data folder' at the cleaned CSVs.")
    st.stop()

f_type = st.sidebar.multiselect("User type", ["founder", "engineer"], [])
f_loc = st.sidebar.multiselect("Location", sorted(U.location.dropna().unique()), [])
f_score = st.sidebar.slider("Algorithm score", 0, 100, (0, 100))

u = U
if f_type:
    u = u[u.user_type.isin(f_type)]
if f_loc:
    u = u[u.location.isin(f_loc)]
# Cascade only when a user filter actually removed rows. Otherwise the known
# orphan foreign keys would drop out and the totals would stop matching the
# verified figures (59 event user_ids and 1 match are unresolvable).
if len(u) == len(U):
    e, m = E, M
else:
    ids = set(u.user_id)
    e = E[E.user_id.isin(ids)]
    m = M[M.founder_id.isin(ids) | M.engineer_id.isin(ids)]
if tuple(f_score) != (0, 100):
    m = m[(m.algorithm_score >= f_score[0]) & (m.algorithm_score <= f_score[1])]

default = (not f_type and not f_loc and tuple(f_score) == (0, 100))
st.sidebar.markdown("---")
st.sidebar.metric("Users", "{:,}".format(len(u)))
st.sidebar.metric("Matches", "{:,}".format(len(m)))
st.sidebar.metric("Events", "{:,}".format(len(e)))
if default:
    st.sidebar.success("Unfiltered: matches the verified figures.")
else:
    st.sidebar.warning("Filtered: all figures recomputed on this subset.")
if u.empty or e.empty:
    st.warning("No data matches these filters.")
    st.stop()

# ----------------------------------------------------------------- header ---
n = [e[e.event_type == s].user_id.nunique() for s in STAGES]
lost = [None] + [n[i - 1] - n[i] for i in range(1, len(n))]
drop = [None] + [round(lost[i] / n[i - 1] * 100, 1) if n[i - 1] else None
                 for i in range(1, len(n))]
fun = pd.DataFrame({"Stage": LABELS, "Users": n, "Lost": lost, "Drop %": drop,
                    "% of signups": [round(v / (n[0] or 1) * 100, 1) for v in n]})

intro = set(e[e.event_type == "intro_sent"].user_id)
fe = e.groupby(["user_id", "event_type"])["event_timestamp"].min().reset_index()
a = fe[fe.event_type == "signup"][["user_id", "event_timestamp"]].rename(
    columns={"event_timestamp": "t0"})
b = fe[fe.event_type == "match_viewed"][["user_id", "event_timestamp"]].rename(
    columns={"event_timestamp": "t1"})
sp = a.merge(b, on="user_id")
med = fr = sr = ratio = np.nan
fast_n = slow_n = 0
if len(sp):
    sp["d"] = (sp.t1 - sp.t0).dt.total_seconds() / 86400
    sp = sp[sp.d >= 0]
    sp["c"] = sp.user_id.isin(intro)
    med = sp.d.median()
    fast, slow = sp[sp.d <= med], sp[sp.d > med]
    fast_n, slow_n = len(fast), len(slow)
    fr = fast.c.mean() * 100 if fast_n else np.nan
    sr = slow.c.mean() * 100 if slow_n else np.nan
    ratio = fr / sr if (sr and sr == sr and sr > 0) else np.nan

worst = fun.dropna(subset=["Drop %"]).sort_values("Drop %", ascending=False).head(1)

st.title("Foundermatcha - what the data supports")
st.caption("Six findings, each stated as a claim and then evidenced. Synthetic dataset.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Signup to meeting", "{}%".format(fun["% of signups"].iloc[-1]))
k2.metric("Biggest leak", "{}%".format(worst["Drop %"].iloc[0]) if len(worst) else "-")
k3.metric("Speed effect", "{:.1f}x".format(ratio) if ratio == ratio else "-")
k4.metric("Matches", "{:,}".format(len(m)))

if f_type == ["engineer"]:
    st.warning("Engineers only: intros are founder-initiated, so stages after Match "
               "viewed are structurally zero - not a leak.")

t1, t2, t3, t4, t5 = st.tabs(
    ["1. The leak", "2. Speed", "3. Who converts", "4. The algorithm", "5. Model"])

# --------------------------------------------------------------- 1. funnel ---
with t1:
    claim("The platform's conversion problem is not acquisition or onboarding. It is a "
          "single decision point: users see a match and do nothing.")
    col = [AQUA if d is None or d != d else RED if d >= 40 else YELLOW if d >= 20 else BLUE
           for d in fun["Drop %"]]
    f = go.Figure(go.Bar(x=fun.Stage, y=fun.Users, marker_color=col,
                         text=["{:,}".format(v) for v in fun.Users],
                         textposition="outside",
                         customdata=np.stack([fun["Drop %"].fillna(0),
                                              fun["Lost"].fillna(0)], axis=-1),
                         hovertemplate="<b>%{x}</b><br>%{y:,} users<br>"
                                       "lost entering: %{customdata[1]:,} "
                                       "(%{customdata[0]}%)<extra></extra>"))
    f.update_layout(title="Unique users reaching each funnel stage",
                    yaxis_title="Users")
    st.plotly_chart(fx(f, 430, False), use_container_width=True)

    if len(worst) and f_type != ["engineer"]:
        st.info("**{:,} users ({}%) are lost entering *{}*** - the largest single-stage "
                "loss in the funnel, and at a rate more than double the next worst "
                "stage. Everything upstream holds: {}% of signups reach a viewed match."
                .format(int(worst["Lost"].iloc[0]), worst["Drop %"].iloc[0],
                        worst.Stage.iloc[0], fun["% of signups"].iloc[4]))
    st.dataframe(fun, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Supporting: the two sides fail in different places.**")
    bt = {}
    for t in ("founder", "engineer"):
        ids = set(u[u.user_type == t].user_id)
        tot = len(ids) or 1
        bt[t] = [round(e[(e.event_type == s) & (e.user_id.isin(ids))].user_id.nunique()
                       / tot * 100, 1) for s in STAGES]
    f = go.Figure()
    for t, c in (("founder", BLUE), ("engineer", ORANGE)):
        cnt = int((u.user_type == t).sum())
        if not cnt:
            continue
        y = bt[t][:5] if t == "engineer" else bt[t]
        f.add_scatter(x=LABELS[:len(y)], y=y, mode="lines+markers",
                      name="{} (n={:,})".format(t.title(), cnt),
                      line=dict(color=c, width=2), marker=dict(size=8),
                      hovertemplate="%{x}<br>%{y}%<extra></extra>")
    f.update_layout(title="Share of each user type reaching each stage",
                    yaxis_title="% of that type")
    st.plotly_chart(fx(f, 340), use_container_width=True)
    st.caption("Engineers complete profiles slightly better than founders, then fall "
               "away at match suggestion - a supply problem, not an engagement one. "
               "Their line stops at Match viewed because intros are founder-initiated.")

# ---------------------------------------------------------------- 2. speed ---
with t2:
    claim("Speed to first match is the strongest lever in the dataset, and it is an "
          "engineering problem rather than a persuasion problem.")
    if ratio != ratio:
        st.info("Not enough users with both a signup and a match view in this slice.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            f = go.Figure(go.Bar(
                x=["Within {:.0f} days".format(med), "Slower"], y=[fr, sr],
                marker_color=[BLUE, ORANGE], customdata=[fast_n, slow_n],
                text=["{:.1f}%".format(fr), "{:.1f}%".format(sr)],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:.1f}%<br>n=%{customdata:,}<extra></extra>"))
            f.update_layout(title="Conversion by speed to first match view",
                            yaxis_title="% who got an intro")
            st.plotly_chart(fx(f, 360, False), use_container_width=True)
        with c2:
            st.markdown(
                "Users split at the **{:.0f}-day median** time from signup to first "
                "match view.\n\n"
                "- Viewed within {:.0f} days: **{:.1f}%** got an intro (n={:,})\n"
                "- Took longer: **{:.1f}%** (n={:,})\n\n"
                "That is a **{:.1f}x** difference - the largest effect measured "
                "anywhere in this project.\n\n"
                "The slow group has not been deliberating. It has churned. This is why "
                "the recommendation is to compress time-to-first-match rather than to "
                "send more reminders."
                .format(med, med, fr, fast_n, sr, slow_n, ratio))
        st.caption("Direction of causation is not established here: users who were "
                   "always going to engage may also view faster. The size of the gap "
                   "is what makes it worth acting on.")

# ------------------------------------------------------------- 3. segments ---
with t3:
    claim("Only three founder attributes predict conversion. Most differences that "
          "look meaningful on a chart do not survive a significance test.")
    c1, c2 = st.columns([1, 3])
    with c1:
        ut = st.radio("Measure", ["founder", "engineer"],
                      format_func=lambda s: s.title() + "s")
        attr = st.selectbox("Attribute",
                            ["availability", "acquisition_source", "location",
                             "prior_startup_experience", "personality_type",
                             "compensation_preference", "education_level", "age_band"])
    ev = "intro_sent" if ut == "founder" else "match_viewed"
    hit = set(e[e.event_type == ev].user_id)
    sub = u[u.user_type == ut].copy()
    with c2:
        if sub.empty or sub[attr].nunique() < 2:
            st.info("Not enough data for this attribute in the current slice.")
        else:
            sub["ok"] = sub.user_id.isin(hit).astype(int)
            g = (sub.groupby(attr).agg(n=("user_id", "count"), c=("ok", "sum"))
                 .reset_index())
            g["rate"] = (g.c / g.n * 100).round(1)
            g = g.sort_values("rate", ascending=False)
            p = stats.chi2_contingency(pd.crosstab(sub[attr], sub.ok))[1]
            sig = p < 0.05
            base = g.c.sum() / g.n.sum() * 100 if g.n.sum() else 0
            f = go.Figure(go.Bar(
                x=g.rate, y=g[attr].astype(str), orientation="h",
                marker_color=BLUE if sig else MUTED, customdata=g.n,
                text=["{:.1f}%".format(v) for v in g.rate], textposition="outside",
                hovertemplate="%{y}<br>%{x:.1f}%<br>n=%{customdata:,}<extra></extra>"))
            f.add_vline(x=base, line_width=1, line_color=INK2)
            f.update_layout(title="{} rate by {}".format(
                "Intro" if ut == "founder" else "Match view", attr.replace("_", " ")),
                xaxis_title="%")
            f.update_yaxes(autorange="reversed", showgrid=False)
            f.update_xaxes(showgrid=True, gridcolor=GRID)
            st.plotly_chart(fx(f, 360, False), use_container_width=True)
            if sig:
                st.success("Chi-square **p = {:.4f}** - significant. Line marks the "
                           "{:.1f}% baseline.".format(p, base))
            else:
                st.warning("Chi-square **p = {:.4f}** - NOT significant. The spread is "
                           "within what chance produces at this sample size, so it "
                           "should not be acted on.".format(p))

    st.markdown("---")
    rows = []
    for utype in ("founder", "engineer"):
        for c in ["availability", "acquisition_source", "location",
                  "prior_startup_experience", "personality_type",
                  "compensation_preference", "education_level", "age_band"]:
            s2 = u[u.user_type == utype].copy()
            if s2.empty or s2[c].nunique() < 2:
                continue
            ev2 = "intro_sent" if utype == "founder" else "match_viewed"
            h2 = set(e[e.event_type == ev2].user_id)
            s2["ok"] = s2.user_id.isin(h2).astype(int)
            try:
                pp = stats.chi2_contingency(pd.crosstab(s2[c], s2.ok))[1]
            except Exception:
                continue
            gg = s2.groupby(c).ok.mean().mul(100).round(1).sort_values(ascending=False)
            rows.append({"Type": utype.title(), "Attribute": c.replace("_", " "),
                         "p": round(pp, 4), "Significant": "Yes" if pp < 0.05 else "No",
                         "Highest": "{} {}%".format(gg.index[0], gg.iloc[0]),
                         "Lowest": "{} {}%".format(gg.index[-1], gg.iloc[-1])})
    if rows:
        tb = pd.DataFrame(rows).sort_values("p")
        st.dataframe(tb, use_container_width=True, hide_index=True)
        st.caption("{} of {} attributes reach p<0.05. Filtering shrinks the sample and "
                   "widens p-values, so a segment can lose significance simply because "
                   "the slice is smaller."
                   .format(int((tb.Significant == "Yes").sum()), len(tb)))

# ------------------------------------------------------------ 4. algorithm ---
with t4:
    claim("The matching algorithm works, but it is largely a restatement of two "
          "compatibility booleans - so a cheap flag predicts conversion as well as the "
          "score does.")
    if m.empty:
        st.info("No matches in this slice.")
    else:
        mm = m.copy()
        mm["band"] = pd.cut(mm.algorithm_score, [0, 20, 40, 60, 80, 100],
                            labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
        sb = mm.groupby("band", observed=True).agg(
            n=("match_id", "count"), i=("intro_sent", "sum"),
            mt=("outcome", lambda x: (x == "meeting_booked").sum())).reset_index()
        sb["intro"] = (sb.i / sb.n * 100).round(1)
        sb["meet"] = (sb.mt / sb.n * 100).round(1)

        c1, c2 = st.columns(2)
        with c1:
            f = go.Figure()
            f.add_bar(x=sb.band.astype(str), y=sb.intro, name="Intro rate",
                      marker_color=BLUE, customdata=sb.n,
                      hovertemplate="%{x}<br>%{y}%<br>n=%{customdata}<extra></extra>")
            f.add_bar(x=sb.band.astype(str), y=sb.meet, name="Meeting rate",
                      marker_color=ORANGE, customdata=sb.n,
                      hovertemplate="%{x}<br>%{y}%<br>n=%{customdata}<extra></extra>")
            f.update_layout(barmode="group", title="Conversion rises with score band",
                            xaxis_title="Algorithm score band", yaxis_title="%")
            st.plotly_chart(fx(f, 360), use_container_width=True)

        with c2:
            d = mm.copy()
            d["y"] = d.intro_sent.astype(int)
            d["both"] = (d.compensation_compatibility
                         & d.availability_compatibility).astype(int)
            spec = [("Compensation compatibility", "compensation_compatibility", 1),
                    ("Algorithm score", "algorithm_score", 0),
                    ("Both compatibility flags", "both", 1),
                    ("Availability compatibility", "availability_compatibility", 1),
                    ("Prior connection", "prior_connection", 1),
                    ("Skills overlap", "skills_overlap_score", 0)]
            rr = []
            for nm, cl, isb in spec:
                v = d[cl].astype(int) if isb else d[cl]
                if v.nunique() < 2 or d.y.nunique() < 2:
                    continue
                r, p = stats.pointbiserialr(v, d.y)
                rr.append({"f": nm, "r": round(r, 4), "p": p, "s": p < 0.05})
            cr = pd.DataFrame(rr).sort_values("r", ascending=False)
            if len(cr):
                f = go.Figure(go.Bar(
                    x=cr.r, y=cr.f, orientation="h",
                    marker_color=[BLUE if s else MUTED for s in cr.s],
                    text=["r={:.3f}".format(v) for v in cr.r], textposition="outside",
                    customdata=cr.p,
                    hovertemplate="%{y}<br>r=%{x:.3f}<br>p=%{customdata:.4f}"
                                  "<extra></extra>"))
                f.update_layout(title="Correlation with intro sent (blue = p<0.05)",
                                xaxis_title="Point-biserial r")
                f.update_yaxes(autorange="reversed", showgrid=False)
                f.update_xaxes(showgrid=True, gridcolor=GRID)
                st.plotly_chart(fx(f, 360, False), use_container_width=True)

        # how much of the score is just the two flags
        combo = (mm.availability_compatibility.map({True: "avail ok", False: "avail no"})
                 + " / "
                 + mm.compensation_compatibility.map({True: "comp ok", False: "comp no"}))
        gsum = mm.groupby(combo).algorithm_score.agg(["count", "mean"]).round(1)
        eta2 = (mm.groupby(combo).algorithm_score.transform("mean").var()
                / mm.algorithm_score.var()) if mm.algorithm_score.var() else np.nan
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(gsum.reset_index().rename(
                columns={"index": "combination", "count": "n", "mean": "mean score"}),
                use_container_width=True, hide_index=True)
        with c2:
            if eta2 == eta2:
                st.info("**The two booleans explain {:.1f}% of the variance in the "
                        "algorithm score** (eta-squared = {:.3f}), at roughly +19 points "
                        "per flag. So compensation compatibility out-correlating the "
                        "score is not a contradiction - the score is mostly built from "
                        "those flags plus noise.".format(eta2 * 100, eta2))
            st.caption("Skills overlap, the most intuitive matching feature, does not "
                       "reach significance at all.")

        st.markdown("---")
        st.markdown("**Feature redundancy - which inputs are actually independent?**")
        mv = (m.merge(U.add_prefix("f_").rename(columns={"f_user_id": "founder_id"}),
                      on="founder_id", how="left")
               .merge(U.add_prefix("e_").rename(columns={"e_user_id": "engineer_id"}),
                      on="engineer_id", how="left"))
        fspec = [("Algorithm score", "algorithm_score", 1),
                 ("Skills overlap", "skills_overlap_score", 1),
                 ("Avail. compat", "availability_compatibility", 0),
                 ("Comp. compat", "compensation_compatibility", 0),
                 ("Prior connection", "prior_connection", 0), ("Outcome", "outcome", 0),
                 ("F: acquisition", "f_acquisition_source", 0),
                 ("F: availability", "f_availability", 0),
                 ("F: location", "f_location", 0),
                 ("F: personality", "f_personality_type", 0),
                 ("F: profile compl", "f_profile_completeness_at_signup", 1),
                 ("E: seniority", "e_seniority_level", 0),
                 ("E: experience", "e_developer_experience_years", 1),
                 ("E: profile compl", "e_profile_completeness_at_signup", 1)]
        feat = {nm: (q5(mv[c]) if q else mv[c].astype(str))
                for nm, c, q in fspec if c in mv.columns}
        names = list(feat)
        V = np.eye(len(names))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                V[i, j] = V[j, i] = cramers_v(feat[names[i]], feat[names[j]])
        cv = pd.DataFrame(V, index=names, columns=names)
        f = go.Figure(go.Heatmap(z=cv.values, x=names, y=names, colorscale=SEQ,
                                 zmin=0, zmax=0.6, colorbar=dict(title="V"),
                                 hovertemplate="%{y}<br>%{x}<br>V=%{z:.3f}"
                                               "<extra></extra>"))
        f.update_layout(title="Cramer's V between features "
                              "(high = redundant, low = complementary)")
        f.update_xaxes(tickangle=-45, showgrid=False)
        f.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(fx(f, 560, False), use_container_width=True)
        off = cv.where(~np.eye(len(cv), dtype=bool)).unstack().dropna()
        off = off[off.index.get_level_values(0) < off.index.get_level_values(1)]
        off = off.sort_values(ascending=False).head(6)
        st.dataframe(pd.DataFrame({
            "Feature A": [i[0] for i in off.index],
            "Feature B": [i[1] for i in off.index],
            "V": [round(v, 3) for v in off.values],
            "Reading": ["REDUNDANT" if v >= 0.4 else "overlapping" if v >= 0.15
                        else "mild" for v in off.values]}),
            use_container_width=True, hide_index=True)
        st.caption("Engineer seniority and experience-years are the same variable twice "
                   "- seniority is a deterministic re-binning of experience, so never "
                   "feed both to one model.")

# ---------------------------------------------------------------- 5. model ---
with t5:
    claim("A model built on these features predicts conversion only modestly, and no "
          "non-linear method beats a linear one - so the interpretable model should "
          "ship.")
    st.caption("Verified Objective 5 results: 5-fold x 5-repeat stratified CV, seed 42, "
               "leakage-free features. Static by design - refitting nine models on every "
               "filter change would take about a minute per interaction.")
    rank = pd.DataFrame([
        ("Logistic Regression", 0.6228, 0.0258, None, "WINNER"),
        ("Random Forest", 0.6188, 0.0288, 0.2162, "not significantly different"),
        ("Gradient Boosting", 0.6174, 0.0307, 0.1475, "not significantly different"),
        ("Extra Trees", 0.6155, 0.0276, 0.0618, "marginally worse"),
        ("HistGradientBoosting", 0.6088, 0.0247, 0.0015, "significantly worse"),
        ("SVM (RBF)", 0.6017, 0.0302, 0.0000, "significantly worse"),
        ("Naive Bayes", 0.6005, 0.0311, 0.0001, "significantly worse"),
        ("K-Nearest Neighbors", 0.5972, 0.0363, 0.0002, "significantly worse"),
        ("Decision Tree", 0.5787, 0.0323, 0.0000, "significantly worse"),
    ], columns=["Model", "ROC-AUC", "SD", "p vs winner", "Verdict"])
    c1, c2 = st.columns([3, 2])
    with c1:
        colr = [BLUE if v == "WINNER" else AQUA if "not significantly" in v
                else YELLOW if "marginally" in v else MUTED for v in rank.Verdict]
        f = go.Figure(go.Scatter(
            x=rank["ROC-AUC"], y=rank.Model, mode="markers",
            marker=dict(size=13, color=colr),
            error_x=dict(type="data", array=rank.SD, color=MUTED, thickness=1.5),
            customdata=rank.SD,
            hovertemplate="%{y}<br>AUC %{x:.4f} +/- %{customdata:.4f}<extra></extra>"))
        f.add_vline(x=0.5771, line_width=1.5, line_color=ORANGE)
        f.update_layout(title="Nine models, mean ROC-AUC (whiskers = 1 SD)",
                        xaxis_title="ROC-AUC")
        f.update_xaxes(range=[0.55, 0.66], showgrid=True, gridcolor=GRID)
        f.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(fx(f, 420, False), use_container_width=True)
        st.caption("Orange line is the algorithm score alone (0.577). A dot plot, not "
                   "bars: the axis starts at 0.55, so bar length would exaggerate the "
                   "differences.")
    with c2:
        st.dataframe(rank, use_container_width=True, hide_index=True)
    st.info("**Logistic regression leads but does not beat the field.** Random Forest "
            "(p=0.216) and Gradient Boosting (p=0.148) are statistically tied with it; "
            "Extra Trees is marginal (p=0.062). Five of eight are beaten significantly. "
            "The defensible claim is not 'logistic regression is best' but *nothing "
            "non-linear improves on a linear model here* - a statement about the data, "
            "not about one algorithm's luck.")
    st.warning("**AUC 0.623 is modest**: 12.3pp above random, 4.6pp above the algorithm "
               "score alone. Use it to rank a queue of matches, never to judge an "
               "individual pairing. The imputation flag for personality also carries "
               "real importance, which is leakage-adjacent and unresolved.")

st.markdown("---")
st.caption("All live figures recompute from the cleaned CSVs. Cleared filters reproduce "
           "the verified project figures: 2,000 users / 1,917 matches / 12,631 events, "
           "13.9% signup-to-meeting, 62.0% leak, 8.5x speed effect.")