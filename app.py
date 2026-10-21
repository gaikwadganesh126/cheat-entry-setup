"""
Cheat Entry Ranking System | Streamlit dashboard (Hugging Face Spaces).
Run locally:  streamlit run app.py
The screening logic lives in screener.py.
"""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import screener as s

st.set_page_config(page_title="Cheat Entry Precondition Scanner", page_icon="⚡", layout="wide")

CACHE_TTL = 4 * 3600                       # scans are shared by all users for 4 hours
ALL = "All conditions"
SETUPS = list(s.GROUPS)                    # Final Entry Check, AT Dashboard, Multi RS, All conditions
ICONS = {"Y": "✅", "N": "❌", "-": "➖"}

# Short, readable names shown in the UI instead of C1..C13 / F1..F7
SHORT = {
    "C1": "Mkt Cap", "C2": "5Y Uptrend", "C3": "ATH Pullback", "C4": "RS vs CNX500",
    "C5": "RS Nifty50", "C6": "RS CNX500", "C7": "RS Midcap50", "C8": "RS Smallcap250",
    "C9": "WVol > 20SMA", "C10": "Red Vol Dry", "C11": "Wick % < 25", "C12": "U/D Vol Ratio",
    "C13": "RSI 14 (W)", "F1": "Prior Trend", "F2": "Above Fib", "F3": "Above DMAs",
    "F4": "ATH Room", "F5": "AT Dash OK", "F6": "Multi RS OK", "F7": "Above AVWAP",
    "C14": "RS Sector",
}


def rules_md(keys):
    return "\n".join(f"- **{SHORT[k]}** — {s.LABELS[k]}" for k in keys)


def signal(n_failed, final):
    if n_failed == 0:
        return "🟢 PASS"
    return "🔴 FAIL" if final or n_failed > 1 else "🟡 WATCH"


# ====================================================================== STYLE
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1400px}
.hero{background:linear-gradient(135deg,#0b1f3a 0%,#123a6b 55%,#1e5bb8 100%);
      padding:26px 20px;border-radius:16px;text-align:center;
      box-shadow:0 8px 24px rgba(0,0,0,.35);margin-bottom:18px}
.hero h1{color:#fff;margin:0;font-size:34px;font-weight:800;letter-spacing:.5px}
.hero p{color:#c9d6ea;margin:6px 0 0;font-size:16px}
.chip{display:inline-block;padding:4px 12px;border-radius:999px;margin:0 4px;
      background:#111827;border:1px solid #1f2937;color:#e5e7eb;font-size:13px}
section[data-testid="stSidebar"] .stButton button{border-radius:10px;font-weight:600}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
</style>""", unsafe_allow_html=True)


# ====================================================================== ABOUT
@st.dialog("ℹ️ About — Cheat Entry preconditions scanner", width="large")
def about():
    st.markdown("""
Scans NSE stocks on the **weekly chart** and ranks positional swing *cheat entries*:
strong long-term leaders that pulled back from their all-time high (ATH) and are
turning up again with relative strength and volume.

**How it works**
1. Choose a **setup** and a **universe** (index, all NSE stocks, or your watchlist), press **SCAN**.
2. Weekly candles come from Yahoo Finance; stocks with less than 5 years of data are skipped.
3. Every stock is tested against **all conditions**; results show the selected setup.

| Signal | Meaning |
|---|---|
| 🟢 **PASS** | every check of the setup passes |
| 🟡 **WATCH** | only one check missed (AT Dashboard / Multi RS only) |
| 🔴 **FAIL** | setup not met |
""")
    for g in SETUPS:
        keys = s.GROUPS[g]
        with st.expander(f"**{g}** — {len(keys)} checks", expanded=g == s.FINAL):
            st.markdown(rules_md(keys))
    st.markdown("**Benchmarks:** CNX500 · NIFTY 50 · NIFTY Midcap 50 · NIFTY Smallcap 250 · "
                "own sector index (Bank, IT, Pharma, Auto, FMCG, Metal, Realty, Media, Oil & Gas, "
                "Healthcare, Consumer Durables, Financial Services, Energy, Infra; "
                "other stocks: Industrials / Nifty India Manufacturing)")
    st.caption("Weekly signals are final after the Friday close. For education only, not investment advice.")


# ====================================================================== CACHED DATA
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def symbols(index):
    return s.nse_symbols(index)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def scan(syms, names, group, check_all, _progress=None):
    # "_progress" starts with "_" so Streamlit does not use it in the cache key
    sl, all_df = s.run_scan(list(syms), dict(names), lambda *_: None, group, check_all,
                            progress=_progress)
    return sl, all_df


def progress_bar(total):
    """Progress bar with % done, items remaining and estimated time left."""
    bar = st.progress(0.0, text=f"Starting scan of {total} stocks...")
    t0, last = dt.datetime.now(), {"pct": -1}

    def update(frac, text):
        pct = int(frac * 100)
        if pct == last["pct"] and frac < 1:      # redraw only when % changes
            return
        last["pct"] = pct
        secs = (dt.datetime.now() - t0).total_seconds()
        eta = ""
        if 0.05 < frac < 1:
            left = int(secs / frac * (1 - frac))
            eta = f" · ~{left // 60}m {left % 60:02d}s left"
        bar.progress(min(frac, 1.0), text=f"**{pct}%** · {text}{eta}")

    return bar, update


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def weekly(ticker):
    return s._history(ticker, "3y", "1wk")


# ====================================================================== SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Scanner")
    group = st.selectbox("Setup", SETUPS, index=0)
    universe = st.radio("Universe", ["Index", "All NSE stocks", "Watchlist"], horizontal=True)
    index, watch = None, ""
    if universe == "Index":
        names_ = list(s.INDEX_LISTS)
        index = st.selectbox("Index", names_, index=names_.index("Nifty 500"))
    elif universe == "Watchlist":
        watch = st.text_area("Symbols", placeholder="HAL, TCS, BEL", height=100)
    else:
        st.caption("~2,000 stocks · first run takes 5-10 minutes")
    go_btn = st.button("⚡ SCAN", type="primary", width="stretch")
    stop_slot = st.empty()                      # "Stop scan" button appears here while scanning
    if st.button("🔄 Clear cache", width="stretch"):
        st.cache_data.clear()
        st.session_state.pop("res", None)


def _request_stop():
    st.session_state["stopped"] = True


# ====================================================================== HEADER
_, mid, right = st.columns([1, 6, 1], vertical_alignment="center")
with mid:
    st.markdown("""<div class="hero"><h1>Cheat Entry</h1>
<p>Cheat Entry Precondition Scanner</p></div>""", unsafe_allow_html=True)
with right:
    if st.button("ℹ️ About", width="stretch"):
        about()


# ====================================================================== RUN SCAN
if go_btn:
    is_watch = universe == "Watchlist"
    syms, names = [], {}
    if is_watch:
        syms = s.parse_manual(watch)
        if not syms:
            st.error("Type at least one symbol, e.g. `HAL, TCS`.")
            st.stop()
        try:
            names = symbols(None)                # company names only; optional
        except s.ScreenerError:
            st.toast("NSE list unavailable: company names will be blank.", icon="⚠️")
    else:
        try:
            names = symbols(index)
        except s.ScreenerError as e:
            st.error(f"📡 Could not load the stock list. {e}")
            st.stop()
        syms = list(names)
        if not syms:
            st.error("The stock list from NSE is empty. Try again later.")
            st.stop()

    bar, update = progress_bar(len(syms))
    st.session_state["stopped"] = False
    # Clicking Stop makes Streamlit rerun the page, which interrupts this scan at the
    # next progress update; screener then cancels the remaining downloads.
    stop_slot.button("⏹ Stop scan", width="stretch", on_click=_request_stop, key="stop_btn")
    try:
        sl, all_df = scan(tuple(syms), tuple(sorted(names.items())), group, is_watch,
                          _progress=update)
    except s.ScreenerError as e:                 # expected: data source down, bad input
        st.error(f"📡 Scan failed: {e}")
    except Exception as e:                       # unexpected bug: show details
        st.error(f"Unexpected error during scan: {e.__class__.__name__}: {e}")
        with st.expander("Error details"):
            st.exception(e)
    else:
        if all_df.empty:
            st.warning("No stock could be scanned: symbols are wrong, have less than 5 years "
                       "of data, or Yahoo Finance is not responding.")
        st.session_state["res"] = dict(sl=sl, all_df=all_df, group=group, watch=is_watch, syms=syms)
    finally:
        bar.empty()
        stop_slot.empty()

if st.session_state.pop("stopped", False):
    st.warning("⏹ Scan stopped. Press **⚡ SCAN** to start again."
               + (" Showing the previous results." if st.session_state.get("res") else ""))

res = st.session_state.get("res")
if not res:
    st.markdown("<div style='text-align:center;margin-top:40px'>"
                "<span class='chip'>📅 Weekly timeframe</span>"
                "<span class='chip'>🇮🇳 NSE stocks</span>"
                "<span class='chip'>📈 Positional swing</span>"
                "<p style='margin-top:22px;color:#9ca3af'>👈 Choose a setup and universe, "
                "then press <b>⚡ SCAN</b></p></div>", unsafe_allow_html=True)
    st.stop()


# ====================================================================== RESULTS
all_df, final = res["all_df"], res["group"] == s.FINAL
view = (all_df.sort_values(["n_failed", "rsi14w"], ascending=[True, False]) if res["watch"]
        else res["sl"])

final_keys = s.GROUPS[s.FINAL]
check_help = {SHORT[k]: st.column_config.TextColumn(SHORT[k], help=s.LABELS[k], width="small")
              for k in s.LABELS}


def check_table(keys, score_col):
    t = pd.DataFrame({
        "Symbol": view["symbol"].values,
        "Chart": ("https://www.tradingview.com/chart/?symbol=NSE:" + view["symbol"]).values,
        "Company": view["company"].values,
        "Signal": [signal(x, final) for x in view["n_failed"]],
        "Score": (view[score_col].values if score_col in view
                  else [f"{sum(r[k] == 'Y' for k in keys)}/{len(keys)}" for _, r in view.iterrows()]),
    })
    for k in keys:
        t[SHORT[k]] = view[k].map(ICONS).values
    return t


def show(t):
    st.dataframe(t, hide_index=True, width="stretch", height=min(620, 38 + 35 * len(t)),
                 column_config={
                     **check_help,
                     "Chart": st.column_config.LinkColumn("Chart", display_text="TradingView ↗"),
                     "Score": st.column_config.TextColumn("Score", help="Checks passed"),
                     "Final": st.column_config.TextColumn("Final", help="Final Entry checks passed"),
                 })


tab_res, tab_chart = st.tabs(["📋 Results", "📈 Chart"])

with tab_res:
    if view.empty:
        st.warning("No stock meets this setup right now: no trade.")
    else:
        sel = res["group"]
        # Only the selected setup is shown; "All conditions" shows every section
        sections = ([s.FINAL, "AT Dashboard", "Multi RS"] if sel == ALL else [sel])
        icons = {s.FINAL: "✅", "AT Dashboard": "📊", "Multi RS": "📈"}
        main = None
        for g in sections:
            st.markdown(f"#### {icons.get(g, '📋')} {g}")
            t = check_table(s.GROUPS[g], g)
            if g == "Multi RS" and "sector" in view:
                t.insert(5, "Sector", view["sector"].values)
            show(t)
            if main is None:
                main = t                         # table exported to CSV
        st.caption("✅ pass · ❌ fail · ➖ not checked — hover a column header for its rule")

        csv = main.drop(columns="Chart").assign(
            Signal=lambda d: d["Signal"].str[2:],
            Score=lambda d: d["Score"].str.replace("/", " of "),   # avoid Excel date conversion
        ).replace(list(ICONS.values()), ["PASS", "FAIL", "-"])
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button("⬇️ Export CSV", csv.to_csv(index=False).encode("utf-8-sig"),
                           f"cheat_entry_{stamp}.csv", "text/csv")
        if res["watch"]:
            skipped = sorted({x.removesuffix(".NS") for x in res["syms"]} - set(view["symbol"]))
            if skipped:
                st.caption(f"⚠️ Skipped: {', '.join(skipped)} (less than 5 years of data, or wrong symbol)")

with tab_chart:
    pool = list(view["symbol"]) if len(view) else []
    if not pool:
        st.info("No stocks to chart.")
    else:
        sym = st.selectbox("Stock", pool)
        row = all_df.set_index("symbol").loc[sym]
        try:
            df = weekly(sym + ".NS")
        except Exception as e:
            df = None
            st.caption(f"Chart download error: {e}")
        if df is None or df.empty:
            st.error("No chart data from Yahoo Finance right now. Try again later.")
        else:
            fig = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                           low=df["Low"], close=df["Close"], name=sym,
                                           increasing_line_color="#26a69a",
                                           decreasing_line_color="#ef5350"))
            fig.add_scatter(x=df.index, y=df["Close"].rolling(s.CFG["sma_weeks"]).mean(),
                            name="40W SMA", line=dict(color="#f4b400", width=1.5))
            for col, name, colr, dash in [("ath", "ATH", "#ef5350", "dash"),
                                          ("fib382_price", "0.382 Fib", "#42a5f5", "dot"),
                                          ("avwap_ath", "AVWAP (ATH)", "#ab47bc", "dash"),
                                          ("swing_low", "Swing low", "#26a69a", "dot")]:
                if pd.isna(row.get(col)):
                    continue                     # level not available for this stock
                fig.add_hline(y=row[col], line=dict(color=colr, dash=dash, width=1),
                              annotation_text=f"{name} {row[col]:,.2f}",
                              annotation_position="right")
            fig.update_layout(template="plotly_dark", height=560, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10), title=f"{sym} · weekly",
                              legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, width="stretch")