"""
Cheat Entry - Preconditions: NSE weekly-chart screener (data: Yahoo Finance).

Flow
  1. stock list from NSE (all EQ stocks, one index, or typed symbols)
  2. 6 years of WEEKLY candles per stock; < 5 years of data -> skipped
  3. C2-C13 and F1, F2, F4, F7 from the weekly candles
  4. only for stocks that can still pass: market cap (C1) + daily SMAs (F3)
  5. PASS / FAIL (Final Entry Check) or PASS / MISS 1 (other groups)

Averages: every average is an SMA (simple moving average), except inside RSI,
which uses Wilder's EMA (RMA), same as TradingView.

CLI:  python screener.py --index "Nifty 500" [--group "Multi RS"]
      python screener.py --symbols HAL TCS
"""
import io
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ====================================================================== CONFIG
CFG = dict(
    min_mcap_cr=1000,            # C1  market cap > 1000 Cr
    sma_weeks=40,                # C2  40-week SMA of close (~200-day MA)
    slope_weeks=10,              # C2  SMA higher than 10 weeks ago = rising
    years_weeks=260,             # 5 years = 260 weekly candles (C2, C3, F-checks,
                                 #     and FIRST FILTER: less history -> skipped)
    ath_dist=(0.10, 0.30),       # C3  10-30% below 5-year ATH
    rs_short=10, rs_weeks=11,    # C4  10 weeks (= 50 days); C5-C8 11 weeks
    vol_sma=20,                  # C9/C10  20-week SMA of volume
    red_lookback=11,             # C10 red weeks checked in the last 11 weeks
    red_factor=1.0,              # C10 red-week volume < 1.0 x 20W SMA (0.8 = stricter)
    wick_max=0.25,               # C11 upper wick < 25% of candle range
    vol_ratio_weeks=10,          # C12 10 weeks (= 50 days)
    rsi_period=14, rsi_min=55,   # C13 weekly RSI(14) > 55
    fib_max_retrace=0.50,        # F1  no pullback deeper than 0.50 fib
    fib_min_leg=0.20,            # F1  pullbacks counted once leg is >= 20% up
    fib_min=0.382,               # F2  price above 0.382 fib
    daily_sma=(50, 150, 200),    # F3  CMP above these daily SMAs
    ath_min_dist=0.10,           # F4  CMP more than 10% below ATH
    multirs_max_miss=1,          # F6  Multi RS may miss at most 1
    history="6y", workers=16, retries=2, timeout=30,   # download settings
)

# Benchmark indices; the first ticker that returns data is used
INDICES = {"nifty500": ["^CRSLDX"], "nifty50": ["^NSEI"],
           "midcap": ["^NSEMDCP50", "NIFTY_MIDCAP_100.NS"],
           "smallcap": ["NIFTYSMLCAP250.NS", "^CNXSC"]}

LABELS = {
    "C1": "Market cap > 1000 Cr",
    "C2": "Up move for 5 years",
    "C3": "10-30% below 5Y ATH",
    "C4": "RS vs CNX500 (10W / 50D) > 0",
    "C5": "RS vs Nifty50 (11W) > 0",
    "C6": "RS vs CNX500 (11W) > 0",
    "C7": "RS vs Nifty Midcap 50 (11W) > 0",
    "C8": "RS vs Nifty Smallcap 250 (11W) > 0",
    "C9": "Week volume > 20W SMA volume",
    "C10": "Red-week volume below 20W SMA (11W)",
    "C11": "Upper wick < 25%",
    "C12": "Volume ratio (10W) > 1",
    "C13": "Weekly RSI(14) > 55",
    "F1": "Prior uptrend: no retrace > 0.50 fib & rising",
    "F2": "Price above 0.382 fib (ATH -> correction low)",
    "F3": "CMP > 50, 150 & 200-day SMA",
    "F4": "CMP more than 10% below ATH",
    "F5": "ATH Dashboard: all 6 pass",
    "F6": "Multi RS: all pass or only 1 missed",
    "F7": "Weekly close > anchored VWAP (from ATH)",
}
FINAL = "Final Entry Check"
GROUPS = {
    FINAL: [f"F{i}" for i in range(1, 8)],                  # strict PASS / FAIL
    "ATH Dashboard": ["C1", "C4", "C9", "C11", "C12", "C13"],
    "Multi RS": ["C2", "C3", "C5", "C6", "C7", "C8", "C10"],
    "All conditions": [f"C{i}" for i in range(1, 14)],
}

NSE_URLS = ["https://archives.nseindia.com/content/",
            "https://nsearchives.nseindia.com/content/"]
# Index name: (NSE list file "indices/ind_<file>list.csv", TradingView code)
INDEX_LISTS = {
    # ---- broad market
    "Nifty 50": ("nifty50", "NIFTY"),
    "Nifty Next 50": ("niftynext50", "NIFTYJR"),
    "Nifty 100": ("nifty100", "CNX100"),
    "Nifty 200": ("nifty200", "CNX200"),
    "Nifty 500": ("nifty500", "CNX500"),
    "Nifty Total Market": ("niftytotalmarket_", "NIFTY_TOTAL_MKT"),
    "Nifty LargeMidcap 250": ("niftylargemidcap250", "NIFTY_LARGEMID250"),
    "Nifty MidSmallcap 400": ("niftymidsmallcap400", "NIFTY_MIDSML_400"),
    "Nifty Midcap Select": ("niftymidcapselect_", "NIFTY_MID_SELECT"),
    "Nifty Midcap 50": ("niftymidcap50", "NIFTYMIDCAP50"),
    "Nifty Midcap 100": ("niftymidcap100", "CNXMIDCAP"),
    "Nifty Midcap 150": ("niftymidcap150", "NIFTYMIDCAP150"),
    "Nifty Smallcap 50": ("niftysmallcap50", "NIFTYSMLCAP50"),
    "Nifty Smallcap 100": ("niftysmallcap100", "CNXSMALLCAP"),
    "Nifty Smallcap 250": ("niftysmallcap250", "NIFTYSMLCAP250"),
    "Nifty Microcap 250": ("niftymicrocap250_", "NIFTY_MICROCAP250"),
    # ---- sectoral
    "Nifty Bank": ("niftybank", "BANKNIFTY"),
    "Nifty Private Bank": ("nifty_privatebank", "NIFTYPVTBANK"),
    "Nifty PSU Bank": ("niftypsubank", "CNXPSUBANK"),
    "Nifty Financial Services": ("niftyfinance", "CNXFINANCE"),
    "Nifty Fin Services 25/50": ("niftyfinancialservices25_50", "NIFTY_FIN_SERVICE"),
    "Nifty IT": ("niftyit", "CNXIT"),
    "Nifty Auto": ("niftyauto", "CNXAUTO"),
    "Nifty Pharma": ("niftypharma", "CNXPHARMA"),
    "Nifty Healthcare": ("niftyhealthcare", "NIFTY_HEALTHCARE"),
    "Nifty FMCG": ("niftyfmcg", "CNXFMCG"),
    "Nifty Metal": ("niftymetal", "CNXMETAL"),
    "Nifty Realty": ("niftyrealty", "CNXREALTY"),
    "Nifty Media": ("niftymedia", "CNXMEDIA"),
    "Nifty Energy": ("niftyenergy", "CNXENERGY"),
    "Nifty Oil & Gas": ("niftyoilgas", "NIFTY_OIL_AND_GAS"),
    "Nifty Consumer Durables": ("niftyconsumerdurables", "NIFTY_CONSR_DURBL"),
    # ---- thematic
    "Nifty India Defence": ("niftyindiadefence_", "NIFTY_IND_DEFENCE"),
    "Nifty India Digital": ("niftyindiadigital_", "NIFTY_IND_DIGITAL"),
    "Nifty India Tourism": ("niftyindiatourism_", "NIFTY_IND_TOURISM"),
    "Nifty Infrastructure": ("niftyinfra", "CNXINFRA"),
    "Nifty Consumption": ("niftyconsumption", "CNXCONSUMPTION"),
    "Nifty Commodities": ("niftycommodities", "CNXCOMMODITIES"),
    "Nifty CPSE": ("niftycpse", "CPSE"),
    "Nifty PSE": ("niftypse", "CNXPSE"),
    "Nifty MNC": ("niftymnc", "CNXMNC"),
}


# ====================================================================== SYMBOLS
def _nse_csv(path):
    err = None
    for base in NSE_URLS:
        try:
            r = requests.get(base + path, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124"})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            err = e
    raise RuntimeError(f"Could not download from NSE ({err})")


def nse_symbols(index=None):
    """{"TCS.NS": "Tata Consultancy Services Ltd.", ...} for one index,
    or for all NSE equity (SERIES == EQ) stocks when index is None."""
    if index:
        df, sym, name = _nse_csv(f"indices/ind_{INDEX_LISTS[index][0]}list.csv"), "Symbol", "Company Name"
    else:
        df, sym, name = _nse_csv("equities/EQUITY_L.csv"), "SYMBOL", "NAME OF COMPANY"
        df = df[df["SERIES"].str.strip() == "EQ"]
    return {f"{s.strip()}.NS": n.strip() for s, n in zip(df[sym], df[name])}


def parse_manual(text):
    """'tcs, HAL  infy.ns' -> ['TCS.NS', 'HAL.NS', 'INFY.NS']"""
    syms = [s.upper().removesuffix(".NS") + ".NS" for s in text.replace(",", " ").split()]
    return list(dict.fromkeys(syms))          # remove duplicates, keep order


# ====================================================================== DOWNLOAD
def _history(t, period, interval):
    """Candles from Yahoo (retries on timeout); None = no data."""
    for i in range(CFG["retries"] + 1):
        try:
            df = yf.Ticker(t).history(period=period, interval=interval,
                                      auto_adjust=True, timeout=CFG["timeout"])
            if df.empty:
                return None
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df.dropna(subset=["Close"])
        except Exception:
            if i < CFG["retries"]:
                time.sleep(5 * (2 ** i))
    return None


def pmap(fn, items):
    """fn over items in parallel threads -> {item: result}."""
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        return dict(zip(items, ex.map(fn, items)))


def download(tickers, min_weeks, log):
    """Weekly candles for many tickers. FIRST FILTER: fewer than min_weeks
    candles (5 years for stocks) -> skipped before any condition is checked."""
    out, short, n = {}, [], len(tickers)
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        res = ex.map(lambda t: _history(t, CFG["history"], "1wk"), tickers)
        for i, (t, df) in enumerate(zip(tickers, res), 1):
            if df is not None and len(df) >= min_weeks:
                out[t] = df
            elif df is not None:
                short.append(t.removesuffix(".NS"))
            if i % 100 == 0 or i == n:
                log(f"  downloaded {i}/{n} (ok {len(out)}, <5y skipped {len(short)}, "
                    f"no data {i - len(out) - len(short)})")
    if short:
        log(f"  less than 5 years of data: {', '.join(sorted(short)[:15])}"
            + (" ..." if len(short) > 15 else ""))
    return out


def mcap_cr(t):
    """C1: market cap in crore (1 Cr = 1e7 rupees); NaN if unknown."""
    try:
        return yf.Ticker(t).fast_info["marketCap"] / 1e7
    except Exception:
        return np.nan


def daily_sma(t):
    """F3: latest daily close + SMA (plain average) of the last 50/150/200 daily closes."""
    df = _history(t, "2y", "1d")
    if df is None or len(df) < max(CFG["daily_sma"]):
        return None
    c = df["Close"]
    return {"cmp_daily": c.iloc[-1], **{f"sma{n}d": c.iloc[-n:].mean() for n in CFG["daily_sma"]}}


# ====================================================================== CONDITIONS
def rsi(close, n):
    """Wilder's RSI: average gain / loss use Wilder's EMA (alpha = 1/n)."""
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    val = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]
    return 100.0 if pd.isna(val) else float(val)        # no losses at all -> 100


def evaluate(df, idx):
    """Metrics + True/False checks for one stock. Last row = current week."""
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    N, s, w = CFG["years_weeks"], CFG["rs_short"], CFG["rs_weeks"]
    k, lb = CFG["vol_ratio_weeks"], CFG["red_lookback"]
    ret = lambda x, n: x.iloc[-1] / x.iloc[-1 - n] - 1   # % return over n weeks
    sma40 = c.rolling(CFG["sma_weeks"]).mean()            # SMA of close
    vsma = v.rolling(CFG["vol_sma"]).mean()               # SMA of volume (incl. this week)
    green = c >= o
    red = ~green.iloc[-lb:]

    # 5-year window: ATH = highest weekly High, a = ATH week
    win = df.iloc[-N:]
    wh, wl, wc, wv = (win[x].to_numpy(float) for x in ("High", "Low", "Close", "Volume"))
    a = int(wh.argmax())
    ath = wh[a]

    # F1: uptrend leg = lowest Low before the ATH -> ATH. Each pullback:
    #     (highest High so far - Low) / (highest High so far - leg low)
    p = int(wl[:a + 1].argmin())
    max_rt = np.nan
    if a > p:
        hi = np.maximum.accumulate(wh[p:a])               # highest High before each week
        leg = hi - wl[p]
        ok = leg >= CFG["fib_min_leg"] * wl[p]            # ignore noise at leg start
        max_rt = ((hi - wl[p + 1:a + 1]) / leg)[ok].max() if ok.any() else 0.0

    corr_low = wl[a:].min()                               # F2: bottom after the ATH
    tp, vol = (wh[a:] + wl[a:] + wc[a:]) / 3, wv[a:]      # F7: typical price, volume
    rng = h.iloc[-1] - l.iloc[-1]
    up_v, dn_v = v.iloc[-k:][green.iloc[-k:]].sum(), v.iloc[-k:][~green.iloc[-k:]].sum()

    m = {
        "close": c.iloc[-1], "week": f"{df.index[-1]:%Y-%m-%d}",
        "ath_week": f"{win.index[a]:%Y-%m-%d}",
        "ret_5y": c.iloc[-1] / c.iloc[-N] - 1,                        # C2
        "sma40_rising": sma40.iloc[-1] > sma40.iloc[-1 - CFG["slope_weeks"]],
        "ath_dist": (ath - c.iloc[-1]) / ath,                         # C3, F4
        "rs10w_n500": ret(c, s) - ret(idx["nifty500"], s),            # C4
        "rs11w_n50": ret(c, w) - ret(idx["nifty50"], w),              # C5
        "rs11w_n500": ret(c, w) - ret(idx["nifty500"], w),            # C6
        "rs11w_mid": ret(c, w) - ret(idx["midcap"], w),               # C7
        "rs11w_small": ret(c, w) - ret(idx["smallcap"], w),           # C8
        "vol_vs_sma": v.iloc[-1] / vsma.iloc[-1],                     # C9
        "red_vol_max": (v.iloc[-lb:][red] / vsma.iloc[-lb:][red]).max() if red.any() else 0.0,  # C10
        "upper_wick": (h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])) / rng if rng > 0 else 1.0,    # C11
        "vol_ratio": up_v / dn_v if dn_v else np.inf,                 # C12 green / red volume
        "rsi14w": rsi(c, CFG["rsi_period"]),                          # C13
        "uptrend_max_retrace": max_rt,                                # F1
        "fib_level": (c.iloc[-1] - corr_low) / (ath - corr_low) if ath > corr_low else 1.0,  # F2
        "avwap_ath": (tp * vol).sum() / vol.sum() if vol.sum() else tp.mean(),               # F7
        # trading levels (info only, not used in checks)
        "ath": ath, "swing_low": corr_low, "sma40w": sma40.iloc[-1],
        "fib382_price": corr_low + CFG["fib_min"] * (ath - corr_low),
        "upside_ath": ath / c.iloc[-1] - 1,
    }
    lo, hi_ = CFG["ath_dist"]
    ck = {
        "C2": c.iloc[-1] > sma40.iloc[-1] and m["sma40_rising"] and m["ret_5y"] > 0,
        "C3": lo <= m["ath_dist"] <= hi_,
        "C4": m["rs10w_n500"] > 0, "C5": m["rs11w_n50"] > 0, "C6": m["rs11w_n500"] > 0,
        "C7": m["rs11w_mid"] > 0, "C8": m["rs11w_small"] > 0,
        "C9": m["vol_vs_sma"] > 1,
        "C10": m["red_vol_max"] < CFG["red_factor"],
        "C11": m["upper_wick"] < CFG["wick_max"],
        "C12": m["vol_ratio"] > 1,
        "C13": m["rsi14w"] > CFG["rsi_min"],
        "F1": m["uptrend_max_retrace"] <= CFG["fib_max_retrace"] and m["sma40_rising"],
        "F2": m["fib_level"] > CFG["fib_min"],
        "F4": m["ath_dist"] > CFG["ath_min_dist"],
        "F7": m["close"] > m["avwap_ath"],
    }
    return m, {key: bool(x) for key, x in ck.items()}   # NaN comparisons -> False


# ====================================================================== SCAN
def run_scan(symbols, names=None, log=print, group=FINAL, check_all=False):
    """Returns (shortlist, all_results). check_all=True fetches market cap and
    daily SMAs for every stock (manual mode), not only for possible passers."""
    names = names or {}
    log("Downloading index data...")
    raw = download([t for ts in INDICES.values() for t in ts], 60, lambda *_: None)
    idx = {}
    for key, ts in INDICES.items():
        t = next((t for t in ts if t in raw), None)
        if t is None:
            raise RuntimeError(
                f"No Yahoo Finance history for {key} (tried {', '.join(ts)}). "
                "Yahoo Finance may be temporarily rate-limiting requests; wait a few minutes and retry."
            )
        idx[key] = raw[t]["Close"]

    log(f"Downloading weekly data for {len(symbols)} stocks (< 5 years of data = skipped)...")
    rows = []
    for t, df in download(symbols, CFG["years_weeks"], log).items():
        try:
            m, ck = evaluate(df, idx)
        except Exception as e:
            log(f"  {t}: error {e}")
            continue
        rows.append({"symbol": t.removesuffix(".NS"), "company": names.get(t, ""), **m, "ck": ck})

    # C1 market cap and F3 daily SMAs are slow -> only for stocks that can still pass
    final, keys = group == FINAL, GROUPS[group]

    def alive(ck):
        if check_all:
            return True
        if final:
            return (all(ck[x] for x in ("C4", "C9", "C11", "C12", "C13", "F1", "F2", "F4", "F7"))
                    and sum(not ck[x] for x in GROUPS["Multi RS"]) <= CFG["multirs_max_miss"])
        return sum(not ck[x] for x in keys if x != "C1") <= 1

    cand = [r["symbol"] + ".NS" for r in rows if alive(r["ck"])]
    log(f"Fetching market cap{' + daily SMAs' if final or check_all else ''} "
        f"for {len(cand)} candidates...")
    mcap = pmap(mcap_cr, cand)
    dsma = pmap(daily_sma, cand) if final or check_all else {}

    score = lambda ck, g: f"{sum(bool(ck[x]) for x in GROUPS[g])}/{len(GROUPS[g])}"
    for r in rows:
        t, ck = r["symbol"] + ".NS", r.pop("ck")
        d = dsma.get(t) or {}
        r.update(d, mcap_cr=mcap.get(t, np.nan))
        ck["C1"] = bool(r["mcap_cr"] > CFG["min_mcap_cr"])           # unknown (NaN) -> fail
        ck["F3"] = (all(d["cmp_daily"] > d[f"sma{n}d"] for n in CFG["daily_sma"])
                    if d else None)                                   # None = not checked
        ck["F5"] = all(ck[x] for x in GROUPS["ATH Dashboard"])
        ck["F6"] = sum(not ck[x] for x in GROUPS["Multi RS"]) <= CFG["multirs_max_miss"]
        failed = [x for x in keys if not ck[x]]
        r.update({"Final": score(ck, FINAL), "ATH Dashboard": score(ck, "ATH Dashboard"),
                  "Multi RS": score(ck, "Multi RS"), "n_failed": len(failed),
                  "failed": ", ".join(f"{x} {LABELS[x]}" + (" (not checked)" if ck[x] is None else "")
                                      for x in failed)})
        r.update({x: "-" if ck[x] is None else "YN"[not ck[x]] for x in LABELS})

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        empty = pd.DataFrame(columns=["symbol", "status", "n_failed"])
        return empty, empty
    sl = all_df[all_df["n_failed"] <= (0 if final else 1)].copy()
    sl.insert(2, "status", np.where(sl["n_failed"] == 0, "PASS", "MISS 1"))
    sl = sl.sort_values(["n_failed", "rsi14w"], ascending=[True, False], ignore_index=True)
    log(f"[{group}] scanned {len(all_df)} | PASS {(sl['n_failed'] == 0).sum()}"
        + ("" if final else f" | MISS 1 {(sl['n_failed'] == 1).sum()}"))
    return sl, all_df


VIEW = {"symbol": "Symbol", "company": "Company", "status": "Signal", "failed": "Missed",
        "Final": "Final", "ATH Dashboard": "ATH Dash", "Multi RS": "Multi RS",
        "close": "CMP", "ath": "ATH", "ath_dist": "Below ATH %", "upside_ath": "Upside to ATH %",
        "fib382_price": "0.382 Fib", "avwap_ath": "AVWAP (ATH)", "sma40w": "40W SMA",
        "swing_low": "Swing low", "fib_level": "Fib level", "rsi14w": "RSI 14W",
        "vol_vs_sma": "Vol / 20W SMA", "mcap_cr": "MCap (Cr)"}
PCT = ["Below ATH %", "Upside to ATH %"]


def pretty(df):
    """Readable trading table for screens / reports."""
    t = df.reindex(columns=list(VIEW)).rename(columns=VIEW)
    t[PCT] = t[PCT].astype(float) * 100
    return t


# ====================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description="Check Entry - Preconditions")
    ap.add_argument("--index", choices=INDEX_LISTS, help='e.g. "Nifty 500" (default: all NSE)')
    ap.add_argument("--symbols", nargs="*", help="e.g. HAL TCS")
    ap.add_argument("--group", choices=GROUPS, default=FINAL)
    ap.add_argument("--limit", type=int, help="first N stocks only (quick test)")
    a, _ = ap.parse_known_args()

    names = {} if a.symbols else nse_symbols(a.index)
    syms = parse_manual(" ".join(a.symbols)) if a.symbols else list(names)[:a.limit]
    sl, all_df = run_scan(syms, names, group=a.group, check_all=bool(a.symbols))
    sl.to_csv("shortlist.csv", index=False)
    all_df.to_csv("all_results.csv", index=False)
    print(f"\n{a.group}: {len(sl)} stock(s)")
    print(pretty(sl).round(2).to_string(index=False) if len(sl) else "none")
    print("\n" + "\n".join(f"{x:>4}  {LABELS[x]}" for x in GROUPS[a.group]))
    print("\nSaved: shortlist.csv, all_results.csv")


if __name__ == "__main__":
    main()
