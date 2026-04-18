"""
日米業種リードラグ投資戦略 - 毎日更新シグナル生成スクリプト
論文: 部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略

使い方:
  pip install yfinance numpy pandas
  python lead_lag_signal.py

実行すると signal_dashboard.html が生成・更新されます。
"""

import sys
import json
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# ── パラメータ ────────────────────────────────────────────
WINDOW   = 60      # ローリングウィンドウ（営業日）
K        = 3       # 抽出する主成分数
LAMBDA   = 0.9     # 正則化強度
Q        = 0.3     # ロング/ショートの分位点
LOOKBACK = 300     # データ取得日数（余裕を持って多めに）

# 米国セクターETF
US_TICKERS = {
    "XLB":  "Materials（素材）",
    "XLC":  "Communication（通信）",
    "XLE":  "Energy（エネルギー）",
    "XLF":  "Financials（金融）",
    "XLI":  "Industrials（資本財）",
    "XLK":  "Technology（情報技術）",
    "XLP":  "Consumer Staples（生活必需品）",
    "XLRE": "Real Estate（不動産）",
    "XLU":  "Utilities（公益事業）",
    "XLV":  "Health Care（ヘルスケア）",
    "XLY":  "Consumer Disc.（一般消費財）",
}

# 日本セクターETF（TOPIX-17）
JP_TICKERS = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービス他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}

# シクリカル/ディフェンシブ分類
US_CYCLICAL    = {"XLB", "XLE", "XLF", "XLRE"}
US_DEFENSIVE   = {"XLK", "XLP", "XLU", "XLV"}
JP_CYCLICAL    = {"1618.T", "1625.T", "1629.T", "1631.T"}
JP_DEFENSIVE   = {"1617.T", "1621.T", "1627.T", "1630.T"}


def fetch_data():
    """yfinance で日次終値を取得"""
    print("データ取得中...")
    all_tickers = list(US_TICKERS.keys()) + list(JP_TICKERS.keys())
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK * 2)  # 余裕を持って取得

    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"]

    # 共通営業日のみ（NaN行を除外）
    raw = raw.dropna(how="all")
    # 各銘柄で十分なデータがある列のみ残す
    raw = raw.dropna(axis=1, thresh=WINDOW + 10)
    return raw


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Close-to-Close リターン"""
    return prices.pct_change().dropna()


def gram_schmidt(vectors: list) -> np.ndarray:
    """グラム・シュミット直交化"""
    result = []
    for v in vectors:
        v = v.copy().astype(float)
        for u in result:
            v -= np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            result.append(v / norm)
    return np.column_stack(result)


def build_prior_subspace(us_tickers: list, jp_tickers: list) -> np.ndarray:
    """事前部分空間 V0 を構築 (グローバル, 国スプレッド, シクリカル/ディフェンシブ)"""
    nu = len(us_tickers)
    nj = len(jp_tickers)
    n  = nu + nj

    # v1: グローバルファクター（等ウェイト）
    v1 = np.ones(n) / math.sqrt(n)

    # v2: 国スプレッド（米国+, 日本-）
    v2_raw = np.array([1.0] * nu + [-1.0] * nj)

    # v3: シクリカル/ディフェンシブ
    v3_raw = np.zeros(n)
    for i, tk in enumerate(us_tickers):
        if tk in US_CYCLICAL:
            v3_raw[i] = 1.0
        elif tk in US_DEFENSIVE:
            v3_raw[i] = -1.0
    for j, tk in enumerate(jp_tickers):
        idx = nu + j
        if tk in JP_CYCLICAL:
            v3_raw[idx] = 1.0
        elif tk in JP_DEFENSIVE:
            v3_raw[idx] = -1.0

    V0 = gram_schmidt([v1, v2_raw, v3_raw])
    return V0  # shape (N, 3)


def build_prior_corr(V0: np.ndarray, C_full: np.ndarray) -> np.ndarray:
    """事前相関行列 C0 を構築"""
    D0 = np.diag(V0.T @ C_full @ V0)           # (K,)
    C0_raw = V0 @ np.diag(D0) @ V0.T           # (N, N)
    # 対角要素で正規化
    d = np.sqrt(np.diag(C0_raw))
    d[d < 1e-10] = 1.0
    C0 = C0_raw / np.outer(d, d)
    np.fill_diagonal(C0, 1.0)
    return C0


def compute_signal(returns: pd.DataFrame, us_tickers: list, jp_tickers: list):
    """
    最新日のシグナルを計算

    Returns
    -------
    signals       : dict {jp_ticker: signal_value}
    factor_scores : dict {name: score}
    factor_evals  : dict {name: eigenvalue}
    us_rets_today : dict {us_ticker: return}
    signal_date   : str
    """
    all_tickers = us_tickers + jp_tickers
    # 使用可能な列のみ絞り込む
    available = [t for t in all_tickers if t in returns.columns]
    ret = returns[available].copy()

    # 最低 WINDOW+2 行必要
    if len(ret) < WINDOW + 2:
        raise ValueError(f"データが不足しています（{len(ret)}行）")

    # 最新の完全な行を特定（日本株は欠損しやすい）
    jp_available = [t for t in jp_tickers if t in ret.columns]
    us_available = [t for t in us_tickers if t in ret.columns]

    # 最新日: 米国終値が確定している最後の行
    last_idx = len(ret) - 1
    # 米国データが揃っている最後の行を探す
    for i in range(last_idx, last_idx - 5, -1):
        if ret[us_available].iloc[i].notna().all():
            last_idx = i
            break

    signal_date = ret.index[last_idx]
    window_ret  = ret.iloc[max(0, last_idx - WINDOW):last_idx]

    # ウィンドウ内に NaN がある列を除外
    window_ret = window_ret.dropna(axis=1)
    us_in_win  = [t for t in us_available if t in window_ret.columns]
    jp_in_win  = [t for t in jp_available if t in window_ret.columns]
    all_in_win = us_in_win + jp_in_win
    window_ret = window_ret[all_in_win]

    nu = len(us_in_win)
    nj = len(jp_in_win)

    # 標準化リターン行列
    mu    = window_ret.mean()
    sigma = window_ret.std()
    sigma[sigma < 1e-10] = 1.0
    Z = (window_ret - mu) / sigma  # (L, N)

    # 相関行列 Ct
    Ct = Z.T.values @ Z.values / len(Z)

    # 事前部分空間
    V0    = build_prior_subspace(us_in_win, jp_in_win)
    C0    = build_prior_corr(V0, Ct)
    C_reg = (1 - LAMBDA) * Ct + LAMBDA * C0

    # 固有分解
    eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
    # 降順に並び替え
    idx  = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    Vk   = eigenvectors[:, :K]           # (N, K)
    Vk_U = Vk[:nu, :]                    # (nu, K)
    Vk_J = Vk[nu:, :]                    # (nj, K)

    # 当日米国リターンを標準化
    us_ret_today = ret[us_in_win].iloc[last_idx]
    z_U = ((us_ret_today - mu[us_in_win]) / sigma[us_in_win]).values  # (nu,)

    # ファクタースコア
    f = Vk_U.T @ z_U   # (K,)

    # 日本側シグナル
    z_J_hat = Vk_J @ f  # (nj,)

    signals = {tk: float(v) for tk, v in zip(jp_in_win, z_J_hat)}

    factor_names  = ["グローバル", "国スプレッド", "景気敏感/ディフェンシブ"]
    factor_scores = {factor_names[i]: float(f[i]) for i in range(K)}
    factor_evals  = {factor_names[i]: float(eigenvalues[i]) for i in range(K)}

    us_rets = {tk: float(ret[tk].iloc[last_idx]) for tk in us_in_win}

    date_str = signal_date.strftime("%Y-%m-%d") if hasattr(signal_date, "strftime") else str(signal_date)[:10]
    return signals, factor_scores, factor_evals, us_rets, date_str


def make_strength(sig_val: float, max_val: float) -> str:
    """シグナル強度を●で表現"""
    if max_val < 1e-10:
        return "○○○○○"
    ratio = abs(sig_val) / max_val
    filled = round(ratio * 5)
    return "●" * filled + "○" * (5 - filled)


def build_html(signals, factor_scores, factor_evals, us_rets, signal_date):
    """HTMLダッシュボードを生成"""
    # ロング・ショート分類
    sorted_sigs = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_sigs)
    n_long  = max(1, round(n * Q))
    n_short = max(1, round(n * Q))

    long_set  = {tk for tk, _ in sorted_sigs[:n_long]}
    short_set = {tk for tk, _ in sorted_sigs[-n_short:]}
    max_val   = max(abs(v) for _, v in sorted_sigs) if sorted_sigs else 1.0

    # 米国リターン HTML
    us_html = ""
    for tk, ret_val in sorted(us_rets.items(), key=lambda x: x[1], reverse=True):
        color = "#22c55e" if ret_val >= 0 else "#ef4444"
        arrow = "▲" if ret_val >= 0 else "▼"
        nm    = US_TICKERS.get(tk, tk)
        us_html += (
            f'<div class="us-card">'
            f'<span class="us-tk">{tk}</span>'
            f'<span class="us-nm">{nm}</span>'
            f'<span class="us-rt" style="color:{color}">{arrow} {ret_val:+.2%}</span>'
            f'</div>\n'
        )

    # ファクタースコア HTML
    f_html = ""
    for fname, fscore in factor_scores.items():
        feval = factor_evals.get(fname, 0.0)
        color = "#22c55e" if fscore >= 0 else "#ef4444"
        f_html += (
            f'<div class="f-card">'
            f'<div class="f-nm">{fname}</div>'
            f'<div class="f-sc" style="color:{color}">{fscore:+.3f}</div>'
            f'<div class="f-ev">固有値 {feval:.2f}</div>'
            f'</div>\n'
        )

    # ロング HTML
    long_html = ""
    for tk, sig in sorted_sigs[:n_long]:
        nm  = JP_TICKERS.get(tk, tk)
        str_= make_strength(sig, max_val)
        long_html += (
            f'<div class="card long">'
            f'<div class="card-head"><span class="tk">{tk}</span><span class="badge bg-long">買い</span></div>'
            f'<div class="nm">{nm}</div>'
            f'<div class="sig">シグナル {sig:+.4f}</div>'
            f'<div class="str">強度 {str_}</div>'
            f'</div>\n'
        )

    # ショート HTML
    short_html = ""
    for tk, sig in sorted_sigs[-n_short:][::-1]:
        nm  = JP_TICKERS.get(tk, tk)
        str_= make_strength(sig, max_val)
        short_html += (
            f'<div class="card short">'
            f'<div class="card-head"><span class="tk">{tk}</span><span class="badge bg-short">売り</span></div>'
            f'<div class="nm">{nm}</div>'
            f'<div class="sig">シグナル {sig:+.4f}</div>'
            f'<div class="str">強度 {str_}</div>'
            f'</div>\n'
        )

    # ニュートラル HTML
    neutral_html = ""
    neutral_list = [(tk, sig) for tk, sig in sorted_sigs
                    if tk not in long_set and tk not in short_set]
    for tk, sig in neutral_list:
        nm    = JP_TICKERS.get(tk, tk)
        color = "#22c55e" if sig >= 0 else "#ef4444"
        neutral_html += (
            f'<div class="n-row">'
            f'<span class="n-tk">{tk}</span>'
            f'<span class="n-nm">{nm}</span>'
            f'<span class="n-sig" style="color:{color}">{sig:+.4f}</span>'
            f'</div>\n'
        )

    # チャート用 JSON
    chart_data = json.dumps(
        [{"tk": tk, "nm": JP_TICKERS.get(tk, tk), "s": round(sig, 4)}
         for tk, sig in sorted_sigs],
        ensure_ascii=False
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>日米業種リードラグ - PCA_SUB シグナル</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Hiragino Sans",sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px}}
.hd{{text-align:center;padding:28px 0 18px;border-bottom:1px solid #1e293b;margin-bottom:22px}}
.hd h1{{font-size:1.5em;color:#f8fafc}}.hd .sub{{color:#94a3b8;font-size:.85em;margin-top:4px}}
.hd .dt{{color:#60a5fa;font-size:1.05em;font-weight:600;margin-top:8px}}
.hd .pm{{color:#64748b;font-size:.75em;margin-top:2px}}
.hd .upd{{color:#34d399;font-size:.75em;margin-top:4px}}
.sec{{margin-bottom:26px}}.sec-t{{font-size:1.05em;color:#cbd5e1;margin-bottom:12px;padding-left:10px;border-left:3px solid #3b82f6}}
.us-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:6px}}
.us-card{{background:#1e293b;border-radius:8px;padding:9px 11px;display:flex;flex-direction:column;gap:1px}}
.us-tk{{font-weight:700;color:#93c5fd;font-size:.82em}}.us-nm{{color:#64748b;font-size:.68em}}.us-rt{{font-size:.95em;font-weight:600;margin-top:3px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}}
.card{{border-radius:10px;padding:14px}}
.card.long{{background:linear-gradient(135deg,#064e3b,#0f2a1f);border:1px solid #065f46}}
.card.short{{background:linear-gradient(135deg,#4c0519,#1f0a0e);border:1px solid #881337}}
.card-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}}
.tk{{font-weight:700;font-size:1.05em}}.badge{{font-size:.68em;padding:2px 8px;border-radius:12px;font-weight:600}}
.bg-long{{background:#059669;color:#d1fae5}}.bg-short{{background:#be123c;color:#ffe4e6}}
.nm{{color:#94a3b8;font-size:.82em}}.sig{{margin-top:7px;font-family:monospace;font-size:.88em}}
.str{{color:#fbbf24;margin-top:3px;font-size:.82em}}
.n-list{{display:flex;flex-direction:column;gap:3px}}
.n-row{{display:flex;align-items:center;gap:8px;background:#1e293b;padding:7px 11px;border-radius:6px;font-size:.82em}}
.n-tk{{font-weight:600;color:#93c5fd;min-width:52px}}.n-nm{{color:#94a3b8;flex:1}}.n-sig{{font-family:monospace;font-weight:600}}
.f-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.f-card{{background:#1e293b;border-radius:8px;padding:13px;text-align:center}}
.f-nm{{color:#93c5fd;font-size:.82em;font-weight:600}}.f-sc{{font-size:1.1em;font-weight:700;margin-top:5px}}.f-ev{{color:#64748b;font-size:.72em;margin-top:3px}}
.chart-box{{background:#1e293b;border-radius:10px;padding:16px;overflow-x:auto}}
.disc{{margin-top:28px;padding:14px;background:#1e293b;border-radius:8px;border-left:3px solid #f59e0b;font-size:.72em;color:#94a3b8;line-height:1.6}}
@media(max-width:768px){{.us-grid{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}.f-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="wrap">
<div class="hd">
  <h1>日米業種リードラグ投資戦略</h1>
  <div class="sub">部分空間正則化付き PCA (PCA_SUB) シグナル</div>
  <div class="dt">{signal_date}　米国終値 → 翌営業日 日本市場</div>
  <div class="pm">Window={WINDOW} &nbsp; K={K} &nbsp; λ={LAMBDA} &nbsp; q={Q}</div>
  <div class="upd">最終更新: {now_str}</div>
</div>
<div class="sec"><div class="sec-t">米国セクター ETF 当日リターン</div><div class="us-grid">
{us_html}</div></div>
<div class="sec"><div class="sec-t">共通ファクタースコア</div><div class="f-grid">
{f_html}</div></div>
<div class="sec"><div class="sec-t">買い候補（ロング）— 上位 {int(Q*100)}%</div><div class="grid">
{long_html}</div></div>
<div class="sec"><div class="sec-t">売り候補（ショート）— 下位 {int(Q*100)}%</div><div class="grid">
{short_html}</div></div>
<div class="sec"><div class="sec-t">ニュートラル</div><div class="n-list">
{neutral_html}</div></div>
<div class="sec"><div class="sec-t">全業種シグナル分布</div>
<div class="chart-box"><canvas id="cv"></canvas></div></div>
<div class="disc"><strong>⚠ 注意:</strong> 本ツールは学術論文の手法を実装した研究・教育目的のプロトタイプです。投資助言や推奨を行うものではありません。実際の投資判断はご自身の責任で行ってください。シグナルは米国市場の終値確定後（日本時間 6:00 頃）に有効となり、同日の日本市場 Open→Close の方向を予測するものです。過去のパフォーマンスは将来の成果を保証しません。</div>
</div>
<script>
const D={chart_data};
const cv=document.getElementById("cv"),cx=cv.getContext("2d");
cv.width=cv.parentElement.offsetWidth-32;cv.height=280;
const W=cv.width,H=cv.height,P={{t:28,b:58,l:48,r:16}};
const cW=W-P.l-P.r,cH=H-P.t-P.b;
const mx=Math.max(...D.map(d=>Math.abs(d.s)))*1.15||1;
const zY=P.t+cH/2,bW=Math.min(34,(cW/D.length)-4);
cx.fillStyle="#1e293b";cx.fillRect(0,0,W,H);
cx.strokeStyle="#475569";cx.lineWidth=1;cx.beginPath();cx.moveTo(P.l,zY);cx.lineTo(W-P.r,zY);cx.stroke();
D.forEach((d,i)=>{{const x=P.l+(cW/D.length)*i+(cW/D.length-bW)/2;const h=(d.s/mx)*(cH/2);
cx.fillStyle=d.s>=0?"#22c55e":"#ef4444";
if(d.s>=0)cx.fillRect(x,zY-h,bW,h);else cx.fillRect(x,zY,bW,-h);
cx.save();cx.translate(x+bW/2,H-4);cx.rotate(-Math.PI/4);
cx.fillStyle="#94a3b8";cx.font="10px sans-serif";cx.textAlign="right";cx.fillText(d.nm,0,0);cx.restore();
cx.fillStyle="#cbd5e1";cx.font="9px monospace";cx.textAlign="center";
const vY=d.s>=0?zY-h-5:zY-h+12;cx.fillText(d.s.toFixed(3),x+bW/2,vY);}});
cx.fillStyle="#64748b";cx.font="10px sans-serif";cx.textAlign="right";
cx.fillText("0",P.l-4,zY+4);cx.fillText("+"+mx.toFixed(2),P.l-4,P.t+10);cx.fillText("-"+mx.toFixed(2),P.l-4,H-P.b-2);
</script></body></html>"""
    return html


def main():
    output_path = "signal_dashboard.html"

    try:
        prices = fetch_data()
    except Exception as e:
        print(f"データ取得エラー: {e}")
        sys.exit(1)

    returns = compute_returns(prices)

    us_tickers = [t for t in US_TICKERS if t in returns.columns]
    jp_tickers = [t for t in JP_TICKERS if t in returns.columns]

    print(f"  米国ETF: {len(us_tickers)}銘柄, 日本ETF: {len(jp_tickers)}銘柄")

    try:
        signals, factor_scores, factor_evals, us_rets, signal_date = compute_signal(
            returns, us_tickers, jp_tickers
        )
    except Exception as e:
        print(f"シグナル計算エラー: {e}")
        sys.exit(1)

    html = build_html(signals, factor_scores, factor_evals, us_rets, signal_date)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[完了] {output_path} を更新しました")
    print(f"  シグナル日付: {signal_date}")
    print(f"\n  ロング候補:")
    sorted_sigs = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    n_long = max(1, round(len(sorted_sigs) * Q))
    for tk, sig in sorted_sigs[:n_long]:
        print(f"    {tk} ({JP_TICKERS.get(tk, '')})  {sig:+.4f}")
    print(f"\n  ショート候補:")
    for tk, sig in sorted_sigs[-n_long:][::-1]:
        print(f"    {tk} ({JP_TICKERS.get(tk, '')})  {sig:+.4f}")


if __name__ == "__main__":
    main()
