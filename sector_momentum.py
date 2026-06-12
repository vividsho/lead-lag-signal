"""
日本セクター 中期モメンタム・ダッシュボード生成スクリプト

直近1・3・6ヶ月の値動きから「勢いのあるセクター」を順位付けし、
その中の代表銘柄を提案します。週末に1回チェックして月曜に注文、
見直しは月1回程度、というゆっくりした使い方を想定しています。

使い方:
  pip install yfinance numpy pandas
  python sector_momentum.py

実行すると signal_dashboard.html が生成・更新されます。
"""

import sys
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# ── パラメータ ────────────────────────────────────────────
M1, M3, M6  = 20, 60, 120  # 1・3・6ヶ月に相当する営業日数
MA_DAYS     = 65           # トレンド判定の移動平均（約13週）
TOP_Q       = 0.3          # 「強い/弱い」とみなす割合（上位・下位30%）
STOCK_TOP_N = 5            # 各セクターで提案する銘柄数
FETCH_DAYS  = 450          # データ取得期間（暦日。6ヶ月+移動平均分の余裕込み）

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

# 各セクターの代表銘柄（東証プライムの大型・流動性の高い銘柄から選定）
JP_STOCKS = {
    "1617.T": [  # 食品
        ("2914.T", "JT（日本たばこ産業）"),
        ("2802.T", "味の素"),
        ("2502.T", "アサヒグループHD"),
        ("2503.T", "キリンHD"),
        ("2269.T", "明治HD"),
        ("2801.T", "キッコーマン"),
        ("2587.T", "サントリー食品"),
        ("2282.T", "日本ハム"),
    ],
    "1618.T": [  # エネルギー資源
        ("1605.T", "INPEX"),
        ("5020.T", "ENEOSホールディングス"),
        ("5019.T", "出光興産"),
        ("5021.T", "コスモエネルギーHD"),
        ("1662.T", "石油資源開発"),
    ],
    "1619.T": [  # 建設・資材
        ("1925.T", "大和ハウス工業"),
        ("1928.T", "積水ハウス"),
        ("1812.T", "鹿島建設"),
        ("1801.T", "大成建設"),
        ("1802.T", "大林組"),
        ("1803.T", "清水建設"),
        ("5201.T", "AGC"),
        ("5233.T", "太平洋セメント"),
    ],
    "1620.T": [  # 素材・化学
        ("4063.T", "信越化学工業"),
        ("4452.T", "花王"),
        ("4901.T", "富士フイルムHD"),
        ("4188.T", "三菱ケミカルグループ"),
        ("3407.T", "旭化成"),
        ("4911.T", "資生堂"),
        ("4005.T", "住友化学"),
        ("6988.T", "日東電工"),
    ],
    "1621.T": [  # 医薬品
        ("4502.T", "武田薬品工業"),
        ("4568.T", "第一三共"),
        ("4519.T", "中外製薬"),
        ("4503.T", "アステラス製薬"),
        ("4523.T", "エーザイ"),
        ("4578.T", "大塚HD"),
        ("4507.T", "塩野義製薬"),
        ("4151.T", "協和キリン"),
    ],
    "1622.T": [  # 自動車・輸送機
        ("7203.T", "トヨタ自動車"),
        ("7267.T", "ホンダ"),
        ("6902.T", "デンソー"),
        ("7269.T", "スズキ"),
        ("5108.T", "ブリヂストン"),
        ("7270.T", "SUBARU"),
        ("7272.T", "ヤマハ発動機"),
        ("7259.T", "アイシン"),
    ],
    "1623.T": [  # 鉄鋼・非鉄
        ("5401.T", "日本製鉄"),
        ("5411.T", "JFEホールディングス"),
        ("5713.T", "住友金属鉱山"),
        ("5802.T", "住友電気工業"),
        ("5801.T", "古河電気工業"),
        ("5711.T", "三菱マテリアル"),
        ("5406.T", "神戸製鋼所"),
    ],
    "1624.T": [  # 機械
        ("6301.T", "小松製作所（コマツ）"),
        ("6367.T", "ダイキン工業"),
        ("6273.T", "SMC"),
        ("6326.T", "クボタ"),
        ("7011.T", "三菱重工業"),
        ("7013.T", "IHI"),
        ("6361.T", "荏原製作所"),
        ("6113.T", "アマダ"),
    ],
    "1625.T": [  # 電機・精密
        ("8035.T", "東京エレクトロン"),
        ("6758.T", "ソニーグループ"),
        ("6501.T", "日立製作所"),
        ("6861.T", "キーエンス"),
        ("6981.T", "村田製作所"),
        ("6857.T", "アドバンテスト"),
        ("6954.T", "ファナック"),
        ("7741.T", "HOYA"),
        ("6594.T", "ニデック"),
        ("7751.T", "キヤノン"),
    ],
    "1626.T": [  # 情報通信・サービス他
        ("9432.T", "NTT"),
        ("9433.T", "KDDI"),
        ("9984.T", "ソフトバンクグループ"),
        ("9434.T", "ソフトバンク"),
        ("6098.T", "リクルートHD"),
        ("7974.T", "任天堂"),
        ("4661.T", "オリエンタルランド"),
        ("4307.T", "野村総合研究所"),
    ],
    "1627.T": [  # 電力・ガス
        ("9501.T", "東京電力HD"),
        ("9503.T", "関西電力"),
        ("9502.T", "中部電力"),
        ("9531.T", "東京ガス"),
        ("9532.T", "大阪ガス"),
        ("9508.T", "九州電力"),
    ],
    "1628.T": [  # 運輸・物流
        ("9022.T", "JR東海"),
        ("9020.T", "JR東日本"),
        ("9021.T", "JR西日本"),
        ("9101.T", "日本郵船"),
        ("9104.T", "商船三井"),
        ("9107.T", "川崎汽船"),
        ("9201.T", "日本航空（JAL）"),
        ("9202.T", "ANAホールディングス"),
        ("9064.T", "ヤマトHD"),
    ],
    "1629.T": [  # 商社・卸売
        ("8058.T", "三菱商事"),
        ("8031.T", "三井物産"),
        ("8001.T", "伊藤忠商事"),
        ("8053.T", "住友商事"),
        ("8002.T", "丸紅"),
        ("8015.T", "豊田通商"),
        ("2768.T", "双日"),
    ],
    "1630.T": [  # 小売
        ("9983.T", "ファーストリテイリング"),
        ("3382.T", "セブン&アイHD"),
        ("8267.T", "イオン"),
        ("9843.T", "ニトリHD"),
        ("7532.T", "パン・パシフィックHD"),
        ("3092.T", "ZOZO"),
        ("3088.T", "マツキヨココカラ"),
    ],
    "1631.T": [  # 銀行
        ("8306.T", "三菱UFJフィナンシャルG"),
        ("8316.T", "三井住友フィナンシャルG"),
        ("8411.T", "みずほフィナンシャルG"),
        ("8308.T", "りそなHD"),
        ("7186.T", "コンコルディアFG"),
        ("8331.T", "千葉銀行"),
    ],
    "1632.T": [  # 金融（除く銀行）
        ("8766.T", "東京海上HD"),
        ("8750.T", "第一生命HD"),
        ("8725.T", "MS&ADインシュアランスG"),
        ("8630.T", "SOMPOホールディングス"),
        ("8604.T", "野村HD"),
        ("8601.T", "大和証券グループ本社"),
        ("8591.T", "オリックス"),
        ("8697.T", "日本取引所グループ"),
    ],
    "1633.T": [  # 不動産
        ("8801.T", "三井不動産"),
        ("8802.T", "三菱地所"),
        ("8830.T", "住友不動産"),
        ("3289.T", "東急不動産HD"),
        ("8804.T", "東京建物"),
        ("3231.T", "野村不動産HD"),
    ],
}


def clean_prices(prices: pd.DataFrame, max_dev: float = 0.3) -> pd.DataFrame:
    """データ取得元の異常値（桁違いの価格など）を除去する。

    前後数日の中央値から ±30% 以上離れた価格は誤データとみなし、
    直前の正常な価格で置き換える。
    （例: 2026-03-30 の 1629.T が 288円 → 0.56円 と記録されていた）
    """
    cleaned = prices.copy()
    for col in cleaned.columns:
        s   = cleaned[col]
        med = s.rolling(11, center=True, min_periods=3).median()
        bad = ((s - med).abs() / med) > max_dev
        if bad.any():
            orig_nan = s.isna()
            fixed = s.mask(bad).ffill()
            fixed[orig_nan] = np.nan
            cleaned[col] = fixed
    return cleaned


def fetch_prices(tickers: list) -> pd.DataFrame:
    """日次終値を取得して異常値を除去"""
    end   = datetime.today()
    start = end - timedelta(days=FETCH_DAYS)
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    return clean_prices(raw.dropna(how="all"))


def momentum_metrics(prices: pd.Series):
    """1本の価格データから勢い指標を計算する。データ不足なら None

    勢いスコア = 1・3・6ヶ月リターンをそれぞれ「月あたり」に換算して平均。
    トレンド   = 終値が13週移動平均より上なら True（上昇トレンド）。
    """
    s = prices.dropna()
    if len(s) < M6 + 5:
        return None
    last = float(s.iloc[-1])
    r1 = last / float(s.iloc[-1 - M1]) - 1
    r3 = last / float(s.iloc[-1 - M3]) - 1
    r6 = last / float(s.iloc[-1 - M6]) - 1
    score   = (r1 + r3 / 3 + r6 / 6) / 3
    uptrend = last > float(s.iloc[-MA_DAYS:].mean())
    return {"r1": r1, "r3": r3, "r6": r6, "score": score,
            "uptrend": uptrend, "price": last}


def rank_sectors(etf_prices: pd.DataFrame):
    """セクターETFを勢いスコアで順位付け"""
    metrics = {}
    for tk in JP_TICKERS:
        if tk in etf_prices.columns:
            m = momentum_metrics(etf_prices[tk])
            if m is not None:
                metrics[tk] = m
    ranking = sorted(metrics.items(), key=lambda x: x[1]["score"], reverse=True)
    n_top = max(1, round(len(ranking) * TOP_Q))
    return ranking, n_top


def analyze_sector_stocks(target_sectors: list):
    """各セクターの代表銘柄を勢いスコアで順位付けする。

    上昇トレンド（13週移動平均より上）の銘柄を優先し、
    勢いスコアの高い順に上位 STOCK_TOP_N 銘柄を返す。

    Returns
    -------
    results  : dict {sector: [銘柄情報, ...]}  セクターごとの上位銘柄
    all_rows : list 全銘柄の情報（全セクター横断のTop5用）
    """
    codes = []
    for sec in target_sectors:
        codes += [code for code, _ in JP_STOCKS.get(sec, [])]
    codes = list(dict.fromkeys(codes))
    if not codes:
        return {}, []

    print(f"個別銘柄データ取得中...（{len(codes)}銘柄）")
    try:
        prices = fetch_prices(codes)
    except Exception as e:
        print(f"  個別銘柄の取得に失敗しました: {e}（セクターのみ表示します）")
        return {}, []

    results  = {}
    all_rows = []
    for sec in target_sectors:
        rows = []
        for code, name in JP_STOCKS.get(sec, []):
            if code not in prices.columns:
                continue
            m = momentum_metrics(prices[code])
            if m is None:
                continue
            m.update({"code": code, "name": name, "sector": sec})
            rows.append(m)
        picked = [r for r in rows if r["uptrend"]]
        if len(picked) < 3:
            picked = rows
        picked.sort(key=lambda r: r["score"], reverse=True)
        results[sec] = picked[:STOCK_TOP_N]
        all_rows += rows
    return results, all_rows


def pick_top_stocks(all_rows: list) -> list:
    """全セクター横断で勢いの強い銘柄 Top5 を選ぶ（上昇トレンド中のみ）"""
    pool = [r for r in all_rows if r["uptrend"]]
    if len(pool) < STOCK_TOP_N:
        pool = all_rows
    return sorted(pool, key=lambda r: r["score"], reverse=True)[:STOCK_TOP_N]


def build_html(ranking, n_top, stock_analysis, top_stocks, date_str):
    """HTMLダッシュボードを生成"""
    top_set  = {tk for tk, _ in ranking[:n_top]}
    weak_set = {tk for tk, _ in ranking[-n_top:]}

    def pct_span(v, hide=True):
        color = "#22c55e" if v >= 0 else "#ef4444"
        cls = "rk-num hide-m" if hide else "rk-num"
        return f'<span class="{cls}" style="color:{color}">{v:+.1%}</span>'

    # 一番上のサマリー（全セクター横断の銘柄 Top5）
    top5_sec = ""
    if top_stocks:
        top5_html = (
            '<div class="t5-row t5-head"><span>#</span><span class="hide-m">コード</span>'
            '<span>銘柄名</span><span class="hide-m">セクター</span>'
            '<span class="t5-num">100株目安</span><span class="t5-num hide-m">3ヶ月</span>'
            '<span class="t5-num">勢い/月</span></div>\n'
        )
        for i, r in enumerate(top_stocks, 1):
            cs = "#22c55e" if r["score"] >= 0 else "#ef4444"
            c3 = "#22c55e" if r["r3"] >= 0 else "#ef4444"
            cost = r["price"] * 100
            cost_str = f"約{cost / 10000:,.1f}万円" if cost >= 100000 else f"約{cost:,.0f}円"
            top5_html += (
                f'<div class="t5-row">'
                f'<span class="t5-no">{i}</span>'
                f'<span class="t5-cd hide-m">{r["code"].replace(".T", "")}</span>'
                f'<span class="t5-nm">{r["name"]}</span>'
                f'<span class="t5-sec hide-m">{JP_TICKERS.get(r["sector"], "")}</span>'
                f'<span class="t5-num">{cost_str}</span>'
                f'<span class="t5-num hide-m" style="color:{c3}">{r["r3"]:+.1%}</span>'
                f'<span class="t5-num" style="color:{cs};font-weight:700">{r["score"]:+.1%}</span>'
                f'</div>\n'
            )
        top5_sec = (
            f'<div class="sec"><div class="sec-t">今週の注目銘柄 Top5（全セクター対象）</div>'
            f'<div class="sec-note">登録している全17業種・約130銘柄の中から、'
            f'上昇トレンド中で勢いの強い順に5銘柄。</div>\n'
            f'{top5_html}</div>\n'
        )

    # セクターランキング HTML
    rank_html = (
        '<div class="rk-row rk-head"><span class="rk-no">#</span><span>セクター</span>'
        '<span class="rk-num">勢い/月</span><span class="rk-num hide-m">1ヶ月</span>'
        '<span class="rk-num hide-m">3ヶ月</span><span class="rk-num hide-m">6ヶ月</span>'
        '<span class="rk-tr">向き</span><span class="rk-bd"></span></div>\n'
    )
    for i, (tk, m) in enumerate(ranking, 1):
        cls = "strong" if tk in top_set else ("weak" if tk in weak_set else "")
        badge = ""
        if tk in top_set:
            badge = '<span class="badge bg-long">強い</span>'
        elif tk in weak_set:
            badge = '<span class="badge bg-short">弱い</span>'
        tr_mark = ('<span style="color:#22c55e">↑</span>' if m["uptrend"]
                   else '<span style="color:#ef4444">↓</span>')
        sc_color = "#22c55e" if m["score"] >= 0 else "#ef4444"
        rank_html += (
            f'<div class="rk-row {cls}">'
            f'<span class="rk-no">{i}</span>'
            f'<span class="rk-nm">{JP_TICKERS.get(tk, tk)}</span>'
            f'<span class="rk-num" style="color:{sc_color};font-weight:700">{m["score"]:+.1%}</span>'
            f'{pct_span(m["r1"])}{pct_span(m["r3"])}{pct_span(m["r6"])}'
            f'<span class="rk-tr">{tr_mark}</span>'
            f'<span class="rk-bd">{badge}</span>'
            f'</div>\n'
        )

    # 強いセクターの個別銘柄 HTML
    sector_score = dict(ranking)

    def stock_block(sec):
        rows_data = stock_analysis.get(sec, [])
        if not rows_data:
            return ""
        rows = (
            '<div class="stk-row stk-head"><span>コード</span><span>銘柄名</span>'
            '<span class="stk-num hide-m">終値</span><span class="stk-num">100株目安</span>'
            '<span class="stk-num hide-m">1ヶ月</span><span class="stk-num">3ヶ月</span>'
            '<span class="stk-num">勢い/月</span><span class="stk-tr">向き</span></div>\n'
        )
        for r in rows_data:
            c1 = "#22c55e" if r["r1"] >= 0 else "#ef4444"
            c3 = "#22c55e" if r["r3"] >= 0 else "#ef4444"
            cs = "#22c55e" if r["score"] >= 0 else "#ef4444"
            tr = ('<span style="color:#22c55e">↑</span>' if r["uptrend"]
                  else '<span style="color:#ef4444">↓</span>')
            cost = r["price"] * 100
            cost_str = f"約{cost / 10000:,.1f}万円" if cost >= 100000 else f"約{cost:,.0f}円"
            rows += (
                f'<div class="stk-row">'
                f'<span class="stk-cd">{r["code"].replace(".T", "")}</span>'
                f'<span class="stk-nm">{r["name"]}</span>'
                f'<span class="stk-num hide-m">{r["price"]:,.0f}円</span>'
                f'<span class="stk-num">{cost_str}</span>'
                f'<span class="stk-num hide-m" style="color:{c1}">{r["r1"]:+.1%}</span>'
                f'<span class="stk-num" style="color:{c3}">{r["r3"]:+.1%}</span>'
                f'<span class="stk-num" style="color:{cs};font-weight:700">{r["score"]:+.1%}</span>'
                f'<span class="stk-tr">{tr}</span>'
                f'</div>\n'
            )
        sec_m = sector_score[sec]
        return (
            f'<div class="stk-block">'
            f'<div class="stk-bh"><span class="sn">{JP_TICKERS.get(sec, sec)}</span>'
            f'<span class="ss">セクター勢い {sec_m["score"]:+.1%}/月</span></div>'
            f'<div class="stk-rows">{rows}</div></div>\n'
        )

    stock_html = "".join(stock_block(tk) for tk, _ in ranking[:n_top])

    # チャート用 JSON（勢いスコアを%表示）
    chart_data = json.dumps(
        [{"nm": JP_TICKERS.get(tk, tk), "s": round(m["score"] * 100, 2)}
         for tk, m in ranking],
        ensure_ascii=False
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>日本セクター 中期モメンタム・ダッシュボード</title>
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
.sec-note{{color:#64748b;font-size:.72em;margin:-6px 0 10px 13px;line-height:1.5}}
.t5-row{{display:grid;grid-template-columns:24px 52px minmax(130px,1fr) 120px 96px 64px 70px;gap:8px;align-items:center;background:linear-gradient(135deg,#064e3b,#0f2a1f);border:1px solid #065f46;border-radius:8px;padding:9px 12px;margin-bottom:5px;font-size:.85em}}
.t5-row.t5-head{{background:none;border:none;color:#64748b;font-size:.7em;font-weight:600;padding:0 12px;margin-bottom:2px}}
.t5-no{{color:#34d399;font-weight:800}}
.t5-cd{{font-weight:700;color:#93c5fd}}
.t5-nm{{font-weight:700;color:#f8fafc}}
.t5-sec{{color:#94a3b8;font-size:.85em}}
.t5-num{{font-family:monospace;text-align:right}}
.t5-row.t5-head .t5-num{{font-family:inherit}}
.guide{{background:#1e293b;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:9px;font-size:.85em;color:#cbd5e1;line-height:1.5}}
.g-no{{display:inline-block;background:#3b82f6;color:#fff;border-radius:50%;min-width:20px;height:20px;text-align:center;line-height:20px;font-size:.8em;margin-right:8px;font-weight:700}}
.badge{{font-size:.68em;padding:2px 8px;border-radius:12px;font-weight:600;white-space:nowrap}}
.bg-long{{background:#059669;color:#d1fae5}}.bg-short{{background:#be123c;color:#ffe4e6}}
.rk-row{{display:grid;grid-template-columns:28px minmax(110px,1fr) 76px 62px 62px 62px 60px 56px;gap:6px;align-items:center;padding:6px 10px;border-radius:6px;font-size:.82em;background:#1e293b;margin-bottom:3px}}
.rk-row.strong{{background:linear-gradient(135deg,#064e3b,#0f2a1f);border:1px solid #065f46}}
.rk-row.weak{{background:linear-gradient(135deg,#4c0519,#1f0a0e);border:1px solid #881337}}
.rk-row.rk-head{{background:none;color:#64748b;font-size:.72em;font-weight:600;margin-bottom:5px}}
.rk-no{{color:#64748b;font-weight:700}}.rk-nm{{color:#e2e8f0;font-weight:600}}
.rk-num{{font-family:monospace;text-align:right}}
.rk-row.rk-head .rk-num{{font-family:inherit}}
.rk-tr{{text-align:center}}.rk-bd{{text-align:right}}
.stk-block{{background:#1e293b;border-radius:10px;padding:13px 14px;margin-bottom:12px;border-left:3px solid #059669}}
.stk-bh{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;flex-wrap:wrap;gap:4px}}
.stk-bh .sn{{font-weight:700;color:#f8fafc}}.stk-bh .ss{{font-family:monospace;font-size:.78em;color:#94a3b8}}
.stk-row{{display:grid;grid-template-columns:52px minmax(120px,1fr) 80px 96px 58px 58px 64px 64px;gap:6px;align-items:center;padding:5px 8px;border-radius:6px;font-size:.8em}}
.stk-rows .stk-row:nth-child(even){{background:#0f172a}}
.stk-row.stk-head{{background:none;color:#64748b;font-size:.85em;font-weight:600}}
.stk-cd{{font-weight:700;color:#93c5fd}}.stk-nm{{color:#e2e8f0}}
.stk-num{{font-family:monospace;text-align:right}}
.stk-tr{{text-align:center}}
.stk-row.stk-head .stk-num{{font-family:inherit}}
.chart-box{{background:#1e293b;border-radius:10px;padding:16px;overflow-x:auto}}
.disc{{margin-top:28px;padding:14px;background:#1e293b;border-radius:8px;border-left:3px solid #f59e0b;font-size:.72em;color:#94a3b8;line-height:1.6}}
@media(max-width:768px){{.hide-m{{display:none}}.rk-row{{grid-template-columns:24px 1fr 64px 44px 48px}}.stk-row{{grid-template-columns:44px 1fr 76px 52px 52px 34px}}.t5-row{{grid-template-columns:20px 1fr 84px 64px}}}}
</style></head><body>
<div class="wrap">
<div class="hd">
  <h1>日本セクター 中期モメンタム</h1>
  <div class="sub">直近1〜6ヶ月の「勢い」で強いセクターと銘柄を選ぶ（週1チェック想定）</div>
  <div class="dt">データ基準日 {date_str}</div>
  <div class="pm">勢いスコア = 1・3・6ヶ月リターンを月あたりに換算した平均 ／ 向き = 13週移動平均より上(↑)か下(↓)か</div>
  <div class="upd">最終更新: {now_str}</div>
</div>
{top5_sec}<div class="sec"><div class="sec-t">使い方（忙しい人向け）</div>
<div class="guide">
<div><span class="g-no">1</span>週末にこのページを開く（毎営業日の朝に自動更新されています）</div>
<div><span class="g-no">2</span>「強いセクターの注目銘柄」から気になるものを選ぶ</div>
<div><span class="g-no">3</span>月曜の朝までに注文を入れる（中期の勢いを見るので、タイミングの厳密さは不要です）</div>
<div><span class="g-no">4</span>見直しは月1回程度で十分。保有銘柄のセクターが「弱い」に落ちていたら売却を検討</div>
</div></div>
<div class="sec"><div class="sec-t">セクター勢いランキング（全17業種）</div>
<div class="sec-note">「弱い」セクターの銘柄を保有している場合は、見直しの検討材料にしてください。</div>
{rank_html}</div>
<div class="sec"><div class="sec-t">強いセクターの注目銘柄</div>
<div class="sec-note">上昇トレンド中で勢いの強い順。「100株目安」は最低限の投資金額のおおよその目安です（終値×100株）。</div>
{stock_html}</div>
<div class="sec"><div class="sec-t">全セクターの勢い（月あたり%）</div>
<div class="chart-box"><canvas id="cv"></canvas></div></div>
<div class="disc"><strong>⚠ 注意:</strong> 本ツールは過去の値動き（モメンタム＝勢いは続きやすいという経験則）だけを基にした研究・教育目的のプロトタイプであり、投資助言・推奨ではありません。銘柄はあらかじめ登録した大型株リストから機械的に抽出しており、企業業績・ニュース・バリュエーション等は一切考慮していません。1〜3ヶ月程度の保有を想定した指標ですが、相場急変時には機能しないことがあります。過去のパフォーマンスは将来の成果を保証しません。実際の投資判断はご自身の責任で行ってください。</div>
</div>
<script>
const D={chart_data};
const cv=document.getElementById("cv"),cx=cv.getContext("2d");
function draw(){{
const w=cv.parentElement.offsetWidth-32;
if(w<100)return;
cv.width=w;cv.height=280;
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
const vY=d.s>=0?zY-h-5:zY-h+12;cx.fillText(d.s.toFixed(1),x+bW/2,vY);}});
cx.fillStyle="#64748b";cx.font="10px sans-serif";cx.textAlign="right";
cx.fillText("0",P.l-4,zY+4);cx.fillText("+"+mx.toFixed(1)+"%",P.l-4,P.t+10);cx.fillText("-"+mx.toFixed(1)+"%",P.l-4,H-P.b-2);
}}
draw();
let rt;window.addEventListener("resize",()=>{{clearTimeout(rt);rt=setTimeout(draw,150);}});
</script></body></html>"""
    return html


def main():
    output_path = "signal_dashboard.html"

    print("セクターETFデータ取得中...")
    try:
        etf_prices = fetch_prices(list(JP_TICKERS.keys()))
    except Exception as e:
        print(f"データ取得エラー: {e}")
        sys.exit(1)

    ranking, n_top = rank_sectors(etf_prices)
    if len(ranking) < 5:
        print(f"セクターデータが不足しています（{len(ranking)}業種のみ）")
        sys.exit(1)
    print(f"  セクターETF: {len(ranking)}業種")

    top_sectors = [tk for tk, _ in ranking[:n_top]]
    stock_analysis, all_rows = analyze_sector_stocks(list(JP_TICKERS.keys()))
    top_stocks = pick_top_stocks(all_rows)

    date_str = etf_prices.index[-1].strftime("%Y-%m-%d")

    html = build_html(ranking, n_top, stock_analysis, top_stocks, date_str)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[完了] {output_path} を更新しました")
    print(f"  データ基準日: {date_str}")
    print("\n  今週の注目銘柄 Top5（全セクター対象）:")
    for i, r in enumerate(top_stocks, 1):
        print(f"   {i}. {r['code'].replace('.T', '')} {r['name']}"
              f"（{JP_TICKERS.get(r['sector'], '')}）  勢い{r['score']:+.1%}/月")
    print("\n  セクター勢いランキング:")
    for i, (tk, m) in enumerate(ranking, 1):
        mark = "↑" if m["uptrend"] else "↓"
        tag = ""
        if i <= n_top:
            tag = "★強い"
        elif i > len(ranking) - n_top:
            tag = "▼弱い"
        print(f"   {i:2d}. {JP_TICKERS.get(tk, tk)}  勢い{m['score']:+.1%}/月 {mark} {tag}")
    print("\n  強いセクターの注目銘柄:")
    for tk in top_sectors:
        print(f"    {JP_TICKERS.get(tk, tk)}:")
        for r in stock_analysis.get(tk, []):
            print(f"      └ {r['code'].replace('.T', '')} {r['name']}  "
                  f"勢い{r['score']:+.1%}/月  3ヶ月{r['r3']:+.1%}")


if __name__ == "__main__":
    main()
