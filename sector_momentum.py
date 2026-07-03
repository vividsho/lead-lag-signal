"""
日本株 短期・中期・長期 3段構えダッシュボード生成スクリプト

3つの時間軸で「きょう買う候補」を提案します。
  短期: 直近1週間〜1ヶ月の勢い（数日〜数週間の保有想定。値動き荒め）
  中期: 直近1〜6ヶ月の勢い（1〜3ヶ月の保有想定。週1チェック向け）
  長期: 骨太の方針2026（政府の17戦略分野・官民370兆円投資）のテーマ銘柄

どのランキングも「リスク調整後スコア」= 勢い ÷ 値動きの荒さ で並べます。
急騰しただけの荒い銘柄より、安定して上がっている銘柄が上に来ます。

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
W1          = 5            # 1週間に相当する営業日数
M1, M3, M6  = 20, 60, 120  # 1・3・6ヶ月に相当する営業日数
M12         = 240          # 12ヶ月に相当する営業日数
MA_SHORT    = 25           # 短期のトレンド判定移動平均（約5週）
MA_DAYS     = 65           # 中期のトレンド判定移動平均（約13週）
MA_LONG     = 130          # 長期のトレンド判定移動平均（約26週）
TOP_Q       = 0.3          # 「強い/弱い」とみなす割合（上位・下位30%）
STOCK_TOP_N = 5            # 各ランキングで提案する銘柄数
FETCH_DAYS  = 500          # データ取得期間（暦日。12ヶ月+移動平均分の余裕込み）

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

# ── 長期テーマ（骨太の方針2026より） ──────────────────────
# 政府が2026年6月30日に示した「経済財政運営と改革の基本方針2026（骨太の方針）」
# 原案の17戦略分野（官民370兆円投資）を、株式市場で投資しやすい10テーマに
# 集約し、各テーマの代表的な大型銘柄を対応付けたもの。
# テーマと銘柄は年1回、骨太の方針の発表時（毎年6月頃）に見直すこと。
LONG_THEMES = {
    "AI・半導体": [
        ("8035.T", "東京エレクトロン"),
        ("6857.T", "アドバンテスト"),
        ("6146.T", "ディスコ"),
        ("6723.T", "ルネサスエレクトロニクス"),
        ("6981.T", "村田製作所"),
    ],
    "防衛・航空宇宙": [
        ("7011.T", "三菱重工業"),
        ("7012.T", "川崎重工業"),
        ("7013.T", "IHI"),
        ("6503.T", "三菱電機"),
    ],
    "造船・海洋・港湾物流": [
        ("7003.T", "三井E&S"),
        ("7014.T", "名村造船所"),
        ("9101.T", "日本郵船"),
        ("9104.T", "商船三井"),
    ],
    "量子・サイバー・情報通信": [
        ("6702.T", "富士通"),
        ("6701.T", "NEC"),
        ("9432.T", "NTT"),
        ("4307.T", "野村総合研究所"),
    ],
    "エネルギー安全保障・GX・核融合": [
        ("1605.T", "INPEX"),
        ("9501.T", "東京電力HD"),
        ("9503.T", "関西電力"),
        ("5801.T", "古河電気工業"),
        ("5803.T", "フジクラ"),
    ],
    "創薬・先端医療": [
        ("4568.T", "第一三共"),
        ("4519.T", "中外製薬"),
        ("4502.T", "武田薬品工業"),
        ("4901.T", "富士フイルムHD"),
    ],
    "防災・国土強靱化": [
        ("1812.T", "鹿島建設"),
        ("1801.T", "大成建設"),
        ("1802.T", "大林組"),
        ("5233.T", "太平洋セメント"),
    ],
    "コンテンツ（ゲーム・アニメ）": [
        ("7974.T", "任天堂"),
        ("6758.T", "ソニーグループ"),
        ("7832.T", "バンダイナムコHD"),
        ("9468.T", "KADOKAWA"),
    ],
    "フードテック・バイオ": [
        ("2802.T", "味の素"),
        ("1332.T", "ニッスイ"),
        ("2801.T", "キッコーマン"),
        ("4118.T", "カネカ"),
    ],
    "マテリアル（重要鉱物・部素材）": [
        ("4063.T", "信越化学工業"),
        ("5713.T", "住友金属鉱山"),
        ("5711.T", "三菱マテリアル"),
        ("6988.T", "日東電工"),
    ],
}

# 銘柄コード → セクター名（表示用）
CODE_SECTOR = {}
for _sec, _rows in JP_STOCKS.items():
    for _code, _ in _rows:
        CODE_SECTOR.setdefault(_code, JP_TICKERS[_sec])


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


def monthly_vol(s: pd.Series, days: int):
    """直近 days 営業日の日次リターンから「月あたりの値動きの荒さ」を計算"""
    ret = s.pct_change().iloc[-days:].dropna()
    if len(ret) < days // 2:
        return None
    v = float(ret.std()) * np.sqrt(20)
    return v if v > 1e-6 else None


def ret_over(s: pd.Series, days: int):
    """直近 days 営業日のリターン。データ不足なら None"""
    if len(s) < days + 1:
        return None
    return float(s.iloc[-1]) / float(s.iloc[-1 - days]) - 1


def short_metrics(prices: pd.Series):
    """短期（数日〜数週間）: 1週間・1ヶ月の勢いをボラティリティで調整"""
    s = prices.dropna()
    if len(s) < M1 + 10:
        return None
    last = float(s.iloc[-1])
    r5  = ret_over(s, W1)
    r20 = ret_over(s, M1)
    vol = monthly_vol(s, M1)
    if r5 is None or r20 is None or vol is None:
        return None
    score = (r5 * 4 + r20) / 2          # 月あたりに換算した勢い
    return {"r5": r5, "r20": r20, "score": score, "adj": score / vol,
            "vol": vol, "price": last,
            "uptrend": last > float(s.iloc[-MA_SHORT:].mean())}


def mid_metrics(prices: pd.Series):
    """中期（1〜3ヶ月保有）: 1・3・6ヶ月の勢いをボラティリティで調整"""
    s = prices.dropna()
    if len(s) < M6 + 5:
        return None
    last = float(s.iloc[-1])
    r1, r3, r6 = ret_over(s, M1), ret_over(s, M3), ret_over(s, M6)
    vol = monthly_vol(s, M3)
    if r1 is None or r6 is None or vol is None:
        return None
    score = (r1 + r3 / 3 + r6 / 6) / 3  # 月あたりに換算した勢い
    return {"r1": r1, "r3": r3, "r6": r6, "score": score, "adj": score / vol,
            "vol": vol, "price": last,
            "uptrend": last > float(s.iloc[-MA_DAYS:].mean())}


def long_metrics(prices: pd.Series):
    """長期（1年以上）: 6・12ヶ月の勢いと26週トレンド"""
    s = prices.dropna()
    if len(s) < M6 + 5:
        return None
    last = float(s.iloc[-1])
    r6  = ret_over(s, M6)
    r12 = ret_over(s, M12)  # データ不足なら None（表では「—」表示）
    vol = monthly_vol(s, M6)
    if r6 is None or vol is None:
        return None
    score = (r12 / 12) if r12 is not None else (r6 / 6)
    ma = s.iloc[-MA_LONG:].mean() if len(s) >= MA_LONG else s.mean()
    return {"r6": r6, "r12": r12, "score": score, "adj": score / vol,
            "vol": vol, "price": last, "uptrend": last > float(ma)}


def rank_sectors(etf_prices: pd.DataFrame):
    """セクターETFを勢いスコアで順位付け"""
    metrics = {}
    for tk in JP_TICKERS:
        if tk in etf_prices.columns:
            m = mid_metrics(etf_prices[tk])
            if m is not None:
                metrics[tk] = m
    ranking = sorted(metrics.items(), key=lambda x: x[1]["score"], reverse=True)
    n_top = max(1, round(len(ranking) * TOP_Q))
    return ranking, n_top


def analyze_stocks(prices: pd.DataFrame):
    """全銘柄の短期・中期・長期指標をまとめて計算する。

    Returns
    -------
    short_rows : list 短期指標（セクター登録銘柄）
    mid_rows   : list 中期指標（セクター登録銘柄）
    mid_by_sec : dict {sector: [銘柄情報, ...]} セクターごとの中期上位
    long_rows  : dict {テーマ: [銘柄情報, ...]} 長期テーマ別
    """
    short_rows, mid_rows = [], []
    seen = set()
    for sec, stocks in JP_STOCKS.items():
        for code, name in stocks:
            if code in seen or code not in prices.columns:
                continue
            seen.add(code)
            base = {"code": code, "name": name,
                    "sector": CODE_SECTOR.get(code, "")}
            ms = short_metrics(prices[code])
            if ms is not None:
                short_rows.append({**base, **ms})
            mm = mid_metrics(prices[code])
            if mm is not None:
                mid_rows.append({**base, **mm, "sec_key": sec})

    mid_by_sec = {}
    for r in mid_rows:
        mid_by_sec.setdefault(r["sec_key"], []).append(r)
    for sec, rows in mid_by_sec.items():
        picked = [r for r in rows if r["uptrend"]]
        if len(picked) < 3:
            picked = rows
        picked.sort(key=lambda r: r["adj"], reverse=True)
        mid_by_sec[sec] = picked[:STOCK_TOP_N]

    long_rows = {}
    for theme, stocks in LONG_THEMES.items():
        rows = []
        for code, name in stocks:
            if code not in prices.columns:
                continue
            ml = long_metrics(prices[code])
            if ml is None:
                continue
            ml.update({"code": code, "name": name,
                       "sector": CODE_SECTOR.get(code, "")})
            rows.append(ml)
        rows.sort(key=lambda r: r["adj"], reverse=True)
        long_rows[theme] = rows

    return short_rows, mid_rows, mid_by_sec, long_rows


def pick_top(rows: list, n: int = STOCK_TOP_N) -> list:
    """リスク調整後スコアの高い順に Top N（上昇トレンド中を優先）"""
    pool = [r for r in rows if r["uptrend"]]
    if len(pool) < n:
        pool = rows
    return sorted(pool, key=lambda r: r["adj"], reverse=True)[:n]


def fmt_cost(price: float) -> str:
    cost = price * 100
    return f"約{cost / 10000:,.1f}万円" if cost >= 100000 else f"約{cost:,.0f}円"


def build_html(ranking, n_top, short_top, mid_top, mid_by_sec, long_rows,
               date_str):
    """HTMLダッシュボードを生成"""
    top_set  = {tk for tk, _ in ranking[:n_top]}
    weak_set = {tk for tk, _ in ranking[-n_top:]}

    def col(v):
        return "#22c55e" if v >= 0 else "#ef4444"

    def pct_span(v, hide=True):
        cls = "rk-num hide-m" if hide else "rk-num"
        return f'<span class="{cls}" style="color:{col(v)}">{v:+.1%}</span>'

    def top5_table(rows, r_label, r_key, row_cls):
        """短期・中期共通の Top5 テーブル"""
        html = (
            f'<div class="t5-row t5-head"><span>#</span><span class="hide-m">コード</span>'
            f'<span>銘柄名</span><span class="hide-m">セクター</span>'
            f'<span class="t5-num">100株目安</span><span class="t5-num hide-m">{r_label}</span>'
            f'<span class="t5-num hide-m">勢い/月</span>'
            f'<span class="t5-num">安定度</span></div>\n'
        )
        for i, r in enumerate(rows, 1):
            html += (
                f'<div class="t5-row {row_cls}">'
                f'<span class="t5-no">{i}</span>'
                f'<span class="t5-cd hide-m">{r["code"].replace(".T", "")}</span>'
                f'<span class="t5-nm">{r["name"]}</span>'
                f'<span class="t5-sec hide-m">{r["sector"]}</span>'
                f'<span class="t5-num">{fmt_cost(r["price"])}</span>'
                f'<span class="t5-num hide-m" style="color:{col(r[r_key])}">{r[r_key]:+.1%}</span>'
                f'<span class="t5-num hide-m" style="color:{col(r["score"])}">{r["score"]:+.1%}</span>'
                f'<span class="t5-num" style="color:{col(r["adj"])};font-weight:700">{r["adj"]:.2f}</span>'
                f'</div>\n'
            )
        return html

    # ── 短期セクション ──
    short_sec = ""
    if short_top:
        short_sec = (
            f'<div class="sec"><div class="sec-t st-short">短期（数日〜数週間）'
            f'きょう買うなら Top5</div>'
            f'<div class="sec-note">直近1週間・1ヶ月の勢いを値動きの荒さで割り引いた順。'
            f'短期枠は値動きの大きい銘柄が入りやすく、<b>急落リスクも大きめ</b>です。'
            f'少額で・逆指値（損切りライン）を決めて使ってください。</div>\n'
            f'{top5_table(short_top, "1ヶ月", "r20", "t5-short")}</div>\n'
        )

    # ── 中期セクション ──
    mid_sec = ""
    if mid_top:
        mid_sec = (
            f'<div class="sec"><div class="sec-t">中期（1〜3ヶ月）今週の注目銘柄 Top5</div>'
            f'<div class="sec-note">登録している全17業種・約130銘柄から、上昇トレンド中で'
            f'「安定度」（勢い÷値動きの荒さ）の高い順に5銘柄。'
            f'急騰しただけの荒い銘柄は下がり、じわじわ安定して上がる銘柄が上に来ます。</div>\n'
            f'{top5_table(mid_top, "3ヶ月", "r3", "")}</div>\n'
        )

    # ── セクターランキング ──
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
        rank_html += (
            f'<div class="rk-row {cls}">'
            f'<span class="rk-no">{i}</span>'
            f'<span class="rk-nm">{JP_TICKERS.get(tk, tk)}</span>'
            f'<span class="rk-num" style="color:{col(m["score"])};font-weight:700">{m["score"]:+.1%}</span>'
            f'{pct_span(m["r1"])}{pct_span(m["r3"])}{pct_span(m["r6"])}'
            f'<span class="rk-tr">{tr_mark}</span>'
            f'<span class="rk-bd">{badge}</span>'
            f'</div>\n'
        )

    # ── 強いセクターの個別銘柄 ──
    sector_score = dict(ranking)

    def stock_block(sec):
        rows_data = mid_by_sec.get(sec, [])
        if not rows_data:
            return ""
        rows = (
            '<div class="stk-row stk-head"><span>コード</span><span>銘柄名</span>'
            '<span class="stk-num hide-m">終値</span><span class="stk-num">100株目安</span>'
            '<span class="stk-num hide-m">3ヶ月</span><span class="stk-num">勢い/月</span>'
            '<span class="stk-num">安定度</span><span class="stk-tr">向き</span></div>\n'
        )
        for r in rows_data:
            tr = ('<span style="color:#22c55e">↑</span>' if r["uptrend"]
                  else '<span style="color:#ef4444">↓</span>')
            rows += (
                f'<div class="stk-row">'
                f'<span class="stk-cd">{r["code"].replace(".T", "")}</span>'
                f'<span class="stk-nm">{r["name"]}</span>'
                f'<span class="stk-num hide-m">{r["price"]:,.0f}円</span>'
                f'<span class="stk-num">{fmt_cost(r["price"])}</span>'
                f'<span class="stk-num hide-m" style="color:{col(r["r3"])}">{r["r3"]:+.1%}</span>'
                f'<span class="stk-num" style="color:{col(r["score"])}">{r["score"]:+.1%}</span>'
                f'<span class="stk-num" style="color:{col(r["adj"])};font-weight:700">{r["adj"]:.2f}</span>'
                f'<span class="stk-tr">{tr}</span>'
                f'</div>\n'
            )
        sec_m = sector_score[sec]
        return (
            f'<div class="stk-block"><div class="stk-bh">'
            f'<span class="sn">{JP_TICKERS.get(sec, sec)}</span>'
            f'<span class="ss">セクター勢い {sec_m["score"]:+.1%}/月</span></div>'
            f'<div class="stk-rows">{rows}</div></div>\n'
        )

    stock_html = "".join(stock_block(tk) for tk, _ in ranking[:n_top])

    # ── 長期セクション（骨太の方針2026テーマ） ──
    def theme_block(theme, rows_data):
        if not rows_data:
            return ""
        rows = (
            '<div class="stk-row stk-head"><span>コード</span><span>銘柄名</span>'
            '<span class="stk-num hide-m">終値</span><span class="stk-num">100株目安</span>'
            '<span class="stk-num hide-m">6ヶ月</span><span class="stk-num">12ヶ月</span>'
            '<span class="stk-num">安定度</span><span class="stk-tr">向き</span></div>\n'
        )
        for r in rows_data:
            tr = ('<span style="color:#22c55e">↑</span>' if r["uptrend"]
                  else '<span style="color:#ef4444">↓</span>')
            r12 = (f'<span class="stk-num" style="color:{col(r["r12"])}">{r["r12"]:+.1%}</span>'
                   if r["r12"] is not None else '<span class="stk-num">—</span>')
            rows += (
                f'<div class="stk-row">'
                f'<span class="stk-cd">{r["code"].replace(".T", "")}</span>'
                f'<span class="stk-nm">{r["name"]}</span>'
                f'<span class="stk-num hide-m">{r["price"]:,.0f}円</span>'
                f'<span class="stk-num">{fmt_cost(r["price"])}</span>'
                f'<span class="stk-num hide-m" style="color:{col(r["r6"])}">{r["r6"]:+.1%}</span>'
                f'{r12}'
                f'<span class="stk-num" style="color:{col(r["adj"])};font-weight:700">{r["adj"]:.2f}</span>'
                f'<span class="stk-tr">{tr}</span>'
                f'</div>\n'
            )
        return (
            f'<div class="stk-block lt-block"><div class="stk-bh">'
            f'<span class="sn">{theme}</span></div>'
            f'<div class="stk-rows">{rows}</div></div>\n'
        )

    long_html = "".join(theme_block(t, rows) for t, rows in long_rows.items())
    long_sec = ""
    if long_html:
        long_sec = (
            f'<div class="sec"><div class="sec-t st-long">長期（1年以上）'
            f'骨太の方針2026の重点テーマ銘柄</div>'
            f'<div class="sec-note">政府が2026年6月30日に示した「骨太の方針2026」の'
            f'17戦略分野（官民370兆円投資）を10テーマに集約し、代表的な大型銘柄を'
            f'対応付けたもの。長期枠は「勢い」より「国の重点投資テーマに乗っているか」で'
            f'選ぶ考え方です。急いで買わず、押し目（下がった時）や積立でコツコツが向きます。'
            f'テーマは毎年6月の方針発表に合わせて見直してください。</div>\n'
            f'{long_html}</div>\n'
        )

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
<title>日本株 短期・中期・長期ダッシュボード</title>
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
.sec-t.st-short{{border-left-color:#f59e0b}}
.sec-t.st-long{{border-left-color:#a78bfa}}
.sec-note{{color:#64748b;font-size:.72em;margin:-6px 0 10px 13px;line-height:1.5}}
.sec-note b{{color:#fbbf24}}
.t5-row{{display:grid;grid-template-columns:24px 52px minmax(120px,1fr) 110px 96px 60px 64px 56px;gap:8px;align-items:center;background:linear-gradient(135deg,#064e3b,#0f2a1f);border:1px solid #065f46;border-radius:8px;padding:9px 12px;margin-bottom:5px;font-size:.85em}}
.t5-row.t5-short{{background:linear-gradient(135deg,#451a03,#271203);border-color:#92400e}}
.t5-row.t5-head{{background:none;border:none;color:#64748b;font-size:.7em;font-weight:600;padding:0 12px;margin-bottom:2px}}
.t5-no{{color:#34d399;font-weight:800}}
.t5-short .t5-no{{color:#fbbf24}}
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
.stk-block.lt-block{{border-left-color:#a78bfa}}
.stk-bh{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;flex-wrap:wrap;gap:4px}}
.stk-bh .sn{{font-weight:700;color:#f8fafc}}.stk-bh .ss{{font-family:monospace;font-size:.78em;color:#94a3b8}}
.stk-row{{display:grid;grid-template-columns:52px minmax(120px,1fr) 80px 96px 58px 64px 60px 40px;gap:6px;align-items:center;padding:5px 8px;border-radius:6px;font-size:.8em}}
.stk-rows .stk-row:nth-child(even){{background:#0f172a}}
.stk-row.stk-head{{background:none;color:#64748b;font-size:.85em;font-weight:600}}
.stk-cd{{font-weight:700;color:#93c5fd}}.stk-nm{{color:#e2e8f0}}
.stk-num{{font-family:monospace;text-align:right}}
.stk-tr{{text-align:center}}
.stk-row.stk-head .stk-num{{font-family:inherit}}
.chart-box{{background:#1e293b;border-radius:10px;padding:16px;overflow-x:auto}}
.disc{{margin-top:28px;padding:14px;background:#1e293b;border-radius:8px;border-left:3px solid #f59e0b;font-size:.72em;color:#94a3b8;line-height:1.6}}
@media(max-width:768px){{.hide-m{{display:none}}.rk-row{{grid-template-columns:24px 1fr 64px 44px 48px}}.stk-row{{grid-template-columns:44px 1fr 76px 56px 50px 30px}}.t5-row{{grid-template-columns:20px 1fr 84px 48px}}}}
</style></head><body>
<div class="wrap">
<div class="hd">
  <h1>日本株 短期・中期・長期ダッシュボード</h1>
  <div class="sub">短期=直近の勢い ／ 中期=1〜6ヶ月の勢い ／ 長期=骨太の方針2026のテーマ</div>
  <div class="dt">データ基準日 {date_str}</div>
  <div class="pm">勢い/月 = 月あたりに換算した上昇率 ／ 安定度 = 勢い÷値動きの荒さ（高いほど安定して上昇中） ／ 向き = 移動平均より上(↑)か下(↓)か</div>
  <div class="upd">最終更新: {now_str}</div>
</div>
<div class="sec"><div class="sec-t">使い方（忙しい人向け）</div>
<div class="guide">
<div><span class="g-no">1</span><b>短期</b>: 数日〜数週間で回す枠。少額・損切りライン必須。毎営業日の朝に自動更新されています</div>
<div><span class="g-no">2</span><b>中期</b>: 週末にチェックして月曜朝までに注文。見直しは月1回、保有銘柄のセクターが「弱い」に落ちたら売却検討</div>
<div><span class="g-no">3</span><b>長期</b>: 国の重点投資テーマ（骨太の方針2026）。急がず押し目や積立で。見直しは年1回（毎年6月の方針発表時）</div>
<div><span class="g-no">4</span>どの枠も「安定度」の列を優先。勢いが大きくても安定度が低い銘柄は値動きが荒く、急落リスクがあります</div>
</div></div>
{short_sec}{mid_sec}<div class="sec"><div class="sec-t">セクター勢いランキング（全17業種・中期）</div>
<div class="sec-note">「弱い」セクターの銘柄を保有している場合は、見直しの検討材料にしてください。</div>
{rank_html}</div>
<div class="sec"><div class="sec-t">強いセクターの注目銘柄（中期）</div>
<div class="sec-note">上昇トレンド中で安定度の高い順。「100株目安」は最低限の投資金額のおおよその目安です（終値×100株）。</div>
{stock_html}</div>
{long_sec}<div class="sec"><div class="sec-t">全セクターの勢い（月あたり%）</div>
<div class="chart-box"><canvas id="cv"></canvas></div></div>
<div class="disc"><strong>⚠ 注意:</strong> 本ツールは過去の値動き（モメンタム＝勢いは続きやすいという経験則）と、政府の公表資料（骨太の方針2026）に基づく研究・教育目的のプロトタイプであり、投資助言・推奨ではありません。銘柄はあらかじめ登録した大型株リストから機械的に抽出しており、企業業績・ニュース・バリュエーション等は一切考慮していません。「安定度」は過去の値動きの荒さで割り引いた指標であり、将来の急落を防ぐものではありません。政府のテーマに沿った銘柄が上がる保証もありません。過去のパフォーマンスは将来の成果を保証しません。実際の投資判断はご自身の責任で行ってください。</div>
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

    # セクター登録銘柄 + 長期テーマ銘柄をまとめて取得
    codes = []
    for stocks in JP_STOCKS.values():
        codes += [c for c, _ in stocks]
    for stocks in LONG_THEMES.values():
        codes += [c for c, _ in stocks]
    codes = list(dict.fromkeys(codes))

    print(f"個別銘柄データ取得中...（{len(codes)}銘柄）")
    try:
        prices = fetch_prices(codes)
    except Exception as e:
        print(f"データ取得エラー: {e}")
        sys.exit(1)

    short_rows, mid_rows, mid_by_sec, long_rows = analyze_stocks(prices)
    short_top = pick_top(short_rows)
    mid_top   = pick_top(mid_rows)

    date_str = etf_prices.index[-1].strftime("%Y-%m-%d")

    html = build_html(ranking, n_top, short_top, mid_top, mid_by_sec,
                      long_rows, date_str)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[完了] {output_path} を更新しました")
    print(f"  データ基準日: {date_str}")
    print("\n  短期（数日〜数週間）Top5:")
    for i, r in enumerate(short_top, 1):
        print(f"   {i}. {r['code'].replace('.T', '')} {r['name']}"
              f"（{r['sector']}）  勢い{r['score']:+.1%}/月  安定度{r['adj']:.2f}")
    print("\n  中期（1〜3ヶ月）Top5:")
    for i, r in enumerate(mid_top, 1):
        print(f"   {i}. {r['code'].replace('.T', '')} {r['name']}"
              f"（{r['sector']}）  勢い{r['score']:+.1%}/月  安定度{r['adj']:.2f}")
    print("\n  セクター勢いランキング:")
    for i, (tk, m) in enumerate(ranking, 1):
        mark = "↑" if m["uptrend"] else "↓"
        tag = ""
        if i <= n_top:
            tag = "★強い"
        elif i > len(ranking) - n_top:
            tag = "▼弱い"
        print(f"   {i:2d}. {JP_TICKERS.get(tk, tk)}  勢い{m['score']:+.1%}/月 {mark} {tag}")
    print("\n  長期（骨太の方針2026テーマ）:")
    for theme, rows in long_rows.items():
        print(f"    {theme}:")
        for r in rows[:3]:
            r12 = f"12ヶ月{r['r12']:+.1%}" if r['r12'] is not None else "12ヶ月 —"
            print(f"      └ {r['code'].replace('.T', '')} {r['name']}  "
                  f"{r12}  安定度{r['adj']:.2f}")


if __name__ == "__main__":
    main()
