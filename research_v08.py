# =====================================================
# research_v08.py
# 岡畠AI研究所 Ver.0.8
# v07最終候補をさらに厳しく絞り込む分析ツール
#
# 実行方法:
#   python research_v08.py
#
# 入力:
#   data/research_v07/final_candidates.csv
#
# 出力:
#   data/research_v08/final_candidates_v08.csv
#   data/research_v08/watch_candidates_v08.csv
#   data/research_v08/rejected_candidates_v08.csv
#   data/research_v08/research_v08_report.txt
# =====================================================

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data" / "research_v07" / "final_candidates.csv"
OUT_DIR = BASE_DIR / "data" / "research_v08"


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 数値列を安全に変換
    num_cols = [
        "v07最終スコア", "元PF", "元期待値pips/回", "元勝率%", "元取引回数",
        "月プラス率%", "最悪月pips",
        "2025_損益pips", "2025_PF", "2025_期待値", "2025_取引回数",
        "2026_損益pips", "2026_PF", "2026_期待値", "2026_取引回数",
        "フォワード損益pips", "フォワードPF", "フォワード期待値", "フォワード取引回数",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = safe_num(df[c])

    # 2025・2026・フォワードの三段階で落ちないかを見る
    df["年跨ぎ安定"] = (
        (df["2025_損益pips"] > 0)
        & (df["2026_損益pips"] > 0)
        & (df["フォワード損益pips"] > 0)
    )

    df["PF安定"] = (
        (df["2025_PF"] >= 1.15)
        & (df["2026_PF"] >= 1.15)
        & (df["フォワードPF"] >= 1.15)
    )

    df["期待値安定"] = (
        (df["2025_期待値"] > 0)
        & (df["2026_期待値"] > 0)
        & (df["フォワード期待値"] > 0)
    )

    df["取引数十分"] = (
        (df["2025_取引回数"] >= 40)
        & (df["2026_取引回数"] >= 20)
        & (df["フォワード取引回数"] >= 80)
    )

    df["月安定"] = df["月プラス率%"] >= 45

    # 最悪月が深すぎるものは実戦では危険
    df["DD許容"] = df["最悪月pips"] >= -250

    # 0〜100点のv08スコア
    df["v08スコア"] = (
        df["v07最終スコア"] * 0.25
        + np.minimum(df["フォワードPF"], 3.0) / 3.0 * 25
        + np.minimum(df["フォワード期待値"], 5.0) / 5.0 * 20
        + np.minimum(df["月プラス率%"], 100.0) / 100.0 * 15
        + np.where(df["年跨ぎ安定"], 8, 0)
        + np.where(df["PF安定"], 4, 0)
        + np.where(df["DD許容"], 3, 0)
    ).round(1)

    reasons = []
    verdicts = []
    for _, r in df.iterrows():
        bad = []
        good = []

        if r["年跨ぎ安定"]:
            good.append("2025・2026・フォワードすべて利益")
        else:
            bad.append("年別またはフォワードで利益が崩れる")

        if r["PF安定"]:
            good.append("PFが各期間で1.15以上")
        else:
            bad.append("PFの安定性が弱い")

        if r["期待値安定"]:
            good.append("期待値が各期間でプラス")
        else:
            bad.append("期待値がどこかでマイナス")

        if r["取引数十分"]:
            good.append("取引数は最低限あり")
        else:
            bad.append("取引数不足")

        if r["月安定"]:
            good.append("月プラス率45%以上")
        else:
            bad.append("月単位の安定性が弱い")

        if r["DD許容"]:
            good.append("最悪月は許容範囲")
        else:
            bad.append("最悪月の落ち込みが大きい")

        if r["v08スコア"] >= 82 and len(bad) <= 1 and r["年跨ぎ安定"] and r["期待値安定"]:
            verdict = "本命"
        elif r["v08スコア"] >= 72 and len(bad) <= 3:
            verdict = "監視"
        else:
            verdict = "除外"

        verdicts.append(verdict)
        reasons.append(" / ".join(good[:3] + (["注意: " + "、".join(bad[:2])] if bad else [])))

    df["v08判定"] = verdicts
    df["v08理由"] = reasons

    return df.sort_values(["v08判定", "v08スコア"], ascending=[True, False]).reset_index(drop=True)


def build_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("岡畠AI研究所 Ver.0.8 最終候補 絞り込みレポート")
    lines.append("=" * 70)
    lines.append(f"入力候補数: {len(df)}件")
    lines.append("")
    lines.append("判定内訳:")
    for k, v in df["v08判定"].value_counts().items():
        lines.append(f"  {k}: {v}件")

    lines.append("")
    lines.append("=" * 70)
    lines.append("本命候補")
    lines.append("=" * 70)
    honmei = df[df["v08判定"] == "本命"].sort_values("v08スコア", ascending=False)
    if honmei.empty:
        lines.append("本命候補なし。監視候補でデモ検証から開始。")
    else:
        for i, (_, r) in enumerate(honmei.iterrows(), 1):
            lines.append(
                f"{i}. {r['戦略']} TP{int(r['利確pips'])} SL{int(r['損切りpips'])} "
                f"{r['時間帯']} {r['曜日']} {r['トレンド環境']} {r['ボラ環境']}"
            )
            lines.append(
                f"   v08スコア:{r['v08スコア']} / フォワードPF:{r['フォワードPF']} "
                f"/ 期待値:{r['フォワード期待値']}pips / 損益:{r['フォワード損益pips']}pips "
                f"/ 取引:{int(r['フォワード取引回数'])}回"
            )
            lines.append(f"   理由: {r['v08理由']}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("監視候補")
    lines.append("=" * 70)
    watch = df[df["v08判定"] == "監視"].sort_values("v08スコア", ascending=False)
    if watch.empty:
        lines.append("監視候補なし。")
    else:
        for i, (_, r) in enumerate(watch.iterrows(), 1):
            lines.append(
                f"{i}. {r['戦略']} TP{int(r['利確pips'])} SL{int(r['損切りpips'])} "
                f"{r['時間帯']} {r['曜日']} score:{r['v08スコア']} "
                f"FW_PF:{r['フォワードPF']} FW期待値:{r['フォワード期待値']}"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("次にやること")
    lines.append("=" * 70)
    lines.append("1. 本命候補だけをデモ口座または過去データ詳細で追加検証する。")
    lines.append("2. 月別・週別の落ち込みを確認する。")
    lines.append("3. 本命が2件以上あれば、同時稼働時の重複エントリーを確認する。")
    lines.append("4. 次バージョンでは trades を出して、実際のエントリー履歴を確認する。")
    lines.append("")
    lines.append("注意: これは研究用であり、利益を保証するものではありません。")

    return "\n".join(lines)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"入力CSVがありません: {INPUT_CSV}\n"
            "先に python research_v07.py を実行してください。"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    scored = score_candidates(df)

    final_df = scored[scored["v08判定"] == "本命"].sort_values("v08スコア", ascending=False)
    watch_df = scored[scored["v08判定"] == "監視"].sort_values("v08スコア", ascending=False)
    reject_df = scored[scored["v08判定"] == "除外"].sort_values("v08スコア", ascending=False)

    scored.to_csv(OUT_DIR / "all_candidates_v08.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(OUT_DIR / "final_candidates_v08.csv", index=False, encoding="utf-8-sig")
    watch_df.to_csv(OUT_DIR / "watch_candidates_v08.csv", index=False, encoding="utf-8-sig")
    reject_df.to_csv(OUT_DIR / "rejected_candidates_v08.csv", index=False, encoding="utf-8-sig")

    report = build_report(scored)
    (OUT_DIR / "research_v08_report.txt").write_text(report, encoding="utf-8")

    print(report)
    print()
    print("出力しました:")
    print(OUT_DIR / "all_candidates_v08.csv")
    print(OUT_DIR / "final_candidates_v08.csv")
    print(OUT_DIR / "watch_candidates_v08.csv")
    print(OUT_DIR / "rejected_candidates_v08.csv")
    print(OUT_DIR / "research_v08_report.txt")


if __name__ == "__main__":
    main()
