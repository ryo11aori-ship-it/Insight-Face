# -*- coding: utf-8 -*-
"""
顔一致度分析システム - アプリ本体
==================================
GUIモード:  python app.py            → ブラウザが開き、ドラッグ&ドロップで操作
CLIモード:  python app.py --ref 資料写真フォルダまたはファイル... --probe 検体写真B
"""

import argparse
import sys
from pathlib import Path

from matcher import FaceMatcher, SUPPORTED_EXTS, HEIF_OK


def format_report(report) -> str:
    lines = [
        "=" * 52,
        f"  同一人物である推定確率: {report.probability * 100:.1f} %",
        f"  判定: {report.verdict}",
        "=" * 52,
        "",
        f"統合類似度スコア   : {report.combined_score:.4f}",
        f"A群平均顔との類似度: {report.centroid_sim:.4f}",
        f"個別最大類似度     : {report.max_sim:.4f}",
        f"個別平均類似度     : {report.mean_sim:.4f}",
        "",
        "── 資料写真ごとの類似度 ──",
    ]
    for name, s in sorted(report.per_photo_sims, key=lambda x: -x[1]):
        bar = "█" * max(0, int(s * 40))
        lines.append(f"  {name:<24s} {s:+.4f} {bar}")
    if report.warnings:
        lines.append("")
        lines.append("── 注意 ──")
        lines.extend(f"  ⚠ {w}" for w in report.warnings)
    lines.append("")
    lines.append("※ 確率はArcFace類似度分布に基づく統計的推定値であり、")
    lines.append("  法的証明力を持つものではありません。")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI モード
# ----------------------------------------------------------------------
def collect_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(
                f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"⚠ 見つかりません: {p}", file=sys.stderr)
    return files


def run_cli(args):
    ref_files = collect_files(args.ref)
    if not ref_files:
        sys.exit("エラー: 資料写真Aが見つかりません")
    probe = Path(args.probe)
    if not probe.is_file():
        sys.exit(f"エラー: 検体写真Bが見つかりません: {probe}")

    print(f"モデルを読み込み中... (初回はダウンロードが発生します)")
    m = FaceMatcher(det_size=args.det_size)

    print(f"資料写真A群 {len(ref_files)} 枚を登録中...")
    warns = m.register_references(ref_files, [f.name for f in ref_files])
    for w in warns:
        print(f"  ⚠ {w}")

    print(f"検体写真B ({probe.name}) を照合中...\n")
    report = m.match(probe, probe.name)
    print(format_report(report))


# ----------------------------------------------------------------------
# GUI モード (Gradio)
# ----------------------------------------------------------------------
def run_gui(args):
    import gradio as gr

    state = {"matcher": None}

    def get_matcher():
        if state["matcher"] is None:
            state["matcher"] = FaceMatcher(det_size=args.det_size)
        return state["matcher"]

    def register(files):
        if not files:
            return "資料写真A群をアップロードしてください"
        try:
            m = get_matcher()
            paths = [f.name if hasattr(f, "name") else f for f in files]
            warns = m.register_references(paths, [Path(p).name for p in paths])
            msg = f"✅ 資料写真A群: {len(m.reference_faces)} 枚の顔を登録しました"
            if warns:
                msg += "\n" + "\n".join(f"⚠ {w}" for w in warns)
            return msg
        except Exception as e:
            return f"❌ エラー: {e}"

    def analyze(probe):
        if probe is None:
            return "検体写真Bをアップロードしてください"
        try:
            m = get_matcher()
            path = probe.name if hasattr(probe, "name") else probe
            report = m.match(path, Path(path).name)
            return format_report(report)
        except Exception as e:
            return f"❌ エラー: {e}"

    exts = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTS))
    heif_note = "" if HEIF_OK else "(HEIC対応には pip install pillow-heif)"

    with gr.Blocks(title="顔一致度分析システム") as demo:
        gr.Markdown(
            f"""# 顔一致度分析システム
**手順:** ① 同一人物の資料写真A群を複数枚アップロード → 登録 ② 検体写真Bをアップロード → 分析
対応形式: {exts} {heif_note} — サイズ自由・完全ローカル動作"""
        )
        with gr.Row():
            with gr.Column():
                ref_in = gr.File(
                    label="① 資料写真A群(複数枚可)",
                    file_count="multiple", type="filepath")
                reg_btn = gr.Button("A群を登録", variant="primary")
                reg_out = gr.Textbox(label="登録結果", lines=4)
            with gr.Column():
                probe_in = gr.File(label="② 検体写真B(1枚)", type="filepath")
                ana_btn = gr.Button("分析実行", variant="primary")
                result = gr.Textbox(label="分析結果", lines=20,
                                    show_copy_button=True)

        reg_btn.click(register, inputs=ref_in, outputs=reg_out)
        ana_btn.click(analyze, inputs=probe_in, outputs=result)

    demo.launch(inbrowser=True, server_name="127.0.0.1")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="顔一致度分析システム")
    ap.add_argument("--ref", nargs="+",
                    help="資料写真A群(ファイルまたはフォルダ、複数指定可)")
    ap.add_argument("--probe", help="検体写真B(1枚)")
    ap.add_argument("--det-size", type=int, default=640,
                    help="顔検出解像度 (既定 640。集合写真等は 1024 推奨)")
    args = ap.parse_args()

    if args.ref and args.probe:
        run_cli(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
