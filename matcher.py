# -*- coding: utf-8 -*-
"""
顔一致度分析システム - コアエンジン
=====================================
InsightFace (RetinaFace 検出 + ArcFace 埋め込み) を使用。
buffalo_l モデルは LFW ベンチマークで 99.8%+ の精度を持つ、
オープンソースで利用可能な最高水準の顔認証モデルです。

- 資料写真A群(同一人物の複数写真)を登録
- 検体写真Bとの一致確率を算出
- 完全ローカル動作(初回のみモデル自動ダウンロード ~300MB)
"""

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

# HEIC/HEIF (iPhone写真) 対応 - 任意
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

# RAW (DNG等) は Pillow 非対応のため rawpy があれば対応 - 任意
try:
    import rawpy
    RAW_OK = True
except ImportError:
    RAW_OK = False

RAW_EXTS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf"}

SUPPORTED_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".gif", ".ppm", ".pgm", ".jp2", ".ico",
    ".heic", ".heif", ".avif",
} | RAW_EXTS


# ----------------------------------------------------------------------
# 画像読み込み(多種多様な形式・サイズに対応)
# ----------------------------------------------------------------------
def load_image(src: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """あらゆる入力を BGR numpy 配列に正規化する。EXIF回転も補正。"""
    if isinstance(src, np.ndarray):
        img = src
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[..., :3].astype(np.uint8)

    if isinstance(src, Image.Image):
        pil = src
    elif isinstance(src, bytes):
        pil = Image.open(io.BytesIO(src))
    else:
        p = Path(src)
        if p.suffix.lower() in RAW_EXTS:
            if not RAW_OK:
                raise RuntimeError(
                    f"RAW形式 {p.suffix} の読み込みには rawpy が必要です: pip install rawpy")
            with rawpy.imread(str(p)) as raw:
                rgb = raw.postprocess()
            return rgb[..., ::-1].copy()  # RGB -> BGR
        pil = Image.open(p)

    pil = ImageOps.exif_transpose(pil)  # スマホ写真の回転補正
    pil = pil.convert("RGB")

    # 極端に大きい画像は検出精度と速度のため長辺4000pxに縮小
    w, h = pil.size
    long_side = max(w, h)
    if long_side > 4000:
        scale = 4000 / long_side
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    rgb = np.asarray(pil)
    return rgb[..., ::-1].copy()  # RGB -> BGR (OpenCV/InsightFace 標準)


# ----------------------------------------------------------------------
# 顔エンジン
# ----------------------------------------------------------------------
@dataclass
class FaceResult:
    embedding: np.ndarray          # L2正規化済み 512次元
    det_score: float               # 検出信頼度
    bbox: np.ndarray               # 顔の位置
    num_faces_in_image: int        # 画像内の顔の総数
    source_name: str = ""


@dataclass
class MatchReport:
    probability: float             # 同一人物である推定確率 (0-1)
    combined_score: float          # 統合コサイン類似度
    centroid_sim: float            # A群平均埋め込みとの類似度
    max_sim: float                 # A群個別写真との最大類似度
    mean_sim: float                # A群個別写真との平均類似度
    per_photo_sims: list = field(default_factory=list)  # [(名前, 類似度)]
    verdict: str = ""
    warnings: list = field(default_factory=list)


class FaceMatcher:
    """
    A群写真を登録 → B写真と照合して同一人物確率を返す。
    """

    # ArcFace(buffalo_l) のコサイン類似度分布に基づくロジスティック較正。
    # 同一人物ペアは概ね 0.45〜0.75、他人ペアは -0.1〜0.25 に分布する。
    _CALIB_MID = 0.32      # P=50% となる類似度
    _CALIB_SLOPE = 16.0    # 曲線の急峻さ

    def __init__(self, det_size: int = 640, ctx_id: int = 0):
        from insightface.app import FaceAnalysis
        # buffalo_l: RetinaFace-10GF 検出 + ArcFace ResNet100 認識 (最高精度パック)
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self.reference_faces: list[FaceResult] = []

    # ------------------------------------------------------------------
    def _detect_best_face(self, img: np.ndarray, name: str = "") -> FaceResult:
        faces = self.app.get(img)
        if not faces:
            # 小さい顔対策: 2倍に拡大して再試行
            up = np.ascontiguousarray(
                np.kron(img, np.ones((2, 2, 1), dtype=np.uint8))[:img.shape[0]*2, :img.shape[1]*2]
            ) if max(img.shape[:2]) < 1200 else None
            if up is not None:
                faces = self.app.get(up)
            if not faces:
                raise ValueError(f"顔が検出できませんでした: {name or '(画像)'}")
        # 最大の顔を採用
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
                   reverse=True)
        f = faces[0]
        emb = f.normed_embedding.astype(np.float32)
        return FaceResult(
            embedding=emb,
            det_score=float(f.det_score),
            bbox=f.bbox,
            num_faces_in_image=len(faces),
            source_name=name,
        )

    # ------------------------------------------------------------------
    def register_references(self, sources: list, names: list[str] | None = None):
        """資料写真A群を登録。戻り値: 警告メッセージのリスト"""
        self.reference_faces = []
        warnings = []
        names = names or [f"A{i+1}" for i in range(len(sources))]
        for src, name in zip(sources, names):
            try:
                img = load_image(src)
                fr = self._detect_best_face(img, name)
                if fr.num_faces_in_image > 1:
                    warnings.append(
                        f"{name}: 顔が{fr.num_faces_in_image}個検出されたため、最大の顔を使用しました")
                if fr.det_score < 0.6:
                    warnings.append(f"{name}: 検出信頼度が低めです ({fr.det_score:.2f})")
                self.reference_faces.append(fr)
            except Exception as e:
                warnings.append(f"{name}: 登録失敗 - {e}")
        if not self.reference_faces:
            raise ValueError("A群から顔を1つも登録できませんでした")

        # 外れ値チェック: A群内で他と著しく類似しない写真は別人の可能性
        if len(self.reference_faces) >= 3:
            embs = np.stack([f.embedding for f in self.reference_faces])
            simmat = embs @ embs.T
            n = len(embs)
            for i in range(n):
                others = (simmat[i].sum() - 1.0) / (n - 1)
                if others < 0.20:
                    warnings.append(
                        f"{self.reference_faces[i].source_name}: A群内の他写真との類似度が低く、"
                        f"別人が混入している可能性があります (平均類似度 {others:.2f})")
        return warnings

    # ------------------------------------------------------------------
    def match(self, probe_src, probe_name: str = "B") -> MatchReport:
        """検体写真Bを照合し、同一人物確率を返す"""
        if not self.reference_faces:
            raise ValueError("先に資料写真A群を登録してください")

        img = load_image(probe_src)
        pf = self._detect_best_face(img, probe_name)

        warnings = []
        if pf.num_faces_in_image > 1:
            warnings.append(
                f"検体写真に顔が{pf.num_faces_in_image}個あり、最大の顔で照合しました")
        if pf.det_score < 0.6:
            warnings.append(f"検体写真の検出信頼度が低めです ({pf.det_score:.2f})")

        ref_embs = np.stack([f.embedding for f in self.reference_faces])
        sims = ref_embs @ pf.embedding                      # 個別類似度
        centroid = ref_embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroid_sim = float(centroid @ pf.embedding)       # 平均顔との類似度

        max_sim = float(sims.max())
        mean_sim = float(sims.mean())
        # 上位半分の平均(質の悪い資料写真の影響を軽減)
        k = max(1, math.ceil(len(sims) / 2))
        topk_mean = float(np.sort(sims)[-k:].mean())

        # 統合スコア: 平均顔類似度と上位個別類似度をバランス
        combined = 0.45 * centroid_sim + 0.35 * topk_mean + 0.20 * max_sim

        prob = 1.0 / (1.0 + math.exp(-self._CALIB_SLOPE * (combined - self._CALIB_MID)))

        report = MatchReport(
            probability=prob,
            combined_score=combined,
            centroid_sim=centroid_sim,
            max_sim=max_sim,
            mean_sim=mean_sim,
            per_photo_sims=[
                (f.source_name, float(s))
                for f, s in zip(self.reference_faces, sims)
            ],
            verdict=self._verdict(prob),
            warnings=warnings,
        )
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _verdict(p: float) -> str:
        if p >= 0.98:
            return "同一人物である可能性が極めて高い"
        if p >= 0.90:
            return "同一人物である可能性が非常に高い"
        if p >= 0.70:
            return "同一人物の可能性が高い"
        if p >= 0.40:
            return "判定困難(追加の資料写真を推奨)"
        if p >= 0.10:
            return "別人の可能性が高い"
        return "別人である可能性が極めて高い"
