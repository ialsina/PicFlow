from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

LABELS_CSV = "labels.csv"
LABEL_NAMES_CSV = "label_names.csv"

PATH_HEADERS = frozenset({"image_path", "path", "image", "relative_path"})
NAME_HEADERS = frozenset({"image_name", "filename", "name", "file_name"})
LABEL_HEADERS = frozenset({"label", "class", "label_id", "class_id"})
NAME_MAP_LABEL_HEADERS = frozenset({"label", "class", "label_id", "class_id"})
NAME_MAP_NAME_HEADERS = frozenset({"name", "label_name", "class_name"})

MIN_LABEL = 0
MAX_LABEL = 9


def default_label_name(label_id: int) -> str:
    display = "0" if label_id == 9 else str(label_id + 1)
    return f"Class {display}"


def relative_image_path(image_root: Path, absolute_path: str) -> str:
    rel = Path(absolute_path).resolve().relative_to(image_root.resolve())
    return PurePosixPath(rel).as_posix()


def normalize_import_path(value: str) -> str:
    return PurePosixPath(value.strip().replace("\\", "/")).as_posix()


@dataclass
class LabelRow:
    image_path: str | None
    image_name: str | None
    label: int


@dataclass
class LabelNameRow:
    label: int
    name: str


@dataclass
class LabelImportStats:
    applied: int = 0
    skipped_unknown: int = 0
    skipped_ambiguous: int = 0
    skipped_invalid: int = 0
    errors: list[str] = field(default_factory=list)

    def add_error(self, message: str, cap: int = 20) -> None:
        if len(self.errors) < cap:
            self.errors.append(message)


def _pick_column(row: dict[str, str], candidates: frozenset[str]) -> str | None:
    for key, value in row.items():
        if key.strip().lower() in candidates:
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def parse_labels_csv(data: bytes | str) -> list[LabelRow]:
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return []
    out: list[LabelRow] = []
    for i, row in enumerate(reader, start=2):
        if not any(v and str(v).strip() for v in row.values()):
            continue
        image_path = _pick_column(row, PATH_HEADERS)
        image_name = _pick_column(row, NAME_HEADERS)
        label_raw = _pick_column(row, LABEL_HEADERS)
        if label_raw is None:
            raise ValueError(f"labels.csv row {i}: missing label column")
        try:
            label = int(label_raw)
        except ValueError:
            raise ValueError(f"labels.csv row {i}: invalid label '{label_raw}'")
        out.append(
            LabelRow(
                image_path=image_path,
                image_name=image_name,
                label=label,
            )
        )
    return out


def parse_label_names_csv(data: bytes | str) -> list[LabelNameRow]:
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return []
    out: list[LabelNameRow] = []
    for i, row in enumerate(reader, start=2):
        if not any(v and str(v).strip() for v in row.values()):
            continue
        label_raw = _pick_column(row, NAME_MAP_LABEL_HEADERS)
        name = _pick_column(row, NAME_MAP_NAME_HEADERS)
        if label_raw is None or name is None:
            raise ValueError(f"label_names.csv row {i}: need label and name columns")
        try:
            label = int(label_raw)
        except ValueError:
            raise ValueError(f"label_names.csv row {i}: invalid label '{label_raw}'")
        out.append(LabelNameRow(label=label, name=name.strip()))
    return out


def extract_zip_payloads(archive: bytes) -> tuple[bytes, bytes | None]:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = {info.filename: info for info in zf.infolist() if not info.is_dir()}
        labels_key = _find_zip_member(names, LABELS_CSV)
        if not labels_key:
            raise ValueError(f"ZIP archive must contain {LABELS_CSV}")
        labels_data = zf.read(labels_key)
        names_key = _find_zip_member(names, LABEL_NAMES_CSV)
        names_data = zf.read(names_key) if names_key else None
    return labels_data, names_data


def _find_zip_member(names: dict[str, zipfile.ZipInfo], basename: str) -> str | None:
    target = basename.lower()
    for name in names:
        if Path(name).name.lower() == target:
            return name
    return None


def build_labels_csv(rows: list[tuple[str, str, int]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["image_path", "image_name", "label"])
    for image_path, image_name, label in rows:
        writer.writerow([image_path, image_name, label])
    return buf.getvalue().encode("utf-8")


def build_label_names_csv(
    label_names: dict[str, str], used_label_ids: set[int]
) -> bytes:
    ids: set[int] = set(used_label_ids)
    for key in label_names:
        try:
            ids.add(int(key))
        except ValueError:
            continue
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["label", "name"])
    for label_id in sorted(ids):
        custom = label_names.get(str(label_id), "").strip()
        name = custom if custom else default_label_name(label_id)
        writer.writerow([label_id, name])
    return buf.getvalue().encode("utf-8")


def build_export_zip(labels_csv: bytes, label_names_csv: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(LABELS_CSV, labels_csv)
        zf.writestr(LABEL_NAMES_CSV, label_names_csv)
    return out.getvalue()


@dataclass
class SessionImageIndex:
    by_relative_path: dict[str, int]
    by_basename: dict[str, int]
    ambiguous_basenames: set[str]


def build_session_image_index(
    image_root: Path, paths: list[tuple[int, str]]
) -> SessionImageIndex:
    root = image_root.resolve()
    by_relative_path: dict[str, int] = {}
    basename_to_ids: dict[str, list[int]] = {}
    for image_id, abs_path in paths:
        rel = relative_image_path(root, abs_path)
        by_relative_path[normalize_import_path(rel)] = image_id
        base = Path(abs_path).name
        basename_to_ids.setdefault(base, []).append(image_id)
    by_basename: dict[str, int] = {}
    ambiguous: set[str] = set()
    for base, ids in basename_to_ids.items():
        if len(ids) == 1:
            by_basename[base] = ids[0]
        else:
            ambiguous.add(base)
    return SessionImageIndex(
        by_relative_path=by_relative_path,
        by_basename=by_basename,
        ambiguous_basenames=ambiguous,
    )


def resolve_image_id(row: LabelRow, index: SessionImageIndex) -> str | int:
    """Return image id or error code: 'unknown', 'ambiguous', 'no_key'."""
    if not row.image_path and not row.image_name:
        return "no_key"
    if row.image_path:
        key = normalize_import_path(row.image_path)
        if key in index.by_relative_path:
            return index.by_relative_path[key]
    if row.image_name:
        base = Path(row.image_name.strip()).name
        if base in index.ambiguous_basenames:
            return "ambiguous"
        if base in index.by_basename:
            return index.by_basename[base]
    return "unknown"
