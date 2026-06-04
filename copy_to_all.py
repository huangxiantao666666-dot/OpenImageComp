import os
import shutil
from pathlib import Path

DATASET = Path(__file__).parent / "OPA" / "dataset"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def collect_images(src_dir: Path) -> list[tuple[Path, str]]:
    """Collect all images under src_dir, ignoring the 'all' folder."""
    images = []
    for sub in sorted(src_dir.iterdir()):
        if sub.name == "all" or not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.suffix.lower() in IMG_EXTENSIONS:
                images.append((f, sub.name))
    return images


def copy_all(category: str) -> None:
    src_dir = DATASET / category
    dst_dir = src_dir / "all"
    dst_dir.mkdir(exist_ok=True)

    images = collect_images(src_dir)
    conflicts = {}
    names_seen: dict[str, str] = {}

    # Detect conflicts: same filename from different subfolders
    for f, sub in images:
        if f.name in names_seen and names_seen[f.name] != sub:
            conflicts.setdefault(f.name, {names_seen[f.name]}).add(sub)
        else:
            names_seen[f.name] = sub

    copied = 0
    for f, sub in images:
        if f.name in conflicts:
            # Rename to avoid overwrite: subfolder_filename
            new_name = f"{sub}_{f.name}"
        else:
            new_name = f.name

        dst = dst_dir / new_name
        if dst.exists():
            continue  # already there, skip
        shutil.copy2(f, dst)
        copied += 1

    print(f"[{category}] {copied} copied, {len(conflicts)} name conflicts resolved.")


if __name__ == "__main__":
    for cat in ("background", "foreground"):
        copy_all(cat)
    print("Done.")
