from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_and_keep_json_only(
    source_dir: Path,
    destination_dir: Path,
    overwrite: bool = False,
) -> None:
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Đường dẫn nguồn không phải thư mục: {source_dir}")

    if destination_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Thư mục đích đã tồn tại: {destination_dir}\n"
                "Dùng --overwrite nếu muốn xoá và copy lại."
            )

        shutil.rmtree(destination_dir)

    # Copy nguyên toàn bộ thư mục nguồn sang thư mục đích
    shutil.copytree(source_dir, destination_dir)

    removed_files = 0
    kept_json_files = 0

    # Xoá tất cả file không phải JSON trong thư mục đích
    for file_path in destination_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() == ".json":
            kept_json_files += 1
        else:
            file_path.unlink()
            removed_files += 1
            print(f"Đã xoá: {file_path}")

    # Xoá những thư mục bị rỗng sau khi xoá file
    directories = [
        path
        for path in destination_dir.rglob("*")
        if path.is_dir()
    ]

    # Phải duyệt từ thư mục sâu nhất trở ra
    for directory in sorted(
        directories,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
            print(f"Đã xoá thư mục rỗng: {directory}")
        except OSError:
            # Thư mục vẫn còn file hoặc thư mục con
            pass

    print("\nHoàn thành.")
    print(f"Giữ lại: {kept_json_files} file JSON")
    print(f"Đã xoá: {removed_files} file không phải JSON")
    print(f"Thư mục kết quả: {destination_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy nguyên thư mục nguồn sang thư mục đích, "
            "sau đó chỉ giữ lại các file JSON."
        )
    )

    parser.add_argument(
        "source_dir",
        type=Path,
        help="Thư mục nguồn.",
    )

    parser.add_argument(
        "destination_dir",
        type=Path,
        help="Thư mục đích.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xoá thư mục đích nếu đã tồn tại rồi copy lại.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        copy_and_keep_json_only(
            source_dir=args.source_dir,
            destination_dir=args.destination_dir,
            overwrite=args.overwrite,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        PermissionError,
    ) as error:
        print(f"Lỗi: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()