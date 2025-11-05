import shutil
from pathlib import Path

import click

from megatron.energon.flavors import BaseWebdatasetFactory


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--num-workers", default=4, show_default=True, help="Number of workers for indexing.")
@click.option("--template-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, help="Directory containing template files to copy.")
def prepare_one_dataset(path: Path, num_workers: int, template_dir: Path):
    """
    Prepare a dataset directory for use with Energon.

    PATH: Path to the dataset directory.
    """
    if (path / ".nv-meta" / "dataset.yaml").exists():
        print(f"Dataset {path} already prepared. Skipping.")
        return

    # Fixed settings
    tar_index_only = False
    split_parts_ratio = None
    split_parts_patterns = [
        ("train", "train/.*"),
        ("val", "val/.*")
    ]

    # Get all tar files
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]

    if len(all_tars) == 0:
        print("Did not find any tar files. Exiting.")
        return

    print(f"Found {len(all_tars)} tar files in total. The first and last ones are:")
    print(f"- {all_tars[0]}")
    print(f"- {all_tars[-1]}")

    def progress_fn(els, length=None):
        with click.progressbar(
            els,
            label="Indexing shards",
            show_pos=True,
            length=length,
        ) as bar:
            for el in bar:
                yield el

    found_types, duplicates = BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=split_parts_patterns,
        progress_fn=progress_fn,
        tar_index_only=tar_index_only,
        shuffle_seed=None,
        workers=num_workers,
    )

    # Copy sample loader and dataset.yaml templates
    for file in template_dir.glob("*"):
        shutil.copy(file, path / ".nv-meta" / file.name)

if __name__ == "__main__":
    prepare_one_dataset()
