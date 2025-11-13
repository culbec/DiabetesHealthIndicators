import json
import pathlib
import requests
import zipfile

ROOT_DIR = pathlib.Path(__file__).parent
DATASETS_FILE = ROOT_DIR / "datasets.json"


def download_datasets() -> list[tuple[pathlib.Path, Exception]]:
    datasets: list[tuple[str, Exception | None]] = []

    with open(DATASETS_FILE, "r") as f:
        asset_datasets = json.load(f)

    assert isinstance(asset_datasets, list), "Datasets must be a list"

    for dataset in asset_datasets:
        dataset_name = dataset.get("name")
        dataset_info = dataset.get("info")

        print(f"Downloading dataset {dataset_name}...")

        url = dataset_info.get("url")
        save_path = dataset_info.get("save_path")

        if not url:
            print(f"Dataset {dataset_name} has no URL. Skipping download.")
            continue

        if not save_path:
            save_path = f"{dataset_name}.zip"
        save_path = ROOT_DIR / save_path

        print(f"Downloading dataset from {url} and saving to {save_path}...")

        try:
            response = requests.get(url, allow_redirects=True)
            response.raise_for_status()
            with open(save_path, "wb") as file:
                file.write(response.content)
            datasets.append((save_path, None))
        except Exception as e:
            datasets.append((save_path, e))

    return datasets


def unzip_dataset(dataset: pathlib.Path) -> None:
    # Check first if the dataset is a zip file
    if not zipfile.is_zipfile(dataset):
        print(f"Dataset {dataset} is not a zip file. Skipping unzip.")
        return

    with zipfile.ZipFile(dataset, "r") as zip_ref:
        fpath = dataset.parent / dataset.stem
        fpath.mkdir(exist_ok=True)

        zip_ref.extractall(fpath)


if __name__ == "__main__":
    datasets = download_datasets()
    for dataset, error in datasets:
        if error is None:
            print(f"Successfully downloaded {dataset}. Unzipping...")
            unzip_dataset(dataset)

            print(f"Unzipped {dataset} successfully. Removing the zip file...")
            try:
                dataset.unlink()
                print(f"Removed {dataset} successfully.\n")
            except Exception as e:
                print(f"Failed to remove {dataset}: {e}\n")
        else:
            print(f"Failed to download {dataset}: {error}")
