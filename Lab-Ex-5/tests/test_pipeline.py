import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare import load_dataset  # noqa: E402


def test_load_dataset_shape():
    df = load_dataset()
    assert "MEDV" in df.columns
    assert len(df) > 0
    assert df["MEDV"].nunique() > 2


def test_prepare_script_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(
        "prepare:\n  test_size: 0.2\n  random_state: 42\n"
    )
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    source_csv = Path(__file__).resolve().parents[1] / "data" / "raw" / "HousingData.csv"
    (raw_dir / "HousingData.csv").write_bytes(source_csv.read_bytes())
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "src" / "prepare.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "data" / "train.csv").exists()
    assert (tmp_path / "data" / "test.csv").exists()
