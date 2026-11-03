"""Shared helpers for the ML projects.

The same file is used in two places:
- inside the portfolio (labos-portfolio/ml/): results go to src/data/runs/<id>.json, which the
  site's Model Lab reads, and the digit model goes to public/models/;
- in each project's own repo: results go to results/<id>.json plus a chart, results/<id>.png.

Datasets are downloaded into data/ the first time a script needs them.
"""
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
IN_PORTFOLIO = (HERE.parent / "src" / "data" / "runs").is_dir()
RUNS = HERE.parent / "src" / "data" / "runs" if IN_PORTFOLIO else HERE / "results"
MODELS = HERE.parent / "public" / "models" if IN_PORTFOLIO else HERE / "model"

SOURCES = {
    "cal_housing": "https://ndownloader.figshare.com/files/5976036",  # cal_housing.tgz
    "sms": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "wholesale": "https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip",
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "mnist.npz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
}


def dataset(name):
    """Path to data/<name>, downloading and unpacking it first if it isn't there yet."""
    path = DATA / name
    if path.exists():
        return path
    print(f"downloading {name} ...")
    req = urllib.request.Request(SOURCES[name], headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=300).read()
    DATA.mkdir(exist_ok=True)
    if name.endswith(".npz"):
        path.write_bytes(raw)
    elif raw[:2] == b"PK":
        zipfile.ZipFile(io.BytesIO(raw)).extractall(path)
    else:
        tarfile.open(fileobj=io.BytesIO(raw)).extractall(path, filter="data")
    return path


def r(x, nd=4):
    """Round floats (and lists of floats) for a compact JSON file."""
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    return round(float(x), nd)


def save_result(run_id, result):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{run_id}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {path.relative_to(HERE.parent)}")
    if not IN_PORTFOLIO:
        save_chart(run_id, result)


def save_chart(run_id, result):
    """Draw the result's curve (the same one the portfolio shows) as results/<id>.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = result["curve"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(c["x"], c["train"], marker="o" if len(c["x"]) <= 12 else None, color="#e8a33d", label=c["trainLabel"])
    if c.get("val"):
        ax.plot(c["x"], c["val"], marker="o" if len(c["x"]) <= 12 else None, color="#4fb7c9", label=c["valLabel"])
    ax.set(xlabel=c["xLabel"], ylabel=c["yLabel"], title=f'{result["metric"]["label"]}: {result["metric"]["value"]}')
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = RUNS / f"{run_id}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.relative_to(HERE.parent)}")
