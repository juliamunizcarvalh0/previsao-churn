import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """Treina o modelo no dataset real numa pasta temporária, uma vez por sessão."""
    from churn.train import train

    out = tmp_path_factory.mktemp("artefatos")
    metrics = train(seed=7, out_dir=out)
    return out, metrics
