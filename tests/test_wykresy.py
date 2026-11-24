import matplotlib

matplotlib.use("Agg")

from io import BytesIO

import pandas as pd
import pytest

from src.charts import build_figure


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MedInc": [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
            "MedHouseVal": [0.8, 1.1, 1.4, 1.8, 2.2, 2.8, 3.4, 4.1],
            "HouseAge": [5, 10, 15, 20, 25, 30, 40, 50],
        }
    )


def test_histogram_has_readable_labels() -> None:
    fig = build_figure(frame(), "histogram", ["MedHouseVal"])
    ax = fig.axes[0]
    assert "Mediana wartości domu" in ax.get_xlabel()
    assert ax.get_ylabel() == "Liczba rekordów"
    assert "Rozkład" in ax.get_title()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    assert buffer.getvalue().startswith(b"\x89PNG")


def test_text_column_is_rejected() -> None:
    dirty = pd.DataFrame({"MedInc": ["1.2", "?", "brak", "4.0"]})
    with pytest.raises(ValueError, match="nie jest liczbowa"):
        build_figure(dirty, "histogram", ["MedInc"])


def test_constant_column_is_rejected() -> None:
    stuck = pd.DataFrame({"Stala": [3, 3, 3, 3], "MedHouseVal": [1.0, 1.5, 2.0, 2.5]})
    with pytest.raises(ValueError, match="mniej niż dwie różne wartości"):
        build_figure(stuck, "box", ["Stala"])


def test_scatter_needs_two_different_columns() -> None:
    data = frame()
    with pytest.raises(ValueError, match="dwóch różnych kolumn"):
        build_figure(data, "scatter", ["MedInc"])
    with pytest.raises(ValueError, match="ta sama kolumna"):
        build_figure(data, "scatter", ["MedInc", "MedInc"])


def test_scatter_and_correlation_do_not_claim_cause() -> None:
    data = frame()
    scatter = build_figure(data, "scatter", ["MedInc", "MedHouseVal"])
    matrix = build_figure(data, "correlation", ["MedInc", "MedHouseVal", "HouseAge"])
    assert "nie przyczyna" in scatter.axes[0].get_title()
    assert "nie jest przyczyną" in matrix.axes[0].get_title()
    assert matrix.axes[0].get_xlabel() == "Zmienna"
    assert matrix.axes[0].get_ylabel() == "Zmienna"


def test_four_chart_types_and_unknown_kind() -> None:
    data = frame()
    specs = [
        ("histogram", ["HouseAge"]),
        ("scatter", ["MedInc", "MedHouseVal"]),
        ("box", ["MedHouseVal"]),
        ("bar", ["MedInc", "MedHouseVal"]),
    ]
    for kind, columns in specs:
        fig = build_figure(data, kind, columns)
        assert fig.axes[0].get_title()
        assert fig.axes[0].get_xlabel()
        assert fig.axes[0].get_ylabel()
    with pytest.raises(ValueError, match="Nieznany typ wykresu"):
        build_figure(data, "pie", ["MedInc"])


def test_bar_rejects_too_many_categories() -> None:
    wide = pd.DataFrame(
        {
            "nazwa": [f"k{i}" for i in range(20)],
            "wartosc": list(range(20)),
        }
    )
    with pytest.raises(ValueError, match="nieczytelny"):
        build_figure(wide, "bar", ["nazwa", "wartosc"])


def test_missing_column_names_the_file_columns() -> None:
    with pytest.raises(ValueError, match="MedInc"):
        build_figure(frame(), "histogram", ["NieMa"])
