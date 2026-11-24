"""Wykresy Matplotlib i Seaborn budowane z parametrów.

Funkcje zwracają albo rysują matplotlib.figure.Figure, żeby ten sam wykres
dało się zapisać do pliku i później osadzić w oknie.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

KINDS = ("histogram", "scatter", "box", "bar", "correlation")
SINGLE = {"histogram", "box"}
PAIR = {"scatter", "bar"}
MAX_BAR_LEVELS = 12
MAX_CORRELATION_COLUMNS = 12

KOLOR = "#3B6D8C"

OPISY = {
    "MedInc": "Mediana dochodu (10 tys. USD)",
    "HouseAge": "Wiek domu (lata)",
    "AveRooms": "Średnia liczba pokoi",
    "AveBedrms": "Średnia liczba sypialni",
    "Population": "Liczba mieszkańców bloku",
    "AveOccup": "Średnia liczba osób w domu",
    "Latitude": "Szerokość geograficzna",
    "Longitude": "Długość geograficzna",
    "MedHouseVal": "Mediana wartości domu (100 tys. USD)",
}

KROTKO = {
    "MedInc": "Dochód",
    "HouseAge": "Wiek domu",
    "AveRooms": "Pokoje",
    "AveBedrms": "Sypialnie",
    "Population": "Populacja",
    "AveOccup": "Mieszkańcy",
    "Latitude": "Szer. geogr.",
    "Longitude": "Dł. geogr.",
    "MedHouseVal": "Wartość domu",
}

NAZWA_TYPU = {
    "histogram": "Histogram",
    "scatter": "Wykres punktowy",
    "box": "Wykres pudełkowy",
    "bar": "Wykres słupkowy",
    "correlation": "Korelacja",
}


def opis(column: str) -> str:
    """Czytelna nazwa osi. Nieznana kolumna zostaje przy swojej nazwie."""
    return OPISY.get(column, str(column))


def krotko(column: str) -> str:
    """Krótka etykieta do macierzy, gdzie pełna nazwa by się nie zmieściła."""
    return KROTKO.get(column, str(column))


def build_figure(
    frame: pd.DataFrame,
    kind: str,
    columns: Sequence[str],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    bins: int = 30,
    groups: int = 5,
    edges: Sequence[float] | None = None,
    hist_range: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Zbuduj jeden wykres. Rodzaj i kolumny przychodzą jako parametry."""
    wide = kind == "correlation"
    fig = Figure(figsize=(9.6, 7.6) if wide else (8.4, 5.3), dpi=130, layout="constrained")
    fig.set_facecolor("white")
    ax = fig.add_subplot(111)
    draw_chart(
        ax,
        frame,
        kind,
        columns,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        bins=bins,
        groups=groups,
        edges=edges,
        hist_range=hist_range,
        xlim=xlim,
        ylim=ylim,
    )
    return fig


def draw_chart(
    ax: Axes,
    frame: pd.DataFrame,
    kind: str,
    columns: Sequence[str],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    bins: int = 30,
    groups: int = 5,
    edges: Sequence[float] | None = None,
    hist_range: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Narysuj wykres na podanej osi. Walidacja jest ta sama co w build_figure."""
    names = _validate(
        frame,
        kind,
        columns,
        bins=bins,
        groups=groups,
        edges=edges,
        hist_range=hist_range,
        xlim=xlim,
        ylim=ylim,
    )
    if kind == "histogram":
        _histogram(ax, frame, names[0], bins, hist_range, xlabel, ylabel)
    elif kind == "scatter":
        _scatter(ax, frame, names[0], names[1], xlabel, ylabel)
    elif kind == "box":
        _box(ax, frame, names[0], xlabel, ylabel)
    elif kind == "bar":
        _bar(ax, frame, names[0], names[1], groups, edges, xlabel, ylabel)
    else:
        _correlation(ax, frame, names, xlabel, ylabel)

    ax.set_title(title or _default_title(kind, names), fontsize=12, pad=8, color="#1F2933")
    if xlim is not None and kind != "correlation":
        ax.set_xlim(xlim)
    if ylim is not None and kind != "correlation":
        ax.set_ylim(ylim)
    _grid(ax, kind)


def _validate(
    frame: pd.DataFrame,
    kind: str,
    columns: Sequence[str],
    *,
    bins: int,
    groups: int,
    edges: Sequence[float] | None,
    hist_range: tuple[float, float] | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> list[str]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Wykres przyjmuje tabelę danych (DataFrame).")
    if frame.empty:
        raise ValueError("Tabela jest pusta. Wczytaj plik, w którym są wiersze.")
    if kind not in KINDS:
        choices = ", ".join(KINDS)
        raise ValueError(f"Nieznany typ wykresu: {kind}. Wybierz: {choices}.")

    names = [str(name) for name in columns]
    if not names:
        raise ValueError("Wybierz co najmniej jedną kolumnę.")
    if kind in SINGLE and len(names) != 1:
        raise ValueError(
            f"{NAZWA_TYPU[kind]} używa jednej kolumny. Podaj jedną kolumnę liczbową."
        )
    if kind in PAIR and len(names) != 2:
        raise ValueError(
            f"{NAZWA_TYPU[kind]} potrzebuje dwóch różnych kolumn: oś X i oś Y."
        )
    if kind == "correlation" and len(names) < 2:
        raise ValueError("Korelacja potrzebuje co najmniej dwóch kolumn liczbowych.")
    if kind == "correlation" and len(names) > MAX_CORRELATION_COLUMNS:
        raise ValueError(
            f"Macierz dla {len(names)} kolumn będzie nieczytelna. "
            f"Wybierz najwyżej {MAX_CORRELATION_COLUMNS} kolumn."
        )
    if kind in PAIR and names[0] == names[1]:
        raise ValueError(
            f"Oś X i oś Y to ta sama kolumna ({names[0]}). Wybierz dwie różne kolumny."
        )
    if len(names) != len(set(names)):
        raise ValueError("Kolumny na wykresie powtarzają się. Podaj każdą nazwę raz.")

    missing = [name for name in names if name not in frame.columns]
    if missing:
        available = ", ".join(str(name) for name in frame.columns)
        raise ValueError(
            f"Nie ma kolumny {', '.join(missing)}. Wybierz nazwę z pliku: {available}."
        )

    if kind == "bar":
        _require_bar_columns(frame, names[0], names[1])
    else:
        for name in names:
            _require_numeric(frame, name, NAZWA_TYPU[kind])
            _require_variation(frame, name)

    _require_complete_rows(frame, names)
    _require_limits("Liczba koszy histogramu", bins, minimum=2)
    _require_limits("Liczba przedziałów słupków", groups, minimum=2)
    if edges is not None:
        _require_edges(edges)
    if hist_range is not None:
        _require_span("Zakres histogramu", hist_range)
    if xlim is not None:
        _require_span("Zakres osi X", xlim)
    if ylim is not None:
        _require_span("Zakres osi Y", ylim)
    return names


def _require_bar_columns(frame: pd.DataFrame, x: str, y: str) -> None:
    _require_numeric(frame, y, "Wykres słupkowy")
    _require_variation(frame, y)
    if pd.api.types.is_numeric_dtype(frame[x]):
        _require_variation(frame, x)
        return
    levels = frame[x].dropna().nunique()
    if levels < 2:
        raise ValueError(
            f"Kolumna {x} ma mniej niż dwa poziomy. "
            "Wykres słupkowy potrzebuje porównania. Wybierz inną kolumnę."
        )
    if levels > MAX_BAR_LEVELS:
        raise ValueError(
            f"Kolumna {x} ma {levels} poziomów. Wykres słupkowy byłby nieczytelny. "
            "Wybierz kolumnę z mniejszą liczbą kategorii albo kolumnę liczbową do podziału na przedziały."
        )


def _require_numeric(frame: pd.DataFrame, column: str, chart: str) -> None:
    if pd.api.types.is_numeric_dtype(frame[column]):
        return
    dtype = frame[column].dtype
    raise ValueError(
        f"Kolumna {column} nie jest liczbowa, ma typ {dtype}. "
        f"{chart} rysuje się z liczb. "
        "Wybierz kolumnę liczbową albo zamień tekst na liczby."
    )


def _require_variation(frame: pd.DataFrame, column: str) -> None:
    if frame[column].nunique(dropna=True) >= 2:
        return
    raise ValueError(
        f"Kolumna {column} ma mniej niż dwie różne wartości. "
        "Z takiej kolumny nie powstanie czytelny wykres ani korelacja. "
        "Wybierz kolumnę, która się zmienia."
    )


def _require_complete_rows(frame: pd.DataFrame, columns: list[str]) -> None:
    complete = int(frame[columns].dropna().shape[0])
    if complete >= 2:
        return
    names = ", ".join(columns)
    raise ValueError(
        f"W kolumnach {names} zostały mniej niż dwie kompletne wartości. "
        "Uzupełnij braki albo wybierz inne kolumny."
    )


def _require_limits(label: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} musi być liczbą całkowitą co najmniej {minimum}.")


def _require_edges(edges: Sequence[float]) -> None:
    if len(edges) < 3:
        raise ValueError(
            "Do słupków potrzeba co najmniej dwóch przedziałów, czyli trzech krawędzi."
        )
    numbers = [float(edge) for edge in edges]
    if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
        raise ValueError("Krawędzie przedziałów muszą rosnąć. Popraw kolejność liczb.")


def _require_span(label: str, span: tuple[float, float]) -> None:
    if len(span) != 2 or float(span[0]) >= float(span[1]):
        raise ValueError(f"{label} jest odwrócony albo pusty. Podaj mniejszą liczbę, potem większą.")


def _default_title(kind: str, columns: list[str]) -> str:
    if kind == "histogram":
        return f"Rozkład: {opis(columns[0])}"
    if kind == "scatter":
        return (
            f"{opis(columns[1])} względem {opis(columns[0])}. "
            "Współwystępowanie, nie przyczyna."
        )
    if kind == "box":
        return f"Rozrzut: {opis(columns[0])}"
    if kind == "bar":
        return (
            f"Średnia wartości w przedziałach. "
            f"{opis(columns[1])} według {opis(columns[0])}."
        )
    return "Korelacja Pearsona. Współczynnik nie jest przyczyną."


def _histogram(
    ax: Axes,
    frame: pd.DataFrame,
    column: str,
    bins: int,
    hist_range: tuple[float, float] | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    sns.histplot(
        x=frame[column].dropna(),
        bins=bins,
        binrange=hist_range,
        stat="count",
        element="bars",
        color=KOLOR,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_xlabel(xlabel or opis(column), fontsize=10)
    ax.set_ylabel(ylabel or "Liczba rekordów", fontsize=10)


def _scatter(
    ax: Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    sns.scatterplot(
        data=frame[[x, y]].dropna(),
        x=x,
        y=y,
        ax=ax,
        color=KOLOR,
        s=14,
        alpha=0.28,
        linewidth=0,
        legend=False,
    )
    ax.set_xlabel(xlabel or opis(x), fontsize=10)
    ax.set_ylabel(ylabel or opis(y), fontsize=10)


def _box(
    ax: Axes,
    frame: pd.DataFrame,
    column: str,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    sns.boxplot(
        y=frame[column].dropna(),
        ax=ax,
        color=KOLOR,
        width=0.32,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35, "linewidth": 0},
    )
    ax.set_xlabel(xlabel or "Wszystkie rekordy", fontsize=10)
    ax.set_ylabel(ylabel or opis(column), fontsize=10)


def _bar(
    ax: Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    groups: int,
    edges: Sequence[float] | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    subset = frame[[x, y]].dropna()
    if pd.api.types.is_numeric_dtype(subset[x]):
        labels, values, counts = _numeric_bars(subset, x, y, groups, edges)
        axis_label = xlabel or f"Przedziały: {opis(x)}"
    else:
        grouped = subset.groupby(subset[x].astype(str), observed=True)[y]
        means = grouped.mean().sort_index()
        labels = [str(label) for label in means.index]
        values = [float(value) for value in means.to_numpy()]
        counts = [int(value) for value in grouped.count().reindex(means.index).to_numpy()]
        axis_label = xlabel or opis(x)

    kept = [
        (label, value, count)
        for label, value, count in zip(labels, values, counts)
        if count > 0 and pd.notna(value)
    ]
    if not kept:
        raise ValueError(
            f"W kolumnach {x} i {y} nie zostało nic do słupków. "
            "Sprawdź przedziały albo wybierz inne kolumny."
        )
    labels, values, counts = (list(part) for part in zip(*kept))
    plot_frame = pd.DataFrame({"przedzial": labels, "srednia": values})
    sns.barplot(
        data=plot_frame,
        x="przedzial",
        y="srednia",
        order=labels,
        color=KOLOR,
        errorbar=None,
        ax=ax,
    )
    _annotate_counts(ax, counts)
    ax.set_xlabel(axis_label, fontsize=10)
    ax.set_ylabel(ylabel or f"Średnia: {opis(y)}", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def _numeric_bars(
    subset: pd.DataFrame,
    x: str,
    y: str,
    groups: int,
    edges: Sequence[float] | None,
) -> tuple[list[str], list[float], list[int]]:
    if edges is None:
        bucket_count = min(groups, int(subset[x].nunique(dropna=True)))
        buckets = pd.qcut(subset[x], q=bucket_count, duplicates="drop")
    else:
        buckets = pd.cut(subset[x], bins=[float(edge) for edge in edges], include_lowest=True)
    grouped = subset.groupby(buckets, observed=False)[y]
    means = grouped.mean()
    if edges is None:
        labels = [_interval_label(interval) for interval in means.index]
    else:
        # include_lowest przesuwa lewą krawędź pierwszego kosza o ułamek.
        # Na osi zostają liczby podane w parametrach.
        labels = [
            f"{_fmt_edge(left)} do {_fmt_edge(right)}"
            for left, right in zip(edges[:-1], edges[1:])
        ]
    values = [float(value) if pd.notna(value) else float("nan") for value in means.to_numpy()]
    counts = [int(value) for value in grouped.count().to_numpy()]
    if len(labels) != len(values):
        labels = [_interval_label(interval) for interval in means.index]
    return labels, values, counts


def _interval_label(interval: pd.Interval) -> str:
    return f"{_fmt_edge(interval.left)} do {_fmt_edge(interval.right)}"


def _annotate_counts(ax: Axes, counts: list[int]) -> None:
    heights = [patch.get_height() for patch in ax.patches]
    finite = [height for height in heights if pd.notna(height)]
    if not finite:
        return
    span = max(finite) - min(0.0, min(finite))
    offset = span * 0.02 if span else 0.02
    for patch, count in zip(ax.patches, counts):
        height = patch.get_height()
        if count <= 0 or pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + offset,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#334155",
        )


def _fmt_edge(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-8:
        return str(int(round(number)))
    return f"{number:.1f}"


def _correlation(
    ax: Axes,
    frame: pd.DataFrame,
    columns: list[str],
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    matrix = frame[columns].corr(method="pearson")
    labels = [krotko(column) for column in columns]
    font_size = 8 if len(columns) <= 9 else 7
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": font_size},
        linewidths=0.4,
        linecolor="white",
        square=True,
        cbar_kws={"label": "r Pearsona", "shrink": 0.82},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xlabel(xlabel or "Zmienna", fontsize=10)
    ax.set_ylabel(ylabel or "Zmienna", fontsize=10)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def _grid(ax: Axes, kind: str) -> None:
    if kind == "correlation":
        return
    axis = "both" if kind == "scatter" else "y"
    ax.grid(True, axis=axis, color="#D0D5DD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
