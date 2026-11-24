"""Lab 7: wykresy przed i po czyszczeniu oraz wniosek o korelacji.

Czyszczenie jest tym samym pipeline co w raporcie lab 6. Wykres „przed”
bierze kopię surowych danych i tylko na czas rysowania zamienia tekst na braki.
Pliku surowego to nie zmienia. Wartości poza zakresem zostają w obu tabelach.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.charts import build_figure, draw_chart, opis
from src.session import DataSession

PLIK = ROOT / "california_housing_projekt2.csv"
WYJSCIE = ROOT / "outputs" / "lab7"
CLEANED_LAB6 = ROOT / "outputs" / "lab6" / "cleaned.csv"

# Te same kolumny tekstowe co w scripts/raport_przed_po.py.
KOLUMNY_TEKSTOWE = ["MedInc", "HouseAge", "Population", "MedHouseVal"]
KOLUMNY = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]
# Granice Kalifornii z reguł lab 6. Służą tylko do komentarza o współczynniku,
# nie do usuwania wierszy.
GRANICE_KALIFORNII = {"Latitude": (32.0, 42.0), "Longitude": (-124.5, -114.0)}
KRAWEDZIE_DOCHODU = [-4, 0, 2, 3, 4, 5, 6, 8, 15, 70]
RAMKA_X = (-1.2, 12.0)
RAMKA_Y = (-0.5, 5.3)

UWAGI_PAR = {
    frozenset(("MedInc", "MedHouseVal")): (
        "Wyższy dochód częściej siedzi w tym pliku obok wyższej wartości domu. "
        "To nie rozstrzyga, czy dochód podnosi cenę, czy droższe bloki mają "
        "mieszkańców o wyższym dochodzie, ani czy obie liczby zależą od lokalizacji."
    ),
    frozenset(("Latitude", "Longitude")): (
        "To współrzędne tego samego punktu. r nie znaczy, że szerokość wywołuje długość."
    ),
    frozenset(("AveRooms", "AveBedrms")): (
        "Obie kolumny liczą pomieszczenia w tym samym bloku. "
        "Związek bierze się ze sposobu policzenia zmiennych, a nie z wpływu jednej na drugą."
    ),
    frozenset(("HouseAge", "Population")): (
        "Starsze bloki bywają tu mniej liczne. To współwystępowanie w tabeli, "
        "nie mechanizm, w którym wiek domu zmniejsza liczbę mieszkańców."
    ),
}


def numeric_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """Kopia do rysowania. Tekst staje się brakiem, wiersze zostają."""
    converted = frame.copy()
    for column in converted.columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def prepare(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session = DataSession.from_csv(path)
    raw = session.raw.copy()
    before = numeric_copy(raw)
    session.drop_exact_duplicates()
    session.coerce_to_numeric(KOLUMNY_TEKSTOWE)
    gaps = [column for column in session.current.columns if session.current[column].isna().any()]
    if gaps:
        session.fill_numeric_median(gaps)
    after = session.current.copy()
    saved = CLEANED_LAB6
    if saved.exists():
        reference = pd.read_csv(saved)
        left = after.reset_index(drop=True)
        right = reference.reset_index(drop=True)
        same_shape = list(left.columns) == list(right.columns) and len(left) == len(right)
        if not same_shape:
            raise SystemExit(
                "Tabela po czyszczeniu ma inny kształt niż outputs/lab6/cleaned.csv. "
                "Wykresy mają pokazywać ten sam pipeline co lab 6."
            )
        gap = left.select_dtypes("number") - right.select_dtypes("number")
        if gap.abs().fillna(0).max().max() > 1e-6:
            raise SystemExit(
                "Tabela po czyszczeniu różni się od outputs/lab6/cleaned.csv. "
                "Wykresy mają pokazywać ten sam pipeline co lab 6."
            )
    return raw, before, after


def shared_range(before: pd.DataFrame, after: pd.DataFrame, column: str) -> tuple[float, float]:
    values = pd.concat([before[column], after[column]], ignore_index=True).dropna()
    low = float(values.min())
    high = float(values.max())
    if high == low:
        high = low + 1.0
    pad = (high - low) * 0.03
    return low - pad, high + pad


def chart_specs(before: pd.DataFrame, after: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "file": "histogram_przed_po.png",
            "kind": "histogram",
            "columns": ["MedHouseVal"],
            "suptitle": (
                "Rozkład mediany wartości domu (100 tys. USD)\n"
                "Słupek przy 5 to górny próg zbioru, nie efekt dochodu."
            ),
            "draw": {"bins": 40, "hist_range": shared_range(before, after, "MedHouseVal")},
        },
        {
            "file": "punktowy_przed_po.png",
            "kind": "scatter",
            "columns": ["MedInc", "MedHouseVal"],
            "suptitle": (
                "Dochód a wartość domu. Współwystępowanie, nie przyczyna.\n"
                "Oś dochodu od -1.2 do 12. Punkty poza ramką zostają w tabeli."
            ),
            "draw": {"xlim": RAMKA_X, "ylim": RAMKA_Y},
        },
        {
            "file": "pudelkowy_przed_po.png",
            "kind": "box",
            "columns": ["MedHouseVal"],
            "suptitle": "Rozrzut mediany wartości domu (100 tys. USD)",
            "draw": {},
        },
        {
            "file": "slupkowy_przed_po.png",
            "kind": "bar",
            "columns": ["MedInc", "MedHouseVal"],
            "suptitle": (
                "Średnia wartości domu w przedziałach dochodu.\n"
                "Wyższy słupek to wyższa średnia w tym pliku, nie przyczyna ceny."
            ),
            "draw": {"edges": KRAWEDZIE_DOCHODU, "ylim": (0, 5.6)},
        },
        {
            "file": "korelacja_przed_po.png",
            "kind": "correlation",
            "columns": KOLUMNY,
            "suptitle": "Korelacja Pearsona. Współczynnik opisuje współwystępowanie, nie przyczynę.",
            "draw": {},
        },
    ]


def save_pair(
    path: Path,
    before: pd.DataFrame,
    after: pd.DataFrame,
    kind: str,
    columns: list[str],
    suptitle: str,
    draw: dict[str, object],
) -> None:
    wide = kind == "correlation"
    fig = Figure(
        figsize=(13.6, 7.2) if wide else (12.8, 6.2),
        dpi=130,
        layout="constrained",
    )
    fig.set_facecolor("white")
    if wide:
        axes = fig.subplots(1, 2)
    else:
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
    stages = (("Przed czyszczeniem", before), ("Po czyszczeniu", after))
    for ax, (stage, frame) in zip(axes, stages):
        used = plotted_rows(frame, kind, columns)
        draw_chart(ax, frame, kind, columns, title=f"{stage} (n = {used})", **draw)
        ax.tick_params(labelleft=True, labelbottom=True)
    fig.suptitle(suptitle, fontsize=13, color="#1F2933")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plotted_rows(frame: pd.DataFrame, kind: str, columns: list[str]) -> int:
    if kind == "correlation":
        return len(frame)
    return int(frame[columns].dropna().shape[0])


def outside_frame(frame: pd.DataFrame) -> int:
    subset = frame[["MedInc", "MedHouseVal"]].dropna()
    mask = (
        (subset["MedInc"] < RAMKA_X[0])
        | (subset["MedInc"] > RAMKA_X[1])
        | (subset["MedHouseVal"] < RAMKA_Y[0])
        | (subset["MedHouseVal"] > RAMKA_Y[1])
    )
    return int(mask.sum())


def pair_rows(frame: pd.DataFrame) -> list[tuple[float, str, str, int]]:
    found: list[tuple[float, str, str, int]] = []
    columns = [column for column in KOLUMNY if column in frame.columns]
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            pair = frame[[left, right]].dropna()
            if len(pair) < 2 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            value = float(pair[left].corr(pair[right]))
            found.append((value, left, right, len(pair)))
    found.sort(key=lambda item: abs(item[0]), reverse=True)
    return found


def text_cells(raw: pd.DataFrame) -> int:
    total = 0
    for column in KOLUMNY_TEKSTOWE:
        numeric = pd.to_numeric(raw[column], errors="coerce")
        total += int((raw[column].notna() & numeric.isna()).sum())
    return total


def validation_lines(raw: pd.DataFrame) -> list[str]:
    lines = ["Sprawdzenie typów. Wykres w tych próbach nie powstaje:"]
    constant = pd.DataFrame({"Stala": [1, 1, 1, 1], "MedHouseVal": [1.0, 2.0, 1.5, 2.5]})
    attempts = [
        (raw, "histogram", ["MedInc"], "surowy MedInc, w pliku jest tekst"),
        (constant, "histogram", ["Stala"], "kolumna bez zmienności"),
        (
            constant,
            "scatter",
            ["MedHouseVal"],
            "wykres punktowy z jedną kolumną",
        ),
    ]
    for frame, kind, columns, label in attempts:
        try:
            build_figure(frame, kind, columns)
        except ValueError as error:
            lines.append(f"- {label}: {error}")
        else:
            raise SystemExit(f"Walidacja przepuściła przypadek: {label}.")
    return lines


def format_pair(value: float, left: str, right: str, count: int) -> list[str]:
    lines = [f"- {left} i {right}: r = {value:.3f}, n = {count}."]
    note = UWAGI_PAR.get(frozenset((left, right)))
    if note:
        lines.append(f"  {note}")
    return lines


def report(
    raw: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    files: list[str],
) -> str:
    income = pair_rows(before)
    income_after = pair_rows(after)
    before_map = {(left, right): (value, count) for value, left, right, count in income}
    after_map = {(left, right): (value, count) for value, left, right, count in income_after}
    key = ("MedInc", "MedHouseVal")
    r_before, n_before = before_map[key]
    r_after, n_after = after_map[key]

    geo = after[
        after["Latitude"].between(*GRANICE_KALIFORNII["Latitude"])
        & after["Longitude"].between(*GRANICE_KALIFORNII["Longitude"])
    ]
    r_geo = float(geo["Latitude"].corr(geo["Longitude"]))
    r_geo_full = after_map[("Latitude", "Longitude")][0]
    outside_before = outside_frame(before)
    outside_after = outside_frame(after)
    nonpositive_income = int(before["MedInc"].dropna().le(0).sum())

    lines = [
        "=== LAB 7. WYKRESY PRZED I PO ===",
        f"Plik: {PLIK.name}",
        f"Wiersze surowe: {len(raw)}",
        f"Wiersze po czyszczeniu: {len(after)}",
        f"Tekst zamieniony na braki na kopii do wykresu „przed”: {text_cells(raw)} komórek.",
        "Ta kopia nie zapisuje się z powrotem do pliku surowego.",
        "",
        "Pipeline „po” jest ten sam co w lab 6:",
        "1. Usuń pełne duplikaty.",
        "2. W MedInc, HouseAge, Population i MedHouseVal zamień tekst typu ?, error, brak na braki.",
        "3. Uzupełnij braki medianą tylko w kolumnach, które nadal mają puste komórki.",
        "4. Zakres i IQR nie kasują wierszy. Ujemna liczba i współrzędna poza Kalifornią zostają.",
        "",
        *validation_lines(raw),
        "",
        "Zestaw plików:",
        *[f"- {name}" for name in files],
        "",
        "Osie i tytuły:",
    ]
    for column in ("MedInc", "MedHouseVal", "Latitude", "Longitude"):
        lines.append(f"- {column}: {opis(column)}")
    lines.extend(
        [
            "",
            "=== CO WIDAĆ ===",
            "Histogram wartości domu ma skupienie przy prawej krawędzi, około 5. "
            "To górny próg tego zbioru, a nie skutek dochodu. Kilka wartości poniżej zera "
            "zostaje, bo reguła zakresu ich nie usuwa. Po czyszczeniu słupki są niższe "
            "tam, gdzie zniknęły duplikaty, a środek może lekko urosnąć, bo mediana "
            "wchodzi w puste komórki.",
            "",
            "Wykres punktowy: dochód na osi X, wartość domu na osi Y. "
            f"Poza ramką zostaje {outside_before} punktów przed czyszczeniem "
            f"i {outside_after} po czyszczeniu. Ramka jest po to, żeby skrajny dochód "
            "nie spłaszczył chmury. Te punkty są w tabeli i wchodzą do r.",
            "",
            "Wykres pudełkowy wartości domu pokazuje medianę, kwartyle i punkty poza wąsami, "
            "w tym wartości ujemne. Pudełko nie mówi, skąd się wzięła cena.",
            "",
            "Wykres słupkowy grupuje dochód w te same przedziały przed i po. "
            f"Pierwszy kosz, od -4 do 0, ma przed czyszczeniem {nonpositive_income} wierszy. "
            "To błędne liczby, w tym dochody poniżej zera, a nie grupa gospodarstw. "
            "Ostatni kosz miesza wysokie dochody ze skrajnymi. "
            "Średnia w koszu opisuje ten plik. Nie jest mechanizmem ceny.",
            "",
            "=== KORELACJA ===",
            f"MedInc i MedHouseVal przed czyszczeniem: r = {r_before:.3f}, n = {n_before}.",
            f"MedInc i MedHouseVal po czyszczeniu: r = {r_after:.3f}, n = {n_after}.",
            "Przed czyszczeniem r liczy się na wierszach, w których obie kolumny są liczbą. "
            "Po czyszczeniu braki są już medianą, więc do pary wchodzi więcej wierszy. "
            "Zmiana r bierze się z tej różnicy zbioru i z usuniętych duplikatów. "
            "Nie jest odkryciem przyczyny.",
            "",
            "Najsilniejsze pary po czyszczeniu, liczone Pearsonem:",
        ]
    )
    for item in income_after[:5]:
        lines.extend(format_pair(*item))
    lines.extend(
        [
            "",
            f"Latitude i Longitude na całej tabeli po czyszczeniu: r = {r_geo_full:.3f}.",
            f"To samo r tylko dla współrzędnych w granicach Kalifornii "
            f"(szerokość 32–42, długość -124.5–-114): r = {r_geo:.3f}, n = {len(geo)}.",
            "To przeliczenie nie jest czyszczeniem i nie zmienia wykresu ani pliku. "
            "Pokazuje, że kilka punktów daleko poza stanem spłaszcza współczynnik. "
            "Nawet r bliskie -1 nie znaczy, że jedna współrzędna jest przyczyną drugiej "
            "ani przyczyną ceny.",
            "",
            "AveOccup ma w tym pliku prawie zerowe r z wartością domu. "
            "W kolumnie siedzą skrajne liczby, rzędu tysięcy osób na dom. "
            "Pearson jest na nie wrażliwy. Zerowe r nie jest wnioskiem, "
            "że zagęszczenie w rzeczywistości nie łączy się z ceną.",
            "",
            "=== CZEGO TU NIE MA ===",
            "Żaden wykres ani współczynnik z tego zestawu nie pokazuje przyczyny. "
            "Korelacja i nachylenie chmury mówią tylko, które kolumny w tym pliku "
            "zmieniają się razem. Dochód i cena mogą iść razem, bo tak wygląda okolica, "
            "bo tak zebrano dane albo z obu tych powodów naraz. Tego z samego r nie widać. "
            "Mediana wstawiona w braki dosuwa puste komórki do środka rozkładu "
            "i może przesunąć r bez nowego mechanizmu.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not PLIK.exists():
        raise SystemExit(f"Nie znaleziono pliku {PLIK.name}. Włóż go do katalogu projektu.")

    raw, before, after = prepare(PLIK)
    WYJSCIE.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for spec in chart_specs(before, after):
        file_name = str(spec["file"])
        save_pair(
            WYJSCIE / file_name,
            before,
            after,
            str(spec["kind"]),
            list(spec["columns"]),
            str(spec["suptitle"]),
            dict(spec["draw"]),
        )
        written.append(file_name)

    text = report(raw, before, after, written)
    report_path = WYJSCIE / "wnioski.txt"
    report_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Zapisano {report_path}")
    for name in written:
        print(f"Zapisano {WYJSCIE / name}")


if __name__ == "__main__":
    main()
