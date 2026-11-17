import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.session import DataSession

PLIK = ROOT / "california_housing_projekt2.csv"
WYJSCIE = ROOT / "outputs" / "lab6"
KOLUMNY_TEKSTOWE = ["MedInc", "HouseAge", "Population", "MedHouseVal"]
KOLUMNY_LICZBOWE = [
    "AveRooms",
    "AveBedrms",
    "AveOccup",
    "Latitude",
    "Longitude",
]
# Dolna granica jest włączona: wartość równa granicy zostaje.
# Górna granica None oznacza, że górnego progu nie stosujemy.
ZAKRESY: dict[str, tuple[float, float | None]] = {
    "MedInc": (0, None),
    "HouseAge": (1, 52),
    "AveRooms": (0, None),
    "AveBedrms": (0, None),
    "Population": (0, None),
    "AveOccup": (0, None),
    "Latitude": (32, 42),
    "Longitude": (-124.5, -114),
    "MedHouseVal": (0, None),
}
KOLUMNY_IQR = list(ZAKRESY)


def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def existing_numbers_unchanged(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["Komórki, które już były liczbą:"]
    for column in columns:
        old = pd.to_numeric(before[column], errors="coerce")
        new = pd.to_numeric(after[column], errors="coerce")
        known = old.notna()
        same = old[known].reset_index(drop=True).equals(new[known].reset_index(drop=True))
        lines.append(f"- {column}: bez zmian poza brakami: {same}")
        if not same:
            raise SystemExit(f"Kolumna {column} zmieniła istniejące liczby.")
    return lines


def report_before(session: DataSession) -> list[str]:
    lines = [
        "=== PRZED CZYSZCZENIEM ===",
        f"Plik: {PLIK.name}",
        f"Wiersze: {len(session.current)}",
        f"Kolumny: {len(session.current.columns)}",
        "",
        "Profil kolumn:",
        session.profile().to_string(),
        "",
        "Statystyki liczbowe po odczytaniu liczb z tekstu (tabela robocza bez zmian):",
        numeric_frame(session.current).describe().to_string(),
        "",
        f"Nadmiarowe pełne duplikaty: {int(session.current.duplicated().sum())}",
        f"Wiersze biorące udział w duplikacie: {int(session.current.duplicated(keep=False).sum())}",
        "",
        "Braki:",
    ]
    for column in session.current.columns:
        missing = int(session.current[column].isna().sum())
        lines.append(f"- {column}: {missing}")

    lines.extend(["", "Nieliczbowe wpisy (tabela bez zmian):"])
    for column in KOLUMNY_TEKSTOWE:
        numeric = pd.to_numeric(session.current[column], errors="coerce")
        bad = session.current[column].notna() & numeric.isna()
        examples = session.current.loc[bad, column].astype(str).value_counts()
        shown = ", ".join(f"{name} ({count})" for name, count in examples.items())
        lines.append(f"- {column}: {int(bad.sum())} komórek; {shown}")

    lines.extend(["", "Wartości poza zakresem (tabela bez zmian):"])
    numbers = numeric_frame(session.current)
    for column, (low, high) in ZAKRESY.items():
        mask = pd.Series(False, index=numbers.index)
        if low is not None:
            mask = mask | (numbers[column] < low)
        if high is not None:
            mask = mask | (numbers[column] > high)
        upper = "bez górnej granicy" if high is None else str(high)
        lines.append(f"- {column} od {low} do {upper}: {int(mask.fillna(False).sum())} wierszy")

    lines.extend(["", "Wartości odstające IQR (tabela bez zmian, wiersze zostają):"])
    for column in KOLUMNY_IQR:
        lines.append(f"- {session.describe_iqr(column)}")
    return lines


def main() -> None:
    if not PLIK.exists():
        raise SystemExit(f"Nie znaleziono pliku {PLIK.name}. Włóż go do katalogu projektu.")

    session = DataSession.from_csv(PLIK)
    lines = report_before(session)
    lines.extend(
        [
            "",
            "=== PIPELINE ===",
            "1. Usuń pełne duplikaty.",
            "2. W MedInc, HouseAge, Population i MedHouseVal zamień tekst typu ?, error, brak na braki.",
            "3. Uzupełnij braki medianą tylko w kolumnach, które nadal mają puste komórki.",
            "4. Zakres i IQR tylko opisz. Wierszy nie usuwaj: ujemna liczba albo współrzędna poza Kalifornią",
            "   zostaje w pliku, bo sam raport nie jest decyzją o kasowaniu.",
            "",
        ]
    )

    session.drop_exact_duplicates()
    before_coerce = session.current.copy()
    session.coerce_to_numeric(KOLUMNY_TEKSTOWE)
    lines.extend(existing_numbers_unchanged(before_coerce, session.current, KOLUMNY_TEKSTOWE + KOLUMNY_LICZBOWE))

    before_fill = session.current.copy()
    columns_with_gaps = [
        column for column in session.current.columns if session.current[column].isna().any()
    ]
    session.fill_numeric_median(columns_with_gaps)
    lines.extend(["", *existing_numbers_unchanged(before_fill, session.current, columns_with_gaps)])

    lines.extend(["", "=== DZIENNIK ==="])
    lines.extend(entry.text() for entry in session.audit_log)
    lines.extend(
        [
            "",
            "=== PO CZYSZCZENIU ===",
            f"Wiersze: {len(session.current)}",
            f"Kolumny: {len(session.current.columns)}",
            f"Braki łącznie: {int(session.current.isna().sum().sum())}",
            "",
            "Profil kolumn:",
            session.profile().to_string(),
            "",
            "Statystyki liczbowe:",
            session.current.describe().to_string(),
            "",
            "Poza zakresem i odstające wartości nadal są w tabeli.",
        ]
    )

    WYJSCIE.mkdir(parents=True, exist_ok=True)
    report_path = WYJSCIE / "raport_przed_po.txt"
    cleaned_path = WYJSCIE / "cleaned.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    session.current.to_csv(cleaned_path, index=False)
    print("\n".join(lines))
    print(f"\nZapisano {report_path}")
    print(f"Zapisano {cleaned_path}")


if __name__ == "__main__":
    main()
