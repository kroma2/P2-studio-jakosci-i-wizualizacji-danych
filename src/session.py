from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class AuditEntry:
    """Jeden wpis dziennika."""

    timestamp: datetime
    operation: str
    parameters: dict[str, object]
    rows_before: int
    rows_after: int
    affected: int
    message: str

    def text(self) -> str:
        moment = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{moment} | {self.message} "
            f"(przed: {self.rows_before}, po: {self.rows_after}, "
            f"zmiany: {self.affected})"
        )


@dataclass
class DataSession:
    """Sesja zapisu danych."""

    raw: pd.DataFrame
    current: pd.DataFrame = field(init=False)
    audit_log: list[AuditEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.raw.empty:
            raise ValueError("Plik nie zawiera danych.")
        self.current = self.raw.copy(deep=True)

    def record(
        self,
        operation: str,
        parameters: dict[str, object],
        rows_before: int,
        rows_after: int,
        affected: int,
        message: str,
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=datetime.now(),
            operation=operation,
            parameters=parameters,
            rows_before=rows_before,
            rows_after=rows_after,
            affected=affected,
            message=message,
        )
        self.audit_log.append(entry)
        return entry

    @classmethod
    def from_csv(cls, path: Path) -> "DataSession":
        """Wczytaj dane z pliku CSV."""
        frame = pd.read_csv(path, sep=None, engine="python")
        return cls(frame)

    def profile(self) -> pd.DataFrame:
        """Profil danych."""
        return pd.DataFrame(
            {
                "dtype": self.current.dtypes.astype(str),
                "missing": self.current.isna().sum(),
                "missing_pct": self.current.isna().mean().mul(100).round(2),
                "unique": self.current.nunique(dropna=True),
            }
        )

    def drop_exact_duplicates(self) -> int:
        """Usuń dokładne duplikaty."""
        before = len(self.current)
        self.current = self.current.drop_duplicates().copy()
        removed = before - len(self.current)
        self.record(
            operation="drop_exact_duplicates",
            parameters={},
            rows_before=before,
            rows_after=len(self.current),
            affected=removed,
            message=f"Usunięto dokładne duplikaty: {removed}",
        )
        return removed

    def coerce_to_numeric(self, columns: list[str]) -> int:
        """Zamień nieliczbowy tekst na braki. Istniejących liczb nie rusza."""
        if not columns:
            raise ValueError("Wybierz co najmniej jedną kolumnę do zamiany na liczby.")
        missing_columns = [name for name in columns if name not in self.current.columns]
        if missing_columns:
            names = ", ".join(missing_columns)
            raise ValueError(f"Nie ma kolumn: {names}. Wybierz nazwy z pliku.")

        frame = self.current.copy(deep=True)
        converted = 0
        for name in columns:
            numeric = pd.to_numeric(frame[name], errors="coerce")
            became_missing = frame[name].notna() & numeric.isna()
            converted += int(became_missing.sum())
            frame[name] = numeric

        before = len(self.current)
        self.current = frame
        self.record(
            operation="coerce_to_numeric",
            parameters={"columns": list(columns)},
            rows_before=before,
            rows_after=len(self.current),
            affected=converted,
            message=(
                f"Zamieniono nieliczbowe wpisy na braki w kolumnach {list(columns)}: "
                f"{converted} komórek"
            ),
        )
        return converted

    def fill_numeric_median(self, columns: list[str]) -> int:
        """Uzupełnij braki medianą tylko w podanych kolumnach liczbowych."""
        if not columns:
            raise ValueError(
                "Wybierz co najmniej jedną kolumnę liczbową do uzupełnienia medianą."
            )
        missing_columns = [name for name in columns if name not in self.current.columns]
        if missing_columns:
            names = ", ".join(missing_columns)
            raise ValueError(f"Nie ma kolumn: {names}. Wybierz nazwy z pliku.")

        frame = self.current.copy(deep=True)
        filled = 0
        for name in columns:
            if not pd.api.types.is_numeric_dtype(frame[name]):
                raise ValueError(
                    f"Kolumna {name} nie jest liczbowa. "
                    "Mediana uzupełnia tylko liczby. Wybierz inną kolumnę."
                )
            gaps = int(frame[name].isna().sum())
            if gaps == 0:
                continue
            frame[name] = frame[name].fillna(frame[name].median())
            filled += gaps

        before = len(self.current)
        self.current = frame
        self.record(
            operation="fill_numeric_median",
            parameters={"columns": list(columns), "strategy": "median"},
            rows_before=before,
            rows_after=len(self.current),
            affected=filled,
            message=(
                f"Uzupełniono braki medianą w kolumnach {list(columns)}: "
                f"{filled} komórek"
            ),
        )
        return filled

    def range_mask(
        self,
        column: str,
        minimum: float | None,
        maximum: float | None,
    ) -> pd.Series:
        """Wskaż wartości poza zakresem. Tabeli nie zmienia."""
        if column not in self.current.columns:
            raise ValueError(f"Kolumna {column} nie istnieje.")
        if minimum is None and maximum is None:
            raise ValueError("Podaj dolną albo górną granicę.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("Dolna granica jest większa od górnej. Zamień wartości.")
        values = pd.to_numeric(self.current[column], errors="coerce")
        mask = pd.Series(False, index=self.current.index)
        if minimum is not None:
            mask = mask | (values < minimum)
        if maximum is not None:
            mask = mask | (values > maximum)
        return mask.fillna(False)

    def iqr_mask(self, column: str, factor: float = 1.5) -> pd.Series:
        """Maska wartości odstających. Tabeli nie zmienia."""
        if factor <= 0:
            raise ValueError("Współczynnik musi być dodatni.")
        if column not in self.current.columns:
            raise ValueError(f"Kolumna {column} nie istnieje.")
        values = pd.to_numeric(self.current[column], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return pd.Series(False, index=self.current.index)
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return (values < lower) | (values > upper)

    def describe_iqr(self, column: str, factor: float = 1.5) -> str:
        """Zdanie o granicach IQR. Tabeli nie zmienia."""
        if factor <= 0:
            raise ValueError("Współczynnik musi być dodatni.")
        if column not in self.current.columns:
            raise ValueError(f"Kolumna {column} nie istnieje.")
        values = pd.to_numeric(self.current[column], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return (
                f"Kolumna {column} nie ma liczbowego rozrzutu. "
                "Reguła IQR nie wskazuje wartości odstających."
            )
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        count = int(self.iqr_mask(column, factor).sum())
        return (
            f"IQR dla {column}, współczynnik {factor}: "
            f"granice od {lower:.4g} do {upper:.4g}, "
            f"wartości odstających: {count}. Dane nie zostały zmienione."
        )

    def reset(self) -> None:
        before = len(self.current)
        self.current = self.raw.copy(deep=True)
        self.record(
            operation="reset",
            parameters={},
            rows_before=before,
            rows_after=len(self.current),
            affected=0,
            message="Przywrócono dane do stanu początkowego.",
        )
