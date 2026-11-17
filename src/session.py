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
    affected: str
    message: str
    
    def text(self):
        moment = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return(
            f"{moment} | {self.message}"
            f"(przed: {self.rows_before}, po: {self.rows_after},"
            f"zmiany: {self.affected})"
        )


@dataclass
class DataSession:
    """Sesja zapisu danych."""
    
    raw: pd.DataFrame
    current: pd.DataFrame = field(init=False)
    audit_log: list[AuditEntry] = field(default_factory=list)
    
    def __post_init__(self):
        if self.raw.empty:
            raise ValueError("Plik nie zawiera danych.")
        self.current = self.raw.copy()
        
    def record(
        self,
        operation: str,
        parameters: dict[str, object],
        rows_before: int,
        rows_after: int,
        affected: str,
        message: str,
    ):
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
    def from_csv(cls, path: Path):
        """Wczytaj dane z pliku CSV."""
        frame = pd.read_csv(path, sep=None, engine="python")
        return cls(frame)
    
    def profile(self):
        """Profil danych"""
        return pd.DataFrame(
            {
                "dtype": self.current.dtypes.astype(str),
                "missing": self.current.isna().sum(),
                "missing_pct": self.current.isna().mean().mul(100).round(2),
                "unique": self.current.nunique(dropna=True),
            }
        )
        
    def drop_exact_duplicates(self):
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
    
    def iqr_mask(self, column: str, factor: float = 1.5):
        """Maska wartości odstających."""
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
        count = int(self.iqr_mask(column, factor).sum())
        return (
            f"IQR dla {column}, współczynnik: {factor}: "
            f"granice od {lower:.4g} do {upper:.4g}, "
            f"wartości odstających: {count}. Dane nie zostały zmienione."
        )

def reset(self):
    before = len(self.current)
    self.current = self.raw.copy()
    self.record(
        operation="reset",
        parameters={},
        rows_before=before,
        rows_after=len(self.current),
        affected=0,
        message="Przywrócono dane do stanu początkowego.",
    )
