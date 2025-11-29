# cure_db.py
#
# State-wise cure lookup for Rice + Wheat diseases.
#
# Expects a CSV in the SAME directory as this file:
#   india_statewise_rice_wheat_focused_diseases.csv
#
# Required columns (case-sensitive):
#   Crop,State,Disease,Cultural_Management,
#   Chemical_Management,Dose,Season,Notes

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import csv
import logging

log = logging.getLogger("cure-db")

# Folder where THIS file lives
BASE_DIR: Path = Path(__file__).resolve().parent

# CSV file path (put the CSV next to cure_db.py)
CURE_CSV_PATH: Path = BASE_DIR / "india_statewise_rice_wheat_focused_diseases.csv"


class CureRecord:
    """One row from the cure CSV."""

    def __init__(self, row: Dict[str, str]):
        self.crop: str = (row.get("Crop") or "").strip()
        self.state: str = (row.get("State") or "").strip()
        self.disease: str = (row.get("Disease") or "").strip()

        self.cultural: str = (row.get("Cultural_Management") or "").strip()
        self.chemical: str = (row.get("Chemical_Management") or "").strip()
        self.dose: str = (row.get("Dose") or "").strip()
        self.season: str = (row.get("Season") or "").strip()
        self.notes: str = (row.get("Notes") or "").strip()

    def to_dict(self) -> Dict[str, str]:
        """Return dict ready for Pydantic / API."""
        return {
            "crop": self.crop,
            "state": self.state,
            "disease": self.disease,
            "cultural_management": self.cultural,
            "chemical_management": self.chemical,
            "dose": self.dose,
            "season": self.season,
            "notes": self.notes,
        }


class CureDB:
    """
    In-memory cure database.

    Two maps:
      - _data_exact[(crop, disease, state)]  -> exact, state-specific record
      - _data_default[(crop, disease)]       -> DEFAULT fallback record
    """

    def __init__(self, csv_path: Union[str, Path] = CURE_CSV_PATH):
        # Normalize to Path in case user passes a string
        self.csv_path: Path = Path(csv_path)

        self._data_exact: Dict[Tuple[str, str, str], CureRecord] = {}
        self._data_default: Dict[Tuple[str, str], CureRecord] = {}

        self._load(self.csv_path)

    def _load(self, csv_path: Path) -> None:
        """Load CSV into exact + default maps."""
        if not csv_path.exists():
            log.warning(
                "Cure CSV not found at %s. Cure lookup will be EMPTY.",
                csv_path,
            )
            return

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = CureRecord(row)

                crop_key = rec.crop.lower()          # e.g. "rice" / "wheat"
                disease_key = rec.disease            # MUST match model label exactly
                state_key = rec.state.lower()        # e.g. "odisha" / "punjab" / "default"

                # 1) Exact key: (crop, disease, state)
                self._data_exact[(crop_key, disease_key, state_key)] = rec

                # 2) If state == DEFAULT, also register fallback (crop, disease)
                if rec.state.upper() == "DEFAULT":
                    self._data_default[(crop_key, disease_key)] = rec

        log.info(
            "Loaded %d exact and %d DEFAULT cure records from %s",
            len(self._data_exact),
            len(self._data_default),
            csv_path,
        )

    def get_cure(
        self,
        crop: str,
        disease: str,
        state: str,
    ) -> Optional[CureRecord]:
        """
        Lookup order:
          1) (crop, disease, state)      -> state-specific recommendation
          2) (crop, disease, DEFAULT)    -> generic recommendation
          3) None                        -> no match
        """
        crop_key = (crop or "").lower().strip()
        disease_key = (disease or "").strip()
        state_key = (state or "").lower().strip()

        # 1) exact state-specific record
        key_exact = (crop_key, disease_key, state_key)
        rec = self._data_exact.get(key_exact)
        if rec:
            return rec

        # 2) DEFAULT fallback
        key_default = (crop_key, disease_key)
        rec = self._data_default.get(key_default)
        if rec:
            return rec

        # 3) nothing found
        return None


# Global singleton used by disease_api.py
cure_db = CureDB()
