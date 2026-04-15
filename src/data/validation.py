"""Data validation module using schema-based validation."""

import pandas as pd
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    checks_passed: int
    checks_failed: int
    issues: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return f"Validation {status}: {self.checks_passed} passed, {self.checks_failed} failed"


class DataValidator:
    """Schema-based data validator."""

    def __init__(self):
        self.issues: List[Dict[str, Any]] = []

    def validate(self, df: pd.DataFrame, schema: Dict[str, Any]) -> ValidationResult:
        """Validate a DataFrame against a schema."""
        self.issues = []
        checks_passed = 0
        checks_failed = 0

        required_cols = schema.get('required_columns', [])
        missing = set(required_cols) - set(df.columns)
        if missing:
            self._add_issue("missing_columns", f"Missing required columns: {missing}", ValidationSeverity.ERROR)
            checks_failed += 1
        else:
            checks_passed += 1

        for col, expected_type in schema.get('column_types', {}).items():
            if col in df.columns:
                if not self._check_type(df[col], expected_type):
                    self._add_issue("type_mismatch", f"Column {col}: expected {expected_type}", ValidationSeverity.ERROR)
                    checks_failed += 1
                else:
                    checks_passed += 1

        for col, bounds in schema.get('value_ranges', {}).items():
            if col in df.columns:
                min_val, max_val = bounds.get('min'), bounds.get('max')
                if min_val is not None and df[col].min() < min_val:
                    self._add_issue("out_of_range", f"{col} has values below {min_val}", ValidationSeverity.WARNING)
                    checks_failed += 1
                elif max_val is not None and df[col].max() > max_val:
                    self._add_issue("out_of_range", f"{col} has values above {max_val}", ValidationSeverity.WARNING)
                    checks_failed += 1
                else:
                    checks_passed += 1

        max_null_pct = schema.get('max_null_percentage', 100)
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            if null_pct > max_null_pct:
                self._add_issue("high_nulls", f"{col}: {null_pct:.1f}% null (max: {max_null_pct}%)", ValidationSeverity.WARNING)
                checks_failed += 1
            else:
                checks_passed += 1

        min_rows = schema.get('min_rows', 0)
        if len(df) < min_rows:
            self._add_issue("insufficient_rows", f"Got {len(df)} rows, expected >= {min_rows}", ValidationSeverity.ERROR)
            checks_failed += 1
        else:
            checks_passed += 1

        for col in schema.get('unique_columns', []):
            if col in df.columns and df[col].duplicated().any():
                n_dups = df[col].duplicated().sum()
                self._add_issue("duplicate_values", f"{col} has {n_dups} duplicate values", ValidationSeverity.ERROR)
                checks_failed += 1
            else:
                checks_passed += 1

        is_valid = all(i['severity'] != ValidationSeverity.ERROR.value for i in self.issues)
        result = ValidationResult(
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            issues=self.issues,
        )

        logger.info(result.summary())
        return result

    def _check_type(self, series: pd.Series, expected: str) -> bool:
        type_map = {
            'numeric': lambda s: pd.api.types.is_numeric_dtype(s),
            'string': lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
            'datetime': lambda s: pd.api.types.is_datetime64_any_dtype(s),
            'boolean': lambda s: pd.api.types.is_bool_dtype(s),
            'integer': lambda s: pd.api.types.is_integer_dtype(s),
            'float': lambda s: pd.api.types.is_float_dtype(s),
        }
        checker = type_map.get(expected)
        return checker(series) if checker else True

    def _add_issue(self, check: str, message: str, severity: ValidationSeverity) -> None:
        issue = {"check": check, "message": message, "severity": severity.value}
        self.issues.append(issue)
        log_fn = getattr(logger, severity.value if severity.value != 'error' else 'error')
        log_fn(f"Validation: {message}")


# Pre-defined schemas for common datasets
SCHEMAS = {
    'players': {
        'required_columns': ['player_id', 'name', 'team', 'position', 'fantasy_points'],
        'column_types': {'player_id': 'integer', 'fantasy_points': 'numeric', 'salary': 'numeric'},
        'unique_columns': ['player_id'],
        'min_rows': 10,
        'max_null_percentage': 20,
    },
    'user_profiles': {
        'required_columns': ['user_id', 'username', 'total_entries', 'win_rate'],
        'column_types': {'user_id': 'integer', 'win_rate': 'numeric'},
        'value_ranges': {'win_rate': {'min': 0, 'max': 1}},
        'unique_columns': ['user_id'],
        'min_rows': 10,
    },
    'contests': {
        'required_columns': ['contest_id', 'contest_type', 'entry_fee', 'prize_pool'],
        'column_types': {'contest_id': 'integer', 'entry_fee': 'numeric', 'prize_pool': 'numeric'},
        'value_ranges': {'entry_fee': {'min': 0}, 'prize_pool': {'min': 0}},
        'unique_columns': ['contest_id'],
        'min_rows': 10,
    },
}
