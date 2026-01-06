"""RDKit logging utilities."""

from contextlib import contextmanager

from rdkit import RDLogger


def suppress_rdkit_warnings():
    """Suppress all RDKit warnings (kekulization errors, etc.)."""
    RDLogger.DisableLog('rdApp.*')


def enable_rdkit_warnings():
    """Enable RDKit warnings."""
    RDLogger.EnableLog('rdApp.*')


def set_rdkit_log_level(level: str = "error"):
    """
    Set RDKit logging level.

    Args:
        level: One of 'debug', 'info', 'warning', 'error', 'critical'
               Levels below 'error' will show warnings.
    """
    level_lower = level.lower()
    if level_lower in ("error", "critical"):
        suppress_rdkit_warnings()
    else:
        enable_rdkit_warnings()


@contextmanager
def rdkit_log_level(level: str = "error"):
    """
    Context manager to temporarily set RDKit log level.

    Args:
        level: Log level to set within the context

    Example:
        with rdkit_log_level("error"):
            mol = Chem.MolFromSmiles(smiles)  # No warnings
    """
    # Store current state
    was_suppressed = _warnings_suppressed

    try:
        set_rdkit_log_level(level)
        yield
    finally:
        if was_suppressed:
            suppress_rdkit_warnings()
        else:
            enable_rdkit_warnings()


# Track whether warnings are suppressed globally
_warnings_suppressed = False
_app_warnings_suppressed = False


def configure_rdkit_logging(suppress_warnings: bool = True):
    """
    Configure RDKit logging for the CLI.

    Args:
        suppress_warnings: If True, suppress RDKit warnings (kekulization, etc.)
    """
    global _warnings_suppressed

    if suppress_warnings:
        suppress_rdkit_warnings()
        _warnings_suppressed = True
    else:
        enable_rdkit_warnings()
        _warnings_suppressed = False


def suppress_app_warnings():
    """Suppress application warnings (failed SMILES parsing, etc.)."""
    global _app_warnings_suppressed
    _app_warnings_suppressed = True


def enable_app_warnings():
    """Enable application warnings."""
    global _app_warnings_suppressed
    _app_warnings_suppressed = False


def are_warnings_suppressed() -> bool:
    """Check if RDKit warnings are currently suppressed."""
    return _warnings_suppressed


def are_app_warnings_suppressed() -> bool:
    """Check if application warnings are currently suppressed."""
    return _app_warnings_suppressed


def configure_all_warnings(suppress: bool = True):
    """
    Configure both RDKit and application warnings.

    Args:
        suppress: If True, suppress all warnings
    """
    configure_rdkit_logging(suppress_warnings=suppress)
    if suppress:
        suppress_app_warnings()
    else:
        enable_app_warnings()
