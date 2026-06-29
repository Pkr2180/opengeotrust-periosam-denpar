"""Logging setup using loguru with rich fallback."""
import sys
from pathlib import Path

try:
    from loguru import logger

    def setup_logger(log_file: str | Path | None = None, level: str = "INFO") -> None:
        logger.remove()
        logger.add(sys.stderr, level=level, colorize=True,
                   format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention="7 days")

except ImportError:
    import logging as _logging

    class _LoguruShim:
        def __init__(self):
            self._logger = _logging.getLogger("periosam")
            if not self._logger.handlers:
                h = _logging.StreamHandler()
                h.setFormatter(_logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                                   datefmt="%H:%M:%S"))
                self._logger.addHandler(h)
            self._logger.setLevel(_logging.DEBUG)

        def info(self, msg, *a, **kw): self._logger.info(msg, *a, **kw)
        def debug(self, msg, *a, **kw): self._logger.debug(msg, *a, **kw)
        def warning(self, msg, *a, **kw): self._logger.warning(msg, *a, **kw)
        def error(self, msg, *a, **kw): self._logger.error(msg, *a, **kw)
        def success(self, msg, *a, **kw): self._logger.info(f"[OK] {msg}", *a, **kw)

    logger = _LoguruShim()

    def setup_logger(log_file=None, level="INFO"):
        pass
