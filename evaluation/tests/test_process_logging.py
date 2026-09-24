import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from defenses.core.process_logging import run_logged_command, stage_log_path


class ProcessLoggingTests(unittest.TestCase):
    def test_stdout_stderr_and_exit_status_reach_main_and_detail_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stage.log"
            main = io.StringIO()
            with contextlib.redirect_stdout(main):
                result = run_logged_command(
                    [sys.executable, "-c", "import sys; print('progress', flush=True); print('error', file=sys.stderr); sys.exit(7)"],
                    cwd=Path(tmp), log_path=path,
                )
            self.assertEqual(result.returncode, 7)
            for text in ("progress", "error"):
                self.assertIn(text, main.getvalue())
                self.assertIn(text, path.read_text())

    def test_job_directory_keeps_identically_named_stage_logs_distinct(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"RUN_LOG_DIR": tmp}):
            first = stage_log_path(Path("outputs/clean/seqkd.log"))
            second = stage_log_path(Path("outputs/counter/seqkd.log"))
            self.assertEqual(first.parent, Path(tmp))
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
