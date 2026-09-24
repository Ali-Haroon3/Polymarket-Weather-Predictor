"""The local capture wrapper must build before invoking data-writing binaries."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/daily_capture.sh"


class DailyCaptureTests(unittest.TestCase):
    def run_wrapper(self, build_exit=0, capture_exit=0, minimal_cron_path=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "target/release").mkdir(parents=True)
            (root / "stub-bin").mkdir()
            task_home = root / "test-home"
            cargo_home_bin = task_home / ".cargo/bin"
            cargo_home_bin.mkdir(parents=True)
            cargo_stub_dir = cargo_home_bin if minimal_cron_path else root / "stub-bin"
            script = root / "scripts/daily_capture.sh"
            shutil.copyfile(SCRIPT, script)
            trace = root / "calls.log"
            stubs = {
                cargo_stub_dir / "cargo": (
                    'printf "build:%s\\n" "$*" >> "$DAILY_CAPTURE_TEST_TRACE"\n'
                    'exit "$DAILY_CAPTURE_TEST_BUILD_EXIT"\n'
                ),
                root / "target/release/capture_prices": (
                    'printf "capture\\n" >> "$DAILY_CAPTURE_TEST_TRACE"\n'
                    'exit "$DAILY_CAPTURE_TEST_CAPTURE_EXIT"\n'
                ),
                root / "target/release/weather_dashboard": (
                    'printf "dashboard:%s\\n" "$*" >> "$DAILY_CAPTURE_TEST_TRACE"\n'
                ),
            }
            for path, body in stubs.items():
                path.write_text("#!/bin/bash\n" + body)
                path.chmod(0o755)
            env = dict(os.environ,
                       HOME=str(task_home),
                       PATH=("/usr/bin:/bin" if minimal_cron_path else
                             str(root / "stub-bin") + os.pathsep + "/usr/bin:/bin"),
                       DAILY_CAPTURE_TEST_TRACE=str(trace),
                       DAILY_CAPTURE_TEST_BUILD_EXIT=str(build_exit),
                       DAILY_CAPTURE_TEST_CAPTURE_EXIT=str(capture_exit))
            result = subprocess.run(["bash", str(script)], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=10)
            calls = trace.read_text().splitlines() if trace.exists() else []
            return result, calls

    def test_builds_both_binaries_before_capture_and_dashboard(self):
        result, calls = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [
            "build:build --release --bin capture_prices --bin weather_dashboard",
            "capture",
            "dashboard:--output dashboard.html",
        ])

    def test_failed_build_stops_existing_capture_and_dashboard_binaries(self):
        result, calls = self.run_wrapper(build_exit=7)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [
            "build:build --release --bin capture_prices --bin weather_dashboard",
        ])
        self.assertIn("refusing to run stale", result.stderr)

    def test_capture_runtime_failure_still_renders_last_good_data(self):
        result, calls = self.run_wrapper(capture_exit=9)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [
            "build:build --release --bin capture_prices --bin weather_dashboard",
            "capture",
            "dashboard:--output dashboard.html",
        ])

    def test_minimal_cron_path_finds_cargo_in_isolated_home(self):
        result, calls = self.run_wrapper(minimal_cron_path=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [
            "build:build --release --bin capture_prices --bin weather_dashboard",
            "capture",
            "dashboard:--output dashboard.html",
        ])


if __name__ == "__main__":
    unittest.main()
