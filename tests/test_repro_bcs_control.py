"""Exercise the SYCL diagnostic's success/failure reporting without GPU work."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SYCL_STUB = r"""
#pragma once
#include <cassert>
#include <cstdlib>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
namespace sycl {
using exception_list = std::vector<std::exception_ptr>;
struct event {};
struct handler {
    void depends_on(event) {}
    template<class F> void single_task(F f) { f(); }
};
class queue {
    std::function<void(exception_list)> on_error;
    int copies = 0, kernels = 0;
public:
    explicit queue(std::function<void(exception_list)> f) : on_error(f) {}
    event memcpy(void *, const void *, size_t) { ++copies; return {}; }
    template<class F> void submit(F f) { handler h; f(h); ++kernels; }
    void wait_and_throw() {
        assert(copies == 10 && kernels == 10);
        const std::string mode = std::getenv("SYCL_TEST_MODE");
        if (mode == "sync") {
            throw std::runtime_error("injected synchronous failure");
        }
        if (mode == "copy" || mode == "kernel") {
            on_error({std::make_exception_ptr(std::runtime_error("injected " + mode + " failure"))});
        }
        copies = kernels = 0;
    }
    ~queue() { assert(std::uncaught_exceptions() || (copies == 0 && kernels == 0)); }
};
inline void * malloc_device(size_t n, queue &) { return std::malloc(n); }
inline void free(void * p, queue &) { std::free(p); }
}
"""


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="bcs-control-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        (root / "sycl").mkdir()
        (root / "sycl/sycl.hpp").write_text(SYCL_STUB)
        cls.binary = root / "control"
        subprocess.run([
            "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(root),
            str(ROOT / "repro_bcs_reset.cpp"), "-o", str(cls.binary),
        ], check=True, timeout=30)

    def run_control(self, mode):
        return subprocess.run(
            [str(self.binary)], env={**os.environ, "SYCL_TEST_MODE": mode},
            capture_output=True, text=True, timeout=10,
        )

    def test_success_is_labeled_as_control(self):
        result = self.run_control("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("success does not test the mmap defect", result.stdout)
        self.assertIn("Done.", result.stdout)

    def test_async_copy_and_kernel_failures_cannot_report_success(self):
        for mode in ("copy", "kernel"):
            with self.subTest(mode=mode):
                result = self.run_control(mode)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"Asynchronous SYCL failure: injected {mode} failure", result.stderr)
                self.assertNotIn("Done.", result.stdout)

    def test_synchronous_failure_cannot_report_success(self):
        result = self.run_control("sync")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("SYCL control failed: injected synchronous failure", result.stderr)
        self.assertNotIn("Done.", result.stdout)


if __name__ == "__main__":
    unittest.main()
