# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from typing import List, Tuple

from junitparser import JUnitXml

TEST_DIR = os.path.dirname(
    __file__
)  # this file is at the root of the ./tests directory
LOG_DIR = os.path.join(TEST_DIR, "logs")
TEST_TYPES = ["unit", "integration", "e2e"]


def print_tests(test_type, tests):
    print(f"////////////////////////// {test_type} tests //////////////////////////")
    for t in tests:
        print(f"    {t}")
    print("///////////////////////////////////////////////////////////////////////")


def summarize(reports) -> bool:
    """Return True on success, False on failure."""

    total, passed, failed, skiped = 0, 0, 0, 0
    print("================== Detailed Results ==================")
    for r in reports:
        xml = JUnitXml.fromfile(r)
        for suite in xml:
            for case in suite:
                total += 1
                if case.is_passed:
                    passed += 1
                    status = "PASSED"
                elif case.is_skipped:
                    skiped += 1
                    status = "SKIPPED"
                else:
                    failed += 1
                    status = "FAILED"
                name = case.classname + "::" + case.name
                print(f"{name:70} {status}")

    print("================== Summary ==================")
    print(f"Summary: {total=}, {passed=}, {failed=}, {skiped=}")
    return failed == 0


def show_slowest(tests, durations, k=10):
    def _show(durations, file=sys.stdout, markdown=False):
        if markdown:
            print("## Slowest Tests: ", ",".join(tests), file=file)
            print("```text", file=file)
        print("================== Slowest Tests ==================", file=file)
        for t, d in sorted(durations, key=lambda x: -x[1])[:k]:
            print(f"{t:70} {d:.1f} seconds", file=file)
        if markdown:
            print("```", file=file)

    _show(durations)
    with open(os.path.join(LOG_DIR, "slowest_tests.txt"), "w") as f:
        _show(durations, f)

    if os.environ.get("GITHUB_WORKFLOW", ""):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            _show(durations, f, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        nargs="+",
        required=True,
        help=f"Specify test types to run, may be: {TEST_TYPES}",
    )
    parser.add_argument(
        "--e2e_keys",
        type=str,
        default="",
        help="pytest keys if you want to run specific e2e test cases",
    )
    args = parser.parse_args()

    print(f"/// {args=}")
    for t in args.test:
        if t not in TEST_TYPES:
            raise ValueError(f"Unknown test type: {t}, valid types are: {TEST_TYPES}")

    print(f"/// {TEST_DIR=}")
    print(f"/// {LOG_DIR=}")
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    all_tests = glob.glob(os.path.join(TEST_DIR, "**/*.py"), recursive=True)
    all_tests.remove(__file__)

    e2e_tests = []  # ['tests/integration/test_models_e2e.py']

    integration_tests = (
        []
    )  # glob.glob(os.path.join(TEST_DIR, 'integration/**/*.py'), recursive=True)

    unit_tests = list(set(all_tests) - set(integration_tests) - set(e2e_tests))

    test_cases: List[
        Tuple[str, List[str]]
    ] = []  # list of (test_file, [extra flags]) tuple.
    if "unit" in args.test:
        print_tests("unit", unit_tests)
        test_cases.extend([(t, []) for t in unit_tests])

    if "integration" in args.test:
        print_tests("integration", integration_tests)
        test_cases.extend([(t, []) for t in integration_tests])

    if "e2e" in args.test:
        print_tests("e2e", e2e_tests)
        if args.e2e_keys:
            args.e2e_keys += " and not beam_search and not parallel_sampling"
        else:
            args.e2e_keys = "not beam_search and not parallel_sampling"
        if args.e2e_keys:
            test_cases.extend(
                [(t, ["-k", args.e2e_keys] if args.e2e_keys else []) for t in e2e_tests]
            )
        else:
            test_cases.extend([(t, []) for t in e2e_tests])

    # NB(xiafei.qiuxf): Why run tests one by one, instead of just `pytest ./tests`?
    # Because tests involving torch.distributed need to be run in their own sub-process.
    # These tests are are children of pytest main process. If we use pytest-forked extension,
    # it would be problematic because torch.cuda requires 'spawn' method to create new process,
    # and rejects 'fork' method. pytest-xdist is also tried, but it start sa group of new
    # process and asigns tests to them, which may also put multiple tests into single process.
    # I also tried to hack a spawn version of pytest-forked, but some of pytest's internal
    # data structures are not picklable. So finally we come to this solution.
    reports = []
    durations = []
    for test, flags in test_cases:
        xml_fname = os.path.join(
            LOG_DIR, test.replace("/", "__").replace(".py", ".xml")
        )
        cmd_args = [
            "pytest",
            test,
            "-v",
            "--durations",
            "3",
            "--junitxml",
            xml_fname,
            *flags,
        ]
        print(f"/// Running {test}: {cmd_args}")
        start = time.time()
        subprocess.call(cmd_args)
        duration = time.time() - start
        durations.append((test, duration))
        reports.append(xml_fname)

    success = summarize(reports)
    show_slowest(args.test, durations)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
