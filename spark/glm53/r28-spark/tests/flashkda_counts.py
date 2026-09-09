"""Require the entire frozen packed/dense GPU test matrix, without skips."""

if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")


class RequiredMatrix:
    def __init__(self):
        self.collected = 0
        self.passed = 0
        self.invalid = []

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        if report.skipped or report.failed or hasattr(report, "wasxfail"):
            self.invalid.append(report.nodeid)
        if report.when == "call" and report.passed:
            self.passed += 1

    def verify(self, exit_code):
        if exit_code != 0 or self.collected != 12 or self.passed != 12 or self.invalid:
            raise RuntimeError(
                f"FlashKDA gate incomplete: exit={exit_code}, collected={self.collected}, passed={self.passed}, invalid={self.invalid}"
            )
