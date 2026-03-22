from __future__ import annotations

import os
import unittest


def _iter_test_cases(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_test_cases(item)
        else:
            yield item


class UnittestDiscoveryContractTests(unittest.TestCase):
    def test_repo_root_unittest_discovery_finds_contract_suites(self) -> None:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        discovered = unittest.defaultTestLoader.discover(
            start_dir=repo_root,
            top_level_dir=repo_root,
        )
        test_ids = {case.id() for case in _iter_test_cases(discovered)}

        self.assertTrue(
            any(
                test_id.startswith("tests.test_db_connection_contract.")
                for test_id in test_ids
            ),
            "Default `python -m unittest` discovery from repo root must include "
            "tests/test_db_connection_contract.py so conventional runs cannot go "
            "false-green with zero contract tests executed.",
        )
        self.assertTrue(
            any(
                test_id.startswith("tests.test_project_metadata_contract.")
                for test_id in test_ids
            ),
            "Default `python -m unittest` discovery from repo root must include "
            "tests/test_project_metadata_contract.py rather than only ad hoc "
            "module-level invocations.",
        )
        self.assertTrue(
            any(
                test_id.startswith("tests.test_contract_test_runner.")
                for test_id in test_ids
            ),
            "Default `python -m unittest` discovery from repo root must include "
            "tests/test_contract_test_runner.py so the authoritative low-friction "
            "runner cannot silently fall out of conventional test runs.",
        )
        self.assertTrue(
            any(
                test_id.startswith("tests.test_postgres_contract_bootstrap_script.")
                for test_id in test_ids
            ),
            "Default `python -m unittest` discovery from repo root must include "
            "tests/test_postgres_contract_bootstrap_script.py so the documented "
            "one-command DB bootstrap entrypoint cannot silently disappear.",
        )
        self.assertTrue(
            any(
                test_id.startswith("tests.test_db_postgres_contract.")
                for test_id in test_ids
            ),
            "Default `python -m unittest` discovery from repo root must include "
            "tests/test_db_postgres_contract.py so Postgres schema coverage does "
            "not silently disappear.",
        )
        self.assertTrue(
            any(
                test_id.startswith("tests.test_db_postgres_harness_contract.")
                for test_id in test_ids
            ),
            "Default `python -m unittest` discovery from repo root must include "
            "tests/test_db_postgres_harness_contract.py so harness-policy coverage "
            "does not silently disappear.",
        )
        self.assertTrue(
            any(
                test_id.startswith("tests.test_db_postgres_smoke_contract.")
                for test_id in test_ids
            ),
            "Default `python -m unittest` discovery from repo root must include "
            "tests/test_db_postgres_smoke_contract.py so the fail-fast real-"
            "Postgres smoke coverage cannot silently disappear.",
        )
        self.assertFalse(
            any(
                test_id.startswith("tests.postgres_contract_bootstrap_script_smoke_contract.")
                for test_id in test_ids
            ),
            "Default `python -m unittest` discovery from repo root must stay "
            "sandbox-safe and therefore must not pick up the DB-backed bootstrap "
            "script smoke probe automatically.",
        )


if __name__ == "__main__":
    unittest.main()
