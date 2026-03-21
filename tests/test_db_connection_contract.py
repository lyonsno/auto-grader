from __future__ import annotations

from contextlib import contextmanager
import inspect
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import parse_qsl, urlparse

from auto_grader.db import create_connection

_DATABASE_URL_REQUIRED_RE = (
    r"(?is)(?=.*(?:database[_ ]?url|\burl\b))(?=.*(?:required|missing|set))"
)
_DATABASE_URL_BLANK_RE = (
    r"(?is)(?=.*(?:database[_ ]?url|\burl\b))(?=.*(?:blank|empty|whitespace))"
)
_DATABASE_URL_SCHEME_RE = (
    r"(?is)(?=.*(?:database[_ ]?url|\burl\b))(?=.*(?:postgres|scheme|unsupported|invalid))"
)


@contextmanager
def _patched_env(**updates: str | None):
    """Temporarily apply environment variable updates for a test."""

    original: dict[str, str | None] = {}
    for key, value in updates.items():
        original[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _ConnectSpy:
    def __init__(self, return_value: object):
        self.return_value = return_value
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.return_value


def _load_psycopg_for_default_connector() -> tuple[object, object]:
    import psycopg
    from psycopg.rows import dict_row

    return psycopg, dict_row


class DatabaseConnectionSignatureContractTests(unittest.TestCase):
    """Fail-first API shape contract for Postgres hard-cut migration."""

    def test_signature_requires_postgres_contract_and_forbids_legacy_path(self) -> None:
        signature = inspect.signature(create_connection)
        params = signature.parameters
        self.assertNotIn(
            "path",
            params,
            "Postgres hard-cut contract forbids legacy SQLite `path` parameter.",
        )
        self.assertIn(
            "database_url",
            params,
            "create_connection() must accept an explicit database_url keyword argument.",
        )
        self.assertIn(
            "connect_fn",
            params,
            "create_connection() must accept an injectable connect_fn for tests.",
        )

    def test_connect_fn_remains_optional_for_production_calls(self) -> None:
        signature = inspect.signature(create_connection)
        params = signature.parameters
        self.assertIn("connect_fn", params)
        self.assertIsNot(
            params["connect_fn"].default,
            inspect.Signature.empty,
            "connect_fn must remain optional for production call sites.",
        )

    def test_legacy_path_keyword_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "path"):
            create_connection(path=":memory:")


class DatabaseConnectionBehaviorContractTests(unittest.TestCase):
    """Behavioral contract tests once Postgres API shape is available."""

    def test_create_connection_requires_url_or_database_url_env_var(self) -> None:
        spy = _ConnectSpy(return_value=object())
        with _patched_env(DATABASE_URL=None):
            with self.assertRaisesRegex(
                ValueError,
                _DATABASE_URL_REQUIRED_RE,
            ):
                self._invoke_connection(database_url=None, connect_fn=spy)
        self.assertEqual(spy.calls, [])

    def test_create_connection_requires_url_or_database_url_env_var_without_connect_fn(
        self,
    ) -> None:
        with _patched_env(DATABASE_URL=None):
            with self.assertRaisesRegex(
                ValueError,
                _DATABASE_URL_REQUIRED_RE,
            ):
                self._invoke_connection(database_url=None)

    def test_create_connection_uses_database_url_env_var_when_url_not_supplied(
        self,
    ) -> None:
        sentinel = object()
        spy = _ConnectSpy(return_value=sentinel)
        with _patched_env(DATABASE_URL="postgresql://localhost/auto_grader_dev"):
            connection = self._invoke_connection(database_url=None, connect_fn=spy)
        self.assertIs(connection, sentinel)
        self._assert_connector_received_database_url(
            spy,
            "postgresql://localhost/auto_grader_dev",
        )

    def test_create_connection_prefers_explicit_url_over_env_var(self) -> None:
        sentinel = object()
        spy = _ConnectSpy(return_value=sentinel)
        with _patched_env(DATABASE_URL="postgresql://localhost/from-env"):
            connection = self._invoke_connection(
                database_url="postgresql://localhost/from-arg",
                connect_fn=spy,
            )
        self.assertIs(connection, sentinel)
        self._assert_connector_received_database_url(
            spy,
            "postgresql://localhost/from-arg",
        )

    def test_create_connection_prefers_valid_explicit_url_even_when_env_is_blank_or_invalid(
        self,
    ) -> None:
        for env_value in ("", "   ", "sqlite:///:memory:"):
            with self.subTest(env_value=env_value):
                sentinel = object()
                spy = _ConnectSpy(return_value=sentinel)
                with _patched_env(DATABASE_URL=env_value):
                    connection = self._invoke_connection(
                        database_url="postgresql://localhost/from-arg",
                        connect_fn=spy,
                    )
                self.assertIs(connection, sentinel)
                self._assert_connector_received_database_url(
                    spy,
                    "postgresql://localhost/from-arg",
                )

    def test_create_connection_without_connect_fn_still_validates_inputs(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            _DATABASE_URL_SCHEME_RE,
        ):
            self._invoke_connection(database_url="sqlite:///:memory:")

    def test_create_connection_without_connect_fn_uses_mapping_rows_and_autocommit(
        self,
    ) -> None:
        database_url = self._require_default_connector_test_database_url()

        connection = self._invoke_connection(database_url=database_url)
        try:
            self.assertIs(
                getattr(connection, "autocommit", None),
                True,
                "Default Postgres connection should enable autocommit.",
            )
            row = connection.execute("SELECT 1 AS value").fetchone()
            self.assertEqual(
                row["value"],
                1,
                "Default Postgres connection should return rows addressable by column name.",
            )
        finally:
            close = getattr(connection, "close", None)
            if callable(close):
                close()

    def test_create_connection_without_connect_fn_accepts_uppercase_postgres_scheme(
        self,
    ) -> None:
        database_url = self._require_default_connector_test_database_url()
        scheme, separator, remainder = database_url.partition("://")
        uppercase_database_url = f"{scheme.upper()}{separator}{remainder}"

        connection = self._invoke_connection(database_url=uppercase_database_url)
        try:
            row = connection.execute("SELECT 1 AS value").fetchone()
            self.assertEqual(
                row["value"],
                1,
                "Default connector path should accept uppercase Postgres URL schemes.",
            )
        finally:
            close = getattr(connection, "close", None)
            if callable(close):
                close()

    def test_create_connection_rejects_blank_urls(self) -> None:
        for database_url in ("", "   "):
            with self.subTest(database_url=database_url):
                spy = _ConnectSpy(return_value=object())
                with self.assertRaisesRegex(
                    ValueError,
                    _DATABASE_URL_BLANK_RE,
                ):
                    self._invoke_connection(
                        database_url=database_url,
                        connect_fn=spy,
                    )
                self.assertEqual(spy.calls, [])

    def test_blank_explicit_url_does_not_fallback_to_env_url(self) -> None:
        for database_url in ("", "   "):
            with self.subTest(database_url=database_url):
                spy = _ConnectSpy(return_value=object())
                with _patched_env(DATABASE_URL="postgresql://localhost/from-env"):
                    with self.assertRaisesRegex(
                        ValueError,
                        _DATABASE_URL_BLANK_RE,
                    ):
                        self._invoke_connection(
                            database_url=database_url,
                            connect_fn=spy,
                        )
                self.assertEqual(
                    spy.calls,
                    [],
                    "Blank explicit URL must fail validation, not connect using env fallback.",
                )

    def test_blank_explicit_url_does_not_fallback_to_env_without_connect_fn(
        self,
    ) -> None:
        for database_url in ("", "   "):
            with self.subTest(database_url=database_url):
                with _patched_env(DATABASE_URL="postgresql://localhost/from-env"):
                    with self.assertRaisesRegex(
                        ValueError,
                        _DATABASE_URL_BLANK_RE,
                    ):
                        self._invoke_connection(database_url=database_url)

    def test_create_connection_rejects_blank_database_url_from_env_var(self) -> None:
        for env_value in ("", "   "):
            with self.subTest(env_value=env_value):
                spy = _ConnectSpy(return_value=object())
                with _patched_env(DATABASE_URL=env_value):
                    with self.assertRaisesRegex(
                        ValueError,
                        _DATABASE_URL_BLANK_RE,
                    ):
                        self._invoke_connection(
                            database_url=None,
                            connect_fn=spy,
                        )
                self.assertEqual(spy.calls, [])

    def test_invalid_explicit_url_does_not_fallback_to_env_url(self) -> None:
        sentinel = object()
        spy = _ConnectSpy(return_value=sentinel)
        with _patched_env(DATABASE_URL="postgresql://localhost/from-env"):
            with self.assertRaisesRegex(
                ValueError,
                _DATABASE_URL_SCHEME_RE,
            ):
                self._invoke_connection(
                    database_url="sqlite:///:memory:",
                    connect_fn=spy,
                )
        self.assertEqual(
            spy.calls,
            [],
            "Invalid explicit URL must fail validation, not connect using env fallback.",
        )

    def test_invalid_explicit_url_does_not_fallback_to_env_without_connect_fn(self) -> None:
        with _patched_env(DATABASE_URL="postgresql://localhost/from-env"):
            with self.assertRaisesRegex(
                ValueError,
                _DATABASE_URL_SCHEME_RE,
            ):
                self._invoke_connection(database_url="sqlite:///:memory:")

    def test_invalid_env_scheme_is_rejected_when_explicit_url_missing(self) -> None:
        spy = _ConnectSpy(return_value=object())
        with _patched_env(DATABASE_URL="sqlite:///:memory:"):
            with self.assertRaisesRegex(ValueError, _DATABASE_URL_SCHEME_RE):
                self._invoke_connection(database_url=None, connect_fn=spy)
        self.assertEqual(spy.calls, [])

    def test_create_connection_rejects_non_postgres_urls(self) -> None:
        for database_url in (
            "sqlite:///:memory:",
            "mysql://localhost/auto_grader_dev",
            "/tmp/auto_grader.db",
        ):
            with self.subTest(database_url=database_url):
                spy = _ConnectSpy(return_value=object())
                with self.assertRaisesRegex(
                    ValueError,
                    _DATABASE_URL_SCHEME_RE,
                ):
                    self._invoke_connection(
                        database_url=database_url,
                        connect_fn=spy,
                    )
                self.assertEqual(spy.calls, [])

    def test_create_connection_accepts_postgres_and_postgresql_schemes(self) -> None:
        for database_url in (
            "postgres://grader:secret@localhost:5432/auto_grader_dev?sslmode=disable",
            "postgresql://grader:secret@localhost:5433/auto_grader_dev?application_name=ag",
            "POSTGRESQL://localhost/auto_grader_dev",
        ):
            with self.subTest(database_url=database_url):
                sentinel = object()
                spy = _ConnectSpy(return_value=sentinel)
                connection = self._invoke_connection(
                    database_url=database_url,
                    connect_fn=spy,
                )
                self.assertIs(connection, sentinel)
                self._assert_connector_received_database_url(spy, database_url)

    def _invoke_connection(self, **kwargs: object) -> object:
        try:
            return create_connection(**kwargs)
        except TypeError as exc:
            message = str(exc)
            if (
                "unexpected keyword argument 'database_url'" in message
                or "unexpected keyword argument 'connect_fn'" in message
            ):
                self.fail(
                    "Behavior contract requires create_connection to accept "
                    "`database_url` and optional `connect_fn`. "
                    f"kwargs={sorted(kwargs.keys())}; got TypeError: {exc}"
                )
            raise

    def _assert_connector_received_database_url(
        self,
        spy: _ConnectSpy,
        expected_url: str,
    ) -> None:
        self.assertEqual(
            len(spy.calls),
            1,
            "connect_fn should be invoked exactly once for a valid database URL.",
        )
        args, kwargs = spy.calls[0]
        for value in (*args, *kwargs.values()):
            if isinstance(value, str) and self._dsn_matches(
                expected_url,
                actual_url=value,
            ):
                return
        self.fail(
            "connect_fn must receive the resolved Postgres DSN as a positional "
            f"or keyword argument. call_args={args!r}, call_kwargs={kwargs!r}"
        )

    def _dsn_matches(self, expected_url: str, actual_url: str) -> bool:
        try:
            self._assert_dsn_equivalent_except_scheme_case(
                expected_url=expected_url,
                actual_url=actual_url,
            )
        except self.failureException:
            return False
        return True

    def _require_default_connector_test_database_url(self) -> str:
        database_url = os.environ.get("TEST_DATABASE_URL")
        if not database_url:
            self.skipTest(
                "Set TEST_DATABASE_URL to run the default-connector integration "
                "contract against a real Postgres instance."
            )
        try:
            psycopg, dict_row = _load_psycopg_for_default_connector()
        except ModuleNotFoundError:
            raise AssertionError(
                "Default-connector integration contract requires psycopg in the "
                "active environment when TEST_DATABASE_URL is explicitly set. "
                "Run `uv sync` first."
            )

        try:
            with psycopg.connect(
                database_url,
                autocommit=True,
                row_factory=dict_row,
            ) as probe:
                row = probe.execute("SELECT 1 AS value").fetchone()
        except Exception as exc:
            raise AssertionError(
                "Default-connector integration contract requires reachable local "
                f"Postgres at {database_url!r}: {exc}"
            ) from exc
        self.assertEqual(
            row["value"],
            1,
            "Default-connector integration probe must return the expected row shape.",
        )

        return database_url

    def _assert_dsn_equivalent_except_scheme_case(
        self,
        expected_url: str,
        actual_url: str,
    ) -> None:
        expected = urlparse(expected_url)
        actual = urlparse(actual_url)
        self.assertIn(actual.scheme.lower(), {"postgres", "postgresql"})
        self.assertIn(expected.scheme.lower(), {"postgres", "postgresql"})
        self.assertEqual(actual.username, expected.username)
        self.assertEqual(actual.password, expected.password)
        self.assertEqual(actual.hostname, expected.hostname)
        self.assertEqual(actual.port, expected.port)
        self.assertEqual(actual.path, expected.path)
        self.assertEqual(
            sorted(parse_qsl(actual.query, keep_blank_values=True)),
            sorted(parse_qsl(expected.query, keep_blank_values=True)),
        )


class DatabaseConnectionHarnessContractTests(unittest.TestCase):
    def test_default_connector_helper_skips_without_test_database_url(self) -> None:
        case = DatabaseConnectionBehaviorContractTests(methodName="runTest")

        with _patched_env(TEST_DATABASE_URL=None):
            with self.assertRaises(unittest.SkipTest):
                case._require_default_connector_test_database_url()

    def test_default_connector_helper_fails_when_explicit_database_url_lacks_driver(
        self,
    ) -> None:
        case = DatabaseConnectionBehaviorContractTests(methodName="runTest")
        module = sys.modules[__name__]

        with _patched_env(TEST_DATABASE_URL="postgresql:///postgres"), mock.patch.object(
            module,
            "_load_psycopg_for_default_connector",
            side_effect=ModuleNotFoundError("No module named 'psycopg'"),
        ):
            with self.assertRaisesRegex(AssertionError, "psycopg"):
                case._require_default_connector_test_database_url()

    def test_default_connector_helper_fails_when_explicit_database_url_is_unreachable(
        self,
    ) -> None:
        case = DatabaseConnectionBehaviorContractTests(methodName="runTest")
        module = sys.modules[__name__]
        fake_psycopg = SimpleNamespace(
            connect=mock.Mock(side_effect=RuntimeError("connection refused"))
        )

        with _patched_env(TEST_DATABASE_URL="postgresql:///postgres"), mock.patch.object(
            module,
            "_load_psycopg_for_default_connector",
            return_value=(fake_psycopg, object()),
        ):
            with self.assertRaisesRegex(
                AssertionError,
                "reachable local Postgres",
            ):
                case._require_default_connector_test_database_url()

    def test_default_connector_helper_does_not_hide_probe_assertion_failures(
        self,
    ) -> None:
        case = DatabaseConnectionBehaviorContractTests(methodName="runTest")
        module = sys.modules[__name__]
        fake_cursor = mock.Mock()
        fake_cursor.fetchone.return_value = {"value": 2}
        fake_connection = mock.Mock()
        fake_connection.execute.return_value = fake_cursor
        fake_context = mock.Mock()
        fake_context.__enter__ = mock.Mock(return_value=fake_connection)
        fake_context.__exit__ = mock.Mock(return_value=False)
        fake_psycopg = SimpleNamespace(connect=mock.Mock(return_value=fake_context))

        with _patched_env(TEST_DATABASE_URL="postgresql:///postgres"), mock.patch.object(
            module,
            "_load_psycopg_for_default_connector",
            return_value=(fake_psycopg, object()),
        ):
            with self.assertRaises(case.failureException):
                case._require_default_connector_test_database_url()


if __name__ == "__main__":
    unittest.main()
