from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
_DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[([^\]]+)\])?")


class ProjectMetadataContractTests(unittest.TestCase):
    """Contract tests for local project bootstrap metadata."""

    def test_pyproject_declares_installable_project_metadata(self) -> None:
        data = self._load_pyproject()

        build_system = data.get("build-system")
        self.assertIsInstance(
            build_system,
            dict,
            "pyproject.toml must declare a [build-system] table.",
        )
        self.assertTrue(
            build_system.get("requires"),
            "pyproject.toml must declare build-system requirements.",
        )
        self.assertTrue(
            build_system.get("build-backend"),
            "pyproject.toml must declare a build backend.",
        )

        project = data.get("project")
        self.assertIsInstance(
            project,
            dict,
            "pyproject.toml must declare a [project] table for installable metadata.",
        )
        self.assertEqual(
            project.get("name"),
            "auto-grader",
            "Project metadata should use the repository package name `auto-grader`.",
        )
        self.assertTrue(
            project.get("requires-python"),
            "Project metadata must declare a Python version requirement.",
        )
        self.assertTrue(
            project.get("version")
            or "version" in project.get("dynamic", []),
            "Project metadata must declare a version or mark it as dynamic.",
        )

    def test_project_declares_psycopg_binary_driver_for_local_postgres_bootstrap(
        self,
    ) -> None:
        data = self._load_pyproject()
        project = data.get("project") or {}
        dependency_entries = self._collect_project_dependencies(project)

        normalized_dependencies = [
            self._parse_dependency(entry) for entry in dependency_entries
        ]
        for name, extras, _ in normalized_dependencies:
            if name == "psycopg" and "binary" in extras:
                return

        self.fail(
            "Project metadata must declare `psycopg[binary]` in runtime or "
            "optional dependencies so the documented v3 local Postgres bootstrap "
            "target remains defined in-repo."
        )

    def _load_pyproject(self) -> dict[str, object]:
        self.assertTrue(
            PYPROJECT_PATH.exists(),
            "pyproject.toml must exist so local setup and connector dependencies are "
            "declared in-repo.",
        )
        with PYPROJECT_PATH.open("rb") as handle:
            return tomllib.load(handle)

    def _parse_dependency(self, entry: object) -> tuple[str, set[str], str]:
        self.assertIsInstance(entry, str, "Each dependency entry must be a string.")
        match = _DEPENDENCY_NAME_RE.match(entry)
        self.assertIsNotNone(
            match,
            f"Dependency entry must start with a package name: {entry!r}",
        )
        assert match is not None
        raw_name, raw_extras = match.groups()
        normalized_name = raw_name.lower().replace("_", "-")
        extras = {
            extra.strip().lower()
            for extra in (raw_extras or "").split(",")
            if extra.strip()
        }
        return normalized_name, extras, entry

    def _collect_project_dependencies(self, project: dict[str, object]) -> list[object]:
        dependency_entries: list[object] = []

        dependencies = project.get("dependencies")
        if dependencies is not None:
            self.assertIsInstance(
                dependencies,
                list,
                "Runtime dependencies must be declared as a list when "
                "[project].dependencies is present.",
            )
            dependency_entries.extend(dependencies)

        optional_dependencies = project.get("optional-dependencies")
        if optional_dependencies is not None:
            self.assertIsInstance(
                optional_dependencies,
                dict,
                "Optional dependency groups must be declared as a table when "
                "[project].optional-dependencies is present.",
            )
            for group_name, entries in optional_dependencies.items():
                self.assertIsInstance(
                    entries,
                    list,
                    "Each optional dependency group must be a list of dependency "
                    f"strings, got {group_name!r}.",
                )
                dependency_entries.extend(entries)

        return dependency_entries


if __name__ == "__main__":
    unittest.main()
