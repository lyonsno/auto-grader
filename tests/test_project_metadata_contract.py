from __future__ import annotations

from pathlib import Path
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.version import Version
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

    def test_project_declares_psycopg_binary_v3_driver_for_local_postgres_bootstrap(
        self,
    ) -> None:
        data = self._load_pyproject()
        project = data.get("project") or {}
        dependency_entries = self._collect_project_dependencies(project)

        normalized_dependencies = [
            self._parse_dependency(entry) for entry in dependency_entries
        ]
        for name, extras, entry in normalized_dependencies:
            if (
                name == "psycopg"
                and "binary" in extras
                and self._is_psycopg_v3_spec(entry)
            ):
                return

        self.fail(
            "Project metadata must declare `psycopg[binary]` with a psycopg v3 "
            "version spec in runtime or optional dependencies so the documented "
            "local Postgres bootstrap target remains defined in-repo."
        )

    def test_psycopg_v3_spec_helper_accepts_valid_v3_ranges_and_rejects_bad_ones(
        self,
    ) -> None:
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3.2,<4"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3.2,<3.4"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>3,<4"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>3.2,<3.3"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]==3.2.1"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3,<4,!=3.2.1"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]==3.2.*"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3.2rc1,<4"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3.2.post1,<4"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3.2.dev1,<4"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3,<4; python_version >= '3.12'"))

        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]>30,<4"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]>=3.2,<5"))
        self.assertFalse(
            self._is_psycopg_v3_spec("psycopg[binary]>=3,<7,!=4.*,!=5.*")
        )
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]<=3.4"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]~=3"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]>=3,<4; python_version < '0'"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]"))

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

    def _is_psycopg_v3_spec(self, entry: str) -> bool:
        try:
            parsed_requirement = Requirement(entry)
        except InvalidRequirement:
            return False

        requirement_name = parsed_requirement.name.lower().replace("_", "-")
        if requirement_name != "psycopg":
            return False
        if parsed_requirement.marker is not None and not parsed_requirement.marker.evaluate():
            return False

        spec_text = str(parsed_requirement.specifier).replace(" ", "")
        if not spec_text:
            return False

        v3_candidates = self._candidate_versions_for_major(parsed_requirement, major=3)
        if not any(
            parsed_requirement.specifier.contains(candidate, prereleases=True)
            for candidate in v3_candidates
        ):
            return False

        for major in sorted(self._probe_majors_for_requirement(parsed_requirement)):
            if major == 3:
                continue
            non_v3_candidates = self._candidate_versions_for_major(
                parsed_requirement,
                major=major,
            )
            if any(
                parsed_requirement.specifier.contains(candidate, prereleases=True)
                for candidate in non_v3_candidates
            ):
                return False

        return True

    def _probe_majors_for_requirement(self, parsed_requirement: Requirement) -> set[int]:
        probe_majors = {0, 1, 2, 3, 4, 5}
        for specifier in parsed_requirement.specifier:
            for major in self._major_neighbors_for_specifier_version(specifier.version):
                probe_majors.add(major)
        return {major for major in probe_majors if major >= 0}

    def _candidate_versions_for_major(
        self,
        parsed_requirement: Requirement,
        *,
        major: int,
    ) -> set[Version]:
        candidate_texts = {
            f"{major}",
            f"{major}.0",
            f"{major}.0.1",
            f"{major}.0.1.post1",
            f"{major}.0.1.dev1",
            f"{major}.1",
            f"{major}.1.1",
            f"{major}.2a1",
            f"{major}.2b1",
            f"{major}.2rc1",
            f"{major}.2",
            f"{major}.2.post1",
            f"{major}.2.dev1",
            f"{major}.2.1",
            f"{major}.2.1.post1",
            f"{major}.2.1.dev1",
            f"{major}.3",
            f"{major}.3.1",
            f"{major}.9",
            f"{major}.9.1",
        }

        for specifier in parsed_requirement.specifier:
            candidate_texts.update(
                self._derived_candidate_texts_for_specifier(
                    version_text=specifier.version,
                    major=major,
                )
            )

        return {Version(text) for text in candidate_texts}

    def _major_neighbors_for_specifier_version(self, version_text: str) -> set[int]:
        if version_text.endswith(".*"):
            version_text = version_text[:-2]

        try:
            version = Version(version_text)
        except Exception:
            return set()

        if not version.release:
            return set()

        major = version.release[0]
        return {major - 1, major, major + 1}

    def _derived_candidate_texts_for_specifier(
        self,
        *,
        version_text: str,
        major: int,
    ) -> set[str]:
        candidate_texts: set[str] = set()
        if version_text.endswith(".*"):
            prefix = version_text[:-2]
            try:
                base_version = Version(prefix)
            except Exception:
                return candidate_texts
            if base_version.release[0] != major:
                return candidate_texts
            candidate_texts.add(prefix)
            candidate_texts.add(self._successor_release_text(base_version))
            return candidate_texts

        try:
            version = Version(version_text)
        except Exception:
            return candidate_texts
        if version.release[0] != major:
            return candidate_texts

        candidate_texts.add(str(version))
        release_text = ".".join(str(part) for part in version.release)
        candidate_texts.add(release_text)
        candidate_texts.add(f"{release_text}.post1")
        candidate_texts.add(f"{release_text}.dev1")
        candidate_texts.add(self._successor_release_text(version))
        return candidate_texts

    def _successor_release_text(self, version: Version) -> str:
        release = list(version.release)
        if len(release) == 1:
            return f"{release[0]}.0.1"
        if len(release) == 2:
            return f"{release[0]}.{release[1]}.1"
        release[-1] += 1
        return ".".join(str(part) for part in release)

    def _parse_version_tuple(self, version_text: str) -> tuple[int, ...]:
        return tuple(int(part) for part in version_text.split("."))

    def _prefix_upper_bound(self, version: tuple[int, ...]) -> tuple[int, ...]:
        incremented = list(version)
        incremented[-1] += 1
        return tuple(incremented)

    def _compare_versions(
        self,
        left: tuple[int, ...],
        right: tuple[int, ...],
    ) -> int:
        width = max(len(left), len(right))
        padded_left = left + (0,) * (width - len(left))
        padded_right = right + (0,) * (width - len(right))
        return (padded_left > padded_right) - (padded_left < padded_right)

    def _tighten_lower_bound(
        self,
        current_bound: tuple[int, ...] | None,
        current_inclusive: bool,
        new_bound: tuple[int, ...],
        new_inclusive: bool,
    ) -> tuple[tuple[int, ...], bool]:
        if current_bound is None:
            return new_bound, new_inclusive
        comparison = self._compare_versions(new_bound, current_bound)
        if comparison > 0:
            return new_bound, new_inclusive
        if comparison < 0:
            return current_bound, current_inclusive
        return current_bound, current_inclusive and new_inclusive

    def _tighten_upper_bound(
        self,
        current_bound: tuple[int, ...] | None,
        current_inclusive: bool,
        new_bound: tuple[int, ...],
        new_inclusive: bool,
    ) -> tuple[tuple[int, ...], bool]:
        if current_bound is None:
            return new_bound, new_inclusive
        comparison = self._compare_versions(new_bound, current_bound)
        if comparison < 0:
            return new_bound, new_inclusive
        if comparison > 0:
            return current_bound, current_inclusive
        return current_bound, current_inclusive and new_inclusive

    def _compatible_release_upper_bound(
        self,
        version: tuple[int, ...],
    ) -> tuple[int, ...]:
        if len(version) == 1:
            return (version[0] + 1, 0, 0)
        prefix = list(version[:-1])
        prefix[-1] += 1
        return tuple(prefix + [0])

    def _minimum_candidate_version(
        self,
        lower_bound: tuple[int, ...],
        lower_inclusive: bool,
    ) -> tuple[int, ...] | None:
        if lower_inclusive:
            return lower_bound
        return lower_bound + (0, 1)

    def _increment_version(self, version: tuple[int, ...]) -> tuple[int, ...]:
        if not version:
            return (1,)
        incremented = list(version)
        incremented[-1] += 1
        return tuple(incremented)

    def _version_is_within_bounds(
        self,
        version: tuple[int, ...],
        upper_bound: tuple[int, ...],
        upper_inclusive: bool,
    ) -> bool:
        comparison = self._compare_versions(version, upper_bound)
        if comparison < 0:
            return True
        if comparison > 0:
            return False
        return upper_inclusive

    def _matching_excluded_prefix_range(
        self,
        candidate: tuple[int, ...],
        excluded_prefix_ranges: list[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        for start, end in excluded_prefix_ranges:
            if (
                self._compare_versions(candidate, start) >= 0
                and self._compare_versions(candidate, end) < 0
            ):
                return start, end
        return None

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
