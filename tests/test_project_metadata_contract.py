from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
_DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[([^\]]+)\])?")
_SPECIFIER_RE = re.compile(r"^\s*(<=|>=|==|!=|~=|<|>)\s*(.+?)\s*$")
_VERSION_RE = re.compile(
    r"^(?P<release>\d+(?:\.\d+)*)"
    r"(?:(?P<pre_tag>a|b|rc)(?P<pre_num>\d+)|"
    r"\.post(?P<post_num>\d+)|"
    r"\.dev(?P<dev_num>\d+))?$"
)
_PHASE_ORDER = {
    "dev": 0,
    "a": 1,
    "b": 2,
    "rc": 3,
    "final": 4,
    "post": 5,
}


@dataclass(frozen=True)
class _ParsedVersion:
    release: tuple[int, ...]
    raw_release: tuple[int, ...]
    phase: str = "final"
    phase_num: int = 0


@dataclass(frozen=True)
class _ParsedSpecifier:
    operator: str
    version_text: str
    version: _ParsedVersion | None = None
    wildcard_prefix: tuple[int, ...] | None = None


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
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]~=3.0"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]~=3.0.0"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3,<4,!=3.0.*"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3.2rc1,<4"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3.2.post1,<4"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>=3.2.dev1,<4"))

        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]>30,<4"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]>=3.2,<5"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]>=3.1,==3.0.*"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]==3.2a1,<3.2"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]<=3.2.post1,>3.2"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]>=3.2rc1,~=3.1.0"))
        self.assertTrue(self._is_psycopg_v3_spec("psycopg[binary]>3.2rc1,<=3.2.post1"))
        self.assertFalse(
            self._is_psycopg_v3_spec("psycopg[binary]>=3,<7,!=4.*,!=5.*")
        )
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]<=3.4"))
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]~=3"))
        self.assertFalse(
            self._is_psycopg_v3_spec("psycopg[binary]>=3,<4; python_version >= '3.12'")
        )
        self.assertFalse(self._is_psycopg_v3_spec("psycopg[binary]>=3,<4; python_version < '0'"))
        self.assertFalse(
            self._is_psycopg_v3_spec("psycopg[binary]>=3,<4; platform_machine == 'arm64'")
        )
        self.assertFalse(
            self._is_psycopg_v3_spec("psycopg[binary]>=3,<4; platform_machine == 'x86_64'")
        )
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
        requirement_text, marker_text = self._split_requirement_and_marker(entry)
        if marker_text is not None:
            return False

        requirement_name, extras, _ = self._parse_dependency(requirement_text)
        if requirement_name != "psycopg" or "binary" not in extras:
            return False
        dependency_match = _DEPENDENCY_NAME_RE.match(requirement_text)
        if dependency_match is None:
            return False

        spec_text = requirement_text[dependency_match.end() :].strip()
        if not spec_text:
            return False
        try:
            specifiers = self._parse_specifiers(spec_text)
        except ValueError:
            return False

        v3_candidates = self._candidate_versions_for_major(specifiers, major=3)
        if not any(
            self._specifier_set_contains(specifiers, candidate)
            for candidate in v3_candidates
        ):
            return False

        for major in sorted(self._probe_majors_for_specifiers(specifiers)):
            if major == 3:
                continue
            non_v3_candidates = self._candidate_versions_for_major(
                specifiers,
                major=major,
            )
            if any(
                self._specifier_set_contains(specifiers, candidate)
                for candidate in non_v3_candidates
            ):
                return False

        return True

    def _split_requirement_and_marker(self, entry: str) -> tuple[str, str | None]:
        requirement_text, separator, marker_text = entry.partition(";")
        if not separator:
            return entry.strip(), None
        return requirement_text.strip(), marker_text.strip()

    def _parse_specifiers(self, spec_text: str) -> list[_ParsedSpecifier]:
        tokens = [token.strip() for token in spec_text.split(",")]
        if not tokens or any(not token for token in tokens):
            raise ValueError("Specifier list must not be blank.")
        return [self._parse_specifier(token) for token in tokens]

    def _parse_specifier(self, token: str) -> _ParsedSpecifier:
        match = _SPECIFIER_RE.fullmatch(token)
        if match is None:
            raise ValueError(f"Unsupported specifier token: {token!r}")

        operator, version_text = match.groups()
        if version_text.endswith(".*"):
            if operator not in {"==", "!="}:
                raise ValueError("Wildcard prefix matches require == or !=.")
            return _ParsedSpecifier(
                operator=operator,
                version_text=version_text,
                wildcard_prefix=self._parse_release_prefix(version_text[:-2]),
            )

        version = self._parse_version(version_text)
        if operator == "~=" and len(version.raw_release) < 2:
            raise ValueError("Compatible-release pins must specify at least major.minor.")
        return _ParsedSpecifier(
            operator=operator,
            version_text=version_text,
            version=version,
        )

    def _probe_majors_for_specifiers(self, specifiers: list[_ParsedSpecifier]) -> set[int]:
        probe_majors = {0, 1, 2, 3, 4, 5}
        for specifier in specifiers:
            for major in self._major_neighbors_for_specifier(specifier):
                probe_majors.add(major)
        return {major for major in probe_majors if major >= 0}

    def _candidate_versions_for_major(
        self,
        specifiers: list[_ParsedSpecifier],
        *,
        major: int,
    ) -> set[_ParsedVersion]:
        candidate_texts = {
            f"{major}",
            f"{major}.0",
            f"{major}.0.1",
            f"{major}.1",
            f"{major}.1.1",
            f"{major}.2",
            f"{major}.2.1",
            f"{major}.3",
            f"{major}.3.1",
            f"{major}.9",
            f"{major}.9.1",
        }

        for specifier in specifiers:
            candidate_texts.update(
                self._derived_candidate_texts_for_specifier(specifier, major=major)
            )

        return {self._parse_version(text) for text in candidate_texts}

    def _major_neighbors_for_specifier(self, specifier: _ParsedSpecifier) -> set[int]:
        if specifier.wildcard_prefix is not None:
            major = specifier.wildcard_prefix[0]
        else:
            assert specifier.version is not None
            major = specifier.version.release[0]
        return {major - 1, major, major + 1}

    def _derived_candidate_texts_for_specifier(
        self,
        specifier: _ParsedSpecifier,
        *,
        major: int,
    ) -> set[str]:
        candidate_texts: set[str] = set()
        if specifier.wildcard_prefix is not None:
            prefix = specifier.wildcard_prefix
            if prefix[0] != major:
                return candidate_texts
            candidate_texts.add(self._release_text(prefix))
            candidate_texts.add(self._successor_release_text(prefix))
            return candidate_texts

        assert specifier.version is not None
        version = specifier.version
        if version.release[0] != major:
            return candidate_texts

        candidate_texts.add(self._format_version(version))
        release_text = self._release_text(version.release)
        candidate_texts.add(release_text)
        if version.phase in {"post", "dev"}:
            candidate_texts.add(f"{release_text}.{version.phase}1")
        candidate_texts.add(self._successor_release_text(version.release))
        return candidate_texts

    def _parse_release_prefix(self, version_text: str) -> tuple[int, ...]:
        if not re.fullmatch(r"\d+(?:\.\d+)*", version_text):
            raise ValueError(f"Unsupported wildcard prefix: {version_text!r}")
        return tuple(int(part) for part in version_text.split("."))

    def _parse_version(self, version_text: str) -> _ParsedVersion:
        match = _VERSION_RE.fullmatch(version_text)
        if match is None:
            raise ValueError(f"Unsupported version text: {version_text!r}")

        raw_release = tuple(int(part) for part in match.group("release").split("."))
        release = self._normalize_release(raw_release)
        if match.group("pre_tag") is not None:
            return _ParsedVersion(
                release=release,
                raw_release=raw_release,
                phase=match.group("pre_tag"),
                phase_num=int(match.group("pre_num")),
            )
        if match.group("post_num") is not None:
            return _ParsedVersion(
                release=release,
                raw_release=raw_release,
                phase="post",
                phase_num=int(match.group("post_num")),
            )
        if match.group("dev_num") is not None:
            return _ParsedVersion(
                release=release,
                raw_release=raw_release,
                phase="dev",
                phase_num=int(match.group("dev_num")),
            )
        return _ParsedVersion(release=release, raw_release=raw_release)

    def _normalize_release(self, release: tuple[int, ...]) -> tuple[int, ...]:
        normalized = list(release)
        while len(normalized) > 1 and normalized[-1] == 0:
            normalized.pop()
        return tuple(normalized)

    def _release_text(self, release: tuple[int, ...]) -> str:
        return ".".join(str(part) for part in release)

    def _format_version(self, version: _ParsedVersion) -> str:
        release_text = self._release_text(version.release)
        if version.phase == "final":
            return release_text
        if version.phase in {"a", "b", "rc"}:
            return f"{release_text}{version.phase}{version.phase_num}"
        return f"{release_text}.{version.phase}{version.phase_num}"

    def _successor_release_text(self, release: tuple[int, ...]) -> str:
        release = list(release)
        if len(release) == 1:
            return f"{release[0]}.0.1"
        if len(release) == 2:
            return f"{release[0]}.{release[1]}.1"
        release[-1] += 1
        return ".".join(str(part) for part in release)

    def _compare_versions(
        self,
        left: _ParsedVersion,
        right: _ParsedVersion,
    ) -> int:
        width = max(len(left.release), len(right.release))
        padded_left = left.release + (0,) * (width - len(left.release))
        padded_right = right.release + (0,) * (width - len(right.release))
        release_comparison = (padded_left > padded_right) - (padded_left < padded_right)
        if release_comparison != 0:
            return release_comparison

        phase_comparison = (_PHASE_ORDER[left.phase] > _PHASE_ORDER[right.phase]) - (
            _PHASE_ORDER[left.phase] < _PHASE_ORDER[right.phase]
        )
        if phase_comparison != 0:
            return phase_comparison
        return (left.phase_num > right.phase_num) - (left.phase_num < right.phase_num)

    def _specifier_set_contains(
        self,
        specifiers: list[_ParsedSpecifier],
        candidate: _ParsedVersion,
    ) -> bool:
        return all(self._specifier_contains(specifier, candidate) for specifier in specifiers)

    def _specifier_contains(
        self,
        specifier: _ParsedSpecifier,
        candidate: _ParsedVersion,
    ) -> bool:
        if specifier.wildcard_prefix is not None:
            padded_release = candidate.raw_release + (0,) * (
                len(specifier.wildcard_prefix) - len(candidate.raw_release)
            )
            matches = (
                padded_release[: len(specifier.wildcard_prefix)]
                == specifier.wildcard_prefix
            )
            if specifier.operator == "==":
                return matches
            if specifier.operator == "!=":
                return not matches
            raise AssertionError("Wildcard prefixes should only use == or !=.")

        assert specifier.version is not None
        comparison = self._compare_versions(candidate, specifier.version)
        if specifier.operator == "==":
            return comparison == 0
        if specifier.operator == "!=":
            return comparison != 0
        if specifier.operator == ">":
            return (
                comparison > 0
                and not self._is_same_release_post_of_exclusive_baseline(
                    candidate,
                    specifier.version,
                )
            )
        if specifier.operator == ">=":
            return comparison >= 0
        if specifier.operator == "<":
            return (
                comparison < 0
                and not self._is_same_release_prerelease_of_final_ceiling(
                    candidate,
                    specifier.version,
                )
            )
        if specifier.operator == "<=":
            return comparison <= 0
        if specifier.operator == "~=":
            upper_bound = _ParsedVersion(
                release=self._normalize_release(
                    self._compatible_release_upper_bound(specifier.version.raw_release)
                ),
                raw_release=self._compatible_release_upper_bound(specifier.version.raw_release),
            )
            upper_comparison = self._compare_versions(candidate, upper_bound)
            return (
                comparison >= 0
                and upper_comparison < 0
                and not self._is_same_release_prerelease_of_final_ceiling(
                    candidate,
                    upper_bound,
                )
            )
        raise AssertionError(f"Unsupported specifier operator: {specifier.operator!r}")

    def _is_same_release_post_of_exclusive_baseline(
        self,
        candidate: _ParsedVersion,
        baseline: _ParsedVersion,
    ) -> bool:
        return (
            baseline.phase != "post"
            and candidate.release == baseline.release
            and candidate.phase == "post"
        )

    def _is_same_release_prerelease_of_final_ceiling(
        self,
        candidate: _ParsedVersion,
        ceiling: _ParsedVersion,
    ) -> bool:
        return (
            ceiling.phase == "final"
            and candidate.release == ceiling.release
            and candidate.phase in {"dev", "a", "b", "rc"}
        )

    def _compatible_release_upper_bound(
        self,
        version: tuple[int, ...],
    ) -> tuple[int, ...]:
        if len(version) == 1:
            return (version[0] + 1, 0, 0)
        prefix = list(version[:-1])
        prefix[-1] += 1
        return tuple(prefix + [0])

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
