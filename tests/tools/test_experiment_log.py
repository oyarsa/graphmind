"""Tests for the experiment log tool."""

from pathlib import Path

from paper.tools.exp_log import (
    find_experiment_dirs,
    get_experiment_name_from_path,
    get_possible_log_names,
    is_experiment_logged,
)


class TestGetExperimentNameFromPath:
    """Tests for get_experiment_name_from_path."""

    def test_orc_simple(self) -> None:
        path = Path("/home/user/project/output/eval_orc/ablation_sans")
        assert get_experiment_name_from_path(path) == "orc/ablation_sans"

    def test_orc_nested(self) -> None:
        path = Path("/output/eval_orc/eval/graph/dev/4o-mini/orc_10/graph/full-graph")
        assert (
            get_experiment_name_from_path(path)
            == "orc/eval/graph/dev/4o-mini/orc_10/graph/full-graph"
        )

    def test_peerread_simple(self) -> None:
        path = Path("/output/eval_peerread/ablation_full")
        assert get_experiment_name_from_path(path) == "peerread/ablation_full"

    def test_peerread_nested(self) -> None:
        path = Path("/output/eval_peerread/rec_structured")
        assert get_experiment_name_from_path(path) == "peerread/rec_structured"

    def test_unknown_path(self) -> None:
        path = Path("/some/other/path/experiment")
        assert get_experiment_name_from_path(path) == "/some/other/path/experiment"


class TestGetPossibleLogNames:
    """Tests for get_possible_log_names."""

    def test_orc_ablation_sans(self) -> None:
        path = Path("/output/eval_orc/ablation_sans")
        names = get_possible_log_names(path)
        assert "ablation_sans" in names
        assert "sans" in names
        assert "ablation_sans_orc" in names
        assert "sans_orc" in names

    def test_peerread_ablation_full(self) -> None:
        path = Path("/output/eval_peerread/ablation_full")
        names = get_possible_log_names(path)
        assert "ablation_full" in names
        assert "full" in names
        assert "ablation_full_peerread" in names
        assert "full_peerread" in names

    def test_norel_graph_special_case(self) -> None:
        path = Path("/output/eval_peerread/ablation_norel_graph")
        names = get_possible_log_names(path)
        assert "norel_graph" in names
        assert "norel" in names
        assert "norel_peerread" in names

    def test_no_ablation_prefix(self) -> None:
        path = Path("/output/eval_orc/verification_sans")
        names = get_possible_log_names(path)
        assert "verification_sans" in names
        assert "verification_sans_orc" in names

    def test_100item_experiments(self) -> None:
        path = Path("/output/eval_orc/100item_basic")
        names = get_possible_log_names(path)
        assert "100item_basic" in names
        assert "100item_basic_orc" in names


class TestIsExperimentLogged:
    """Tests for is_experiment_logged."""

    def test_direct_match(self) -> None:
        path = Path("/output/eval_orc/ablation_sans")
        logged = {"ablation_sans", "other_exp"}
        assert is_experiment_logged(path, logged)

    def test_match_without_ablation_prefix(self) -> None:
        path = Path("/output/eval_orc/ablation_sans")
        logged = {"sans", "other_exp"}
        assert is_experiment_logged(path, logged)

    def test_match_with_suffix(self) -> None:
        path = Path("/output/eval_orc/ablation_sans")
        logged = {"sans_orc", "other_exp"}
        assert is_experiment_logged(path, logged)

    def test_match_norel_graph_to_norel(self) -> None:
        path = Path("/output/eval_peerread/ablation_norel_graph")
        logged = {"norel_peerread", "other_exp"}
        assert is_experiment_logged(path, logged)

    def test_no_match(self) -> None:
        path = Path("/output/eval_orc/new_experiment")
        logged = {"sans_orc", "full_orc", "other_exp"}
        assert not is_experiment_logged(path, logged)

    def test_empty_logged_names(self) -> None:
        path = Path("/output/eval_orc/ablation_sans")
        assert not is_experiment_logged(path, set())


class TestFindExperimentDirs:
    """Tests for find_experiment_dirs."""

    def test_finds_metrics_in_root(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiment1"
        exp_dir.mkdir()
        (exp_dir / "metrics.json").touch()

        result = find_experiment_dirs(tmp_path)
        assert exp_dir in result

    def test_finds_metrics_in_run_subdirs(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiment2"
        run_dir = exp_dir / "run_0"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").touch()

        result = find_experiment_dirs(tmp_path)
        assert exp_dir in result
        assert run_dir not in result

    def test_finds_multiple_experiments(self, tmp_path: Path) -> None:
        exp1 = tmp_path / "exp1"
        exp1.mkdir()
        (exp1 / "metrics.json").touch()

        exp2 = tmp_path / "exp2"
        (exp2 / "run_0").mkdir(parents=True)
        (exp2 / "run_0" / "metrics.json").touch()

        result = find_experiment_dirs(tmp_path)
        assert exp1 in result
        assert exp2 in result
        assert len(result) == 2

    def test_ignores_non_run_subdirs(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiment"
        other_dir = exp_dir / "other_subdir"
        other_dir.mkdir(parents=True)
        (other_dir / "metrics.json").touch()

        result = find_experiment_dirs(tmp_path)
        assert other_dir in result
        assert exp_dir not in result

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = find_experiment_dirs(tmp_path)
        assert len(result) == 0

    def test_nested_structure(self, tmp_path: Path) -> None:
        nested = tmp_path / "eval" / "graph" / "experiment"
        (nested / "run_0").mkdir(parents=True)
        (nested / "run_0" / "metrics.json").touch()
        (nested / "run_1").mkdir()
        (nested / "run_1" / "metrics.json").touch()

        result = find_experiment_dirs(tmp_path)
        assert nested in result
        assert len(result) == 1
