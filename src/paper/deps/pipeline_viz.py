"""Generate DAG visualizations from pipeline scripts."""

import ast
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from paper.util import seqcat
from paper.util.cli import die
from paper.util.typing import TSeq


class NodeType(StrEnum):
    """Types of pipeline nodes for coloring."""

    DATA_PROCESSING = "data"
    GPT_OPERATIONS = "gpt"
    GRAPH_BUILDING = "graph"
    EVALUATION = "eval"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PipelineStep:
    """Represents a single step in the pipeline."""

    name: str
    command: TSeq[str]
    output_file: Path
    input_files: TSeq[Path]
    node_type: NodeType = NodeType.UNKNOWN
    layer: int = field(default=-1, compare=False)


@dataclass(frozen=True)
class Pipeline:
    """Represents the entire pipeline structure."""

    steps: Sequence[PipelineStep]
    file_to_producer: Mapping[Path, PipelineStep]
    file_to_consumers: Mapping[Path, TSeq[PipelineStep]]


@dataclass(frozen=True)
class ParseState:
    """Immutable state for parsing pipeline scripts."""

    steps: TSeq[PipelineStep]
    current_outputs: Mapping[str, Path]
    current_title: str | None


def parse_pipeline_script(script_path: Path) -> Pipeline:
    """Parse a pipeline script to extract structure."""
    tree = ast.parse(script_path.read_text())
    parse_state = _parse_ast(tree)
    initial_pipeline = _build_pipeline_from_steps(parse_state.steps)
    return _fix_input_dependencies(initial_pipeline)


def _parse_ast(tree: ast.AST) -> ParseState:
    """Parse AST and return immutable parse state."""
    parser = _PipelineASTParser()
    parser.visit(tree)

    return ParseState(
        steps=tuple(parser.steps),
        current_outputs=dict(parser.current_outputs),
        current_title=parser.current_title,
    )


class _PipelineASTParser(ast.NodeVisitor):
    """Parse pipeline steps from Python script AST."""

    def __init__(self) -> None:
        self.steps: list[PipelineStep] = []
        self.current_outputs: dict[str, Path] = {}
        self.current_title: str | None = None

    def visit_Call(self, node: ast.Call) -> None:
        """Extract _checkrun calls and title calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id == "title" and node.args:
                # Extract title for the next step
                if isinstance(node.args[0], ast.Constant):
                    self.current_title = node.args[0].value

            elif node.func.id == "_checkrun":
                step = self._parse_checkrun(node)
                if step:
                    self.steps.append(step)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variable assignments for file paths."""
        if not isinstance(node.value, ast.BinOp) or not isinstance(
            node.value.op, ast.Div
        ):
            self.generic_visit(node)
            return

        # Track Path assignments like: processed = output_dir / "orc_merged.json.zst"
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            if path_str := _extract_path_from_binop(node.value):
                # Simplify the path for display
                simplified_path = _simplify_path_str(path_str)
                self.current_outputs[target.id] = Path(simplified_path)

        self.generic_visit(node)

    def _parse_checkrun(self, node: ast.Call) -> PipelineStep | None:
        """Parse a _checkrun call to extract pipeline step info."""
        if not node.args or len(node.args) < 2:
            return None

        # First arg is the output path
        output_var = node.args[0].id if isinstance(node.args[0], ast.Name) else None
        output_path = self.current_outputs.get(output_var) if output_var else None

        if not output_path:
            return None

        command = _extract_command_from_args(node.args[1:], self.current_outputs)

        # Create step
        step_name = self.current_title or " ".join(command[:3])
        input_files = _extract_input_files(command, self.current_outputs)

        step = PipelineStep(
            name=step_name,
            command=tuple(command),
            output_file=output_path,
            input_files=tuple(input_files),
            node_type=_infer_node_type(command),
        )

        self.current_title = None

        return step


def _extract_path_from_binop(node: ast.BinOp) -> str | None:
    """Extract path string from Path division operations."""
    parts: list[str] = []

    def extract_parts(n: ast.AST) -> None:
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            extract_parts(n.left)
            extract_parts(n.right)
        elif isinstance(n, ast.Constant):
            parts.append(str(n.value))
        elif isinstance(n, ast.Name):
            parts.append(f"${n.id}")

    extract_parts(node)
    return "/".join(parts) if parts else None


def _simplify_path_str(path_str: str) -> str:
    """Simplify path string for display."""
    for prefix in [
        "$output_dir/",
        "$info_main_dir/",
        "$info_ref_dir/",
        "$recommended_dir/",
        "$subset_dir/",
        "$context_dir/",
        "$peer_terms_dir/",
        "$s2_terms_dir/",
        "$peter_graph_dir/",
        "$petersum_dir/",
        "$scimon_graph_dir/",
        "$acu_s2_dir/",
        "$acu_peerread_dir/",
        "$acu_query_dir/",
    ]:
        if path_str.startswith(prefix):
            # Extract the dir name and keep it for context
            dir_name = prefix.strip("$/")
            return f"{dir_name}/{path_str[len(prefix) :]}"
    return path_str


def _extract_command_from_args(
    args: list[ast.expr], current_outputs: dict[str, Path]
) -> list[str]:
    """Extract command strings from AST arguments."""
    command: list[str] = []

    for arg in args:
        if isinstance(arg, ast.Constant):
            command.append(str(arg.value))
        elif isinstance(arg, ast.Name):
            # This might be an input file variable
            if arg.id in current_outputs:
                command.append(str(current_outputs[arg.id]))
            else:
                command.append(f"${arg.id}")

    return command


def _extract_input_files(
    command: list[str], current_outputs: dict[str, Path]
) -> list[Path]:
    """Extract input file paths from command."""
    input_files: list[Path] = []

    for part in command:
        if part.startswith("$"):
            var_name = part[1:]
            if var_name in current_outputs:
                input_files.append(current_outputs[var_name])

    return input_files


def _infer_node_type(command: list[str]) -> NodeType:
    """Infer node type from command."""
    if "gpt" in command:
        return NodeType.GPT_OPERATIONS
    elif any(x in command for x in ["orc", "s2", "construct"]):
        return NodeType.DATA_PROCESSING
    elif any(x in command for x in ["peter", "scimon"]):
        return NodeType.GRAPH_BUILDING
    elif any(x in command for x in ["nova", "baselines"]):
        return NodeType.EVALUATION
    return NodeType.UNKNOWN


def _build_pipeline_from_steps(steps: Sequence[PipelineStep]) -> Pipeline:
    """Build pipeline structure from steps."""
    file_to_producer: dict[Path, PipelineStep] = {}
    file_to_consumers: dict[Path, list[PipelineStep]] = defaultdict(list)

    for step in steps:
        file_to_producer[step.output_file] = step
        for input_file in step.input_files:
            file_to_consumers[input_file].append(step)

    # Convert lists to tuples for immutability
    immutable_consumers = {
        path: tuple(consumers) for path, consumers in file_to_consumers.items()
    }

    return Pipeline(
        steps=steps,
        file_to_producer=file_to_producer,
        file_to_consumers=immutable_consumers,
    )


def _fix_input_dependencies(pipeline: Pipeline) -> Pipeline:
    """Return new pipeline with fixed input dependencies."""
    # Map simplified paths to steps
    path_to_step = {_simplify_path(step.output_file): step for step in pipeline.steps}

    # Create new steps with fixed inputs
    fixed_steps = [
        # Create new step with updated inputs if additional_inputs exist
        replace(step, input_files=seqcat(step.input_files, additional_inputs))
        if (additional_inputs := _find_additional_inputs(step, path_to_step))
        else step
        for step in pipeline.steps
    ]
    # Build new pipeline with fixed steps
    return _build_pipeline_from_steps(fixed_steps)


def _find_additional_inputs(
    step: PipelineStep, path_to_step: dict[str, PipelineStep]
) -> list[Path]:
    """Find additional input files for a step based on command analysis."""
    additional_inputs: list[Path] = []

    for cmd_part in step.command:
        simple_cmd = _simplify_path(Path(cmd_part))
        if simple_cmd in path_to_step:
            producer = path_to_step[simple_cmd]
            if producer != step and producer.output_file not in step.input_files:
                additional_inputs.append(producer.output_file)

    return additional_inputs


def _simplify_path(path: Path) -> str:
    """Simplify path for matching (remove $output_dir, etc)."""
    path_str = str(path)

    if path_str.startswith("$"):
        # Remove variable prefix
        parts = path_str.split("/", 1)
        return parts[1] if len(parts) > 1 else path_str

    return path_str


def assign_topological_layers(pipeline: Pipeline) -> Pipeline:
    """Return new pipeline with steps assigned to topological layers."""
    # Build adjacency information
    adjacency = _build_adjacency_info(pipeline)

    # Compute layers for each step
    step_layers = _compute_topological_layers(pipeline.steps, adjacency)

    # Create new steps with updated layers
    updated_steps = tuple(
        replace(step, layer=step_layers[i]) for i, step in enumerate(pipeline.steps)
    )

    # Return new pipeline with updated steps
    return _build_pipeline_from_steps(updated_steps)


@dataclass(frozen=True)
class AdjacencyInfo:
    """Adjacency information for topological sorting."""

    graph: Mapping[int, TSeq[int]]
    in_degree: Mapping[int, int]


def _build_adjacency_info(pipeline: Pipeline) -> AdjacencyInfo:
    """Build adjacency information for the pipeline graph."""
    step_to_idx = {step: i for i, step in enumerate(pipeline.steps)}
    graph: dict[int, list[int]] = defaultdict(list)
    in_degree: dict[int, int] = defaultdict(int)

    for i, step in enumerate(pipeline.steps):
        for input_file in step.input_files:
            if input_file in pipeline.file_to_producer:
                producer = pipeline.file_to_producer[input_file]
                producer_idx = step_to_idx[producer]
                graph[producer_idx].append(i)
                in_degree[i] += 1

    # Convert to immutable structure
    immutable_graph = {k: tuple(v) for k, v in graph.items()}

    return AdjacencyInfo(graph=immutable_graph, in_degree=dict(in_degree))


def _compute_topological_layers(
    steps: Sequence[PipelineStep], adjacency: AdjacencyInfo
) -> dict[int, int]:
    """Compute topological layer for each step."""
    layers: dict[int, int] = {}

    def compute_layer(idx: int) -> int:
        if idx in layers:
            return layers[idx]

        max_pred_layer = -1
        for pred_idx, successors in adjacency.graph.items():
            if idx in successors:
                max_pred_layer = max(max_pred_layer, compute_layer(pred_idx))

        layers[idx] = max_pred_layer + 1
        return layers[idx]

    # Compute layer for all nodes
    for i in range(len(steps)):
        compute_layer(i)

    return layers


def generate_mermaid_dag(pipeline: Pipeline) -> str:
    """Generate Mermaid DAG visualization."""
    lines = ["graph TD"]

    # Node style classes
    lines.extend([
        "    classDef data fill:#4A90E2,stroke:#2E5C8A,color:#fff",
        "    classDef gpt fill:#50C878,stroke:#2F7B4F,color:#fff",
        "    classDef graphNode fill:#9B59B6,stroke:#6B3F86,color:#fff",
        "    classDef eval fill:#FF8C42,stroke:#BF5F1F,color:#fff",
        "    classDef unknown fill:#95A5A6,stroke:#7F8C8D,color:#fff",
        "",
    ])

    # Generate nodes
    step_to_id = {step: f"step{i}" for i, step in enumerate(pipeline.steps)}

    for step, step_id in step_to_id.items():
        # Simplify file name for display
        file_name = step.output_file.name
        if "/" in str(step.output_file):
            file_name = str(step.output_file).split("/")[-2] + "/" + file_name

        # Node definition
        lines.append(f'    {step_id}["{step.name}<br/>{file_name}"]')

    # Add edges
    for step in pipeline.steps:
        step_id = step_to_id[step]
        for input_file in step.input_files:
            if input_file in pipeline.file_to_producer:
                producer = pipeline.file_to_producer[input_file]
                producer_id = step_to_id[producer]
                lines.append(f"    {producer_id} --> {step_id}")

    # Apply node type styling
    for step, step_id in step_to_id.items():
        node_class = step.node_type.value
        if node_class == "graph":
            node_class = "graphNode"
        lines.append(f"    class {step_id} {node_class}")

    return "\n".join(lines)


def save_mermaid_png(diagram: str, output_file: Path) -> None:
    """Convert Mermaid diagram to PNG using mermaid-cli."""
    try:
        subprocess.run(
            ["mmdc", "-i", "-", "-o", str(output_file), "-w", "3200", "-H", "2400"],
            input=diagram.encode(),
            check=True,
        )
    except FileNotFoundError:
        print(
            "mermaid-cli not found. Install with: npm install -g @mermaid-js/mermaid-cli"
        )
        raise
    except subprocess.CalledProcessError as e:
        print(f"mermaid-cli failed: {e}")
        raise


def main(
    pipeline_script: Annotated[
        Path,
        typer.Argument(
            help="Path to pipeline script (e.g., run_full_orc_pipeline.py)."
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(help="Output path for DAG visualization (PNG)."),
    ],
) -> None:
    """Generate DAG and dependency matrix visualizations from pipeline script."""
    if not pipeline_script.exists():
        die(f"Error: Pipeline script {pipeline_script} not found")

    # Chain pure functions - each returns a new object
    pipeline = parse_pipeline_script(pipeline_script)
    print(f"Found {len(pipeline.steps)} pipeline steps")

    pipeline_with_layers = assign_topological_layers(pipeline)
    mermaid_code = generate_mermaid_dag(pipeline_with_layers)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_mermaid_png(mermaid_code, output)
