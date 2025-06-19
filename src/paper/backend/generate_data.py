"""Generate synthetic data for paper explorer."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from paper.backend.model import Model, Paper, PaperId, Related, RelatedType

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    output_file: Annotated[
        Path, typer.Argument(help="Path to save JSON with generated data.")
    ],
    num_papers: Annotated[
        int,
        typer.Option("--num-papers", "-n", help="Number of papers to generate."),
    ] = 5000,
    seed: Annotated[int, typer.Option(help="Seed to random number generator.")] = 0,
) -> None:
    """Generate synthetic paper data and save as JSON.

    Creates a collection of academic papers with realistic citation and semantic
    relationships for testing and development purposes.

    Args:
        output_file: Path where JSON data will be saved.
        num_papers: Total number of papers to generate.
        seed: Random seed for reproducible generation.
    """
    rng = random.Random(seed)

    print("Starting data generation...")
    database = _generate_static_database(rng, num_papers)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(database.model_dump_json())

    _analyse_database(database, output_file)


class StaticDatabase(Model):
    """Container for generated paper data.

    Represents a static dataset with papers and their relationships,
    mimicking the structure of the actual database.

    Attributes:
        papers: Collection of generated paper records.
        related: Collection of relationships between papers.
    """

    papers: Sequence[Paper]
    related: Sequence[Related]


def _generate_paper_title(rng: random.Random, topic: str) -> str:
    """Generate a realistic paper title for a given topic.

    Args:
        rng: Random number generator.
        topic: Academic topic for the paper.

    Returns:
        Generated paper title string.
    """
    method = rng.choice(METHODS)
    adjective = rng.choice(ADJECTIVES)
    subtitle = rng.choice(SUBTITLES)
    return f"{adjective} {topic} {method}{subtitle}"


def _generate_authors(rng: random.Random) -> list[str]:
    """Generate a list of unique author names.

    Args:
        rng: Random number generator.

    Returns:
        List of 2-5 unique author names.
    """
    num_authors = rng.randint(2, 5)
    authors: list[str] = []
    used_names: set[str] = set()

    for _ in range(num_authors):
        full_name: str
        while True:
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            full_name = f"{first} {last}"

            if full_name not in used_names:
                break

        used_names.add(full_name)
        authors.append(full_name)

    return authors


def _generate_abstract(topic: str) -> str:
    """Get a template abstract for the given topic.

    Args:
        topic: Academic topic for the paper.

    Returns:
        Abstract text appropriate for the topic.
    """
    return ABSTRACTS.get(topic, ABSTRACTS["Machine Learning"])


def _generate_additional_papers(rng: random.Random, count: int) -> list[Paper]:
    """Generate additional synthetic papers beyond the core papers.

    Args:
        rng: Random number generator.
        count: Number of papers to generate.

    Returns:
        List of generated paper objects.
    """
    papers: list[Paper] = []

    for i in range(count):
        topic = rng.choice(TOPICS)
        venue = rng.choice(VENUES)
        year = rng.randint(2010, 2023)
        citation_count = rng.randint(50, 5000)

        base_id = f"{year}-{i:0>5}"
        paper_id = PaperId(f"unseen-arxiv-{base_id}")
        doi = f"10.314159/{paper_id}"
        pdf_url = f"https://unreal-arxiv.org/pdf/{base_id}.pdf"

        papers.append(
            Paper(
                id=paper_id,
                title=_generate_paper_title(rng, topic),
                year=year,
                authors=_generate_authors(rng),
                abstract=_generate_abstract(topic),
                venue=f"{venue} {year}",
                citation_count=citation_count,
                doi=doi,
                pdf_url=pdf_url,
            )
        )

    return papers


def _generate_citation_relationships(
    rng: random.Random, papers: Sequence[Paper]
) -> list[Related]:
    """Generate realistic citation relationships between papers.

    Creates citation links where newer papers cite older papers,
    with similarity scores based on recency and random variation.

    Args:
        rng: Random number generator.
        papers: Collection of papers to create citations between.

    Returns:
        List of citation relationships.
    """
    relationships: list[Related] = []

    for paper in papers:
        potential_cited = [
            other
            for other in papers
            if other.year <= paper.year and other.id != paper.id
        ]
        if not potential_cited:
            continue

        num_citations = rng.randint(10, 25)
        cited = rng.sample(potential_cited, min(len(potential_cited), num_citations))

        for cited_paper in cited:
            # Generate more realistic citation similarity scores
            # Base similarity between 0.4-0.8, with some variation
            base_similarity = 0.4 + rng.random() * 0.4

            # Add small bonus for recency (max 0.1)
            year_diff = paper.year - cited_paper.year
            recency_bonus = max(0, 0.1 - year_diff * 0.02)

            similarity = base_similarity + recency_bonus

            relationships.append(
                Related(
                    source=cited_paper.id,
                    target=paper.id,
                    type_=RelatedType.CITATION,
                    similarity=min(similarity, 1.0),
                )
            )

    return relationships


def _generate_semantic_relationships(
    rng: random.Random, papers: Sequence[Paper]
) -> list[Related]:
    """Generate semantic similarity relationships between papers.

    Creates symmetric relationships based on content similarity,
    ensuring each paper has a realistic number of connections.

    Args:
        rng: Random number generator.
        papers: Collection of papers to create relationships between.

    Returns:
        List of semantic relationships.
    """
    relationships: list[Related] = []
    n_papers = len(papers)

    # Track connections and targets for each paper
    connections: defaultdict[int, set[int]] = defaultdict(set)
    targets = [rng.randint(8, 17) for _ in range(n_papers)]

    # Create a list of paper indices that still need connections
    needs_connections = list(range(n_papers))

    while needs_connections:
        # Pick a random paper that needs connections
        idx = rng.randint(0, len(needs_connections) - 1)
        paper_idx = needs_connections[idx]

        # Find how many more connections this paper needs
        current_connections = len(connections[paper_idx])
        needed = targets[paper_idx] - current_connections

        if needed <= 0:
            # Remove from list if no more connections needed
            needs_connections.pop(idx)
            continue

        # Find eligible partners (papers that need connections and aren't already connected)
        eligible = [
            i
            for i in needs_connections
            if i != paper_idx
            and i not in connections[paper_idx]
            and len(connections[i]) < targets[i]
        ]

        if not eligible:
            # No eligible partners, remove from list
            needs_connections.pop(idx)
            continue

        # Select random partners (up to the number needed)
        num_to_select = min(needed, len(eligible))
        selected = rng.sample(eligible, num_to_select)

        # Create relationships
        for partner_idx in selected:
            # Update connection tracking
            connections[paper_idx].add(partner_idx)
            connections[partner_idx].add(paper_idx)

            # Generate similarity score
            alpha, beta = 2.5, 2.5
            raw_score = rng.betavariate(alpha, beta)
            score = 0.1 + raw_score * 0.85

            # Create relationship with consistent ordering
            paper1_id, paper2_id = papers[paper_idx].id, papers[partner_idx].id
            source, target = sorted([paper1_id, paper2_id])

            relationships.append(
                Related(
                    source=source,
                    target=target,
                    type_=RelatedType.SEMANTIC,
                    similarity=score,
                )
            )

        # Check if this paper is done
        if len(connections[paper_idx]) >= targets[paper_idx]:
            needs_connections.pop(idx)

    return relationships


def _generate_static_database(rng: random.Random, paper_count: int) -> StaticDatabase:
    """Generate complete synthetic database with papers and relationships.

    Args:
        rng: Random number generator.
        paper_count: Total number of papers to generate.

    Returns:
        StaticDatabase containing all generated data.
    """
    print("Generating core papers...")
    core_papers = CORE_PAPERS

    print("Generating additional papers...")
    num_additional = paper_count - len(core_papers)
    additional_papers = _generate_additional_papers(rng, num_additional)

    all_papers = [*core_papers, *additional_papers]

    print("Generating citation relationships...")
    citations = _generate_citation_relationships(rng, all_papers)

    print("Generating semantic relationships...")
    semantics = _generate_semantic_relationships(rng, all_papers)

    all_relationships = [*citations, *semantics]
    print(f"Total relationships: {len(all_relationships)}")

    return StaticDatabase(papers=all_papers, related=all_relationships)


def _analyse_database(database: StaticDatabase, output_file: Path) -> None:
    """Analyse and print statistics about the generated database.

    Provides comprehensive statistics about the generated papers,
    relationships, and their distributions.

    Args:
        database: Generated database to analyse.
        output_file: Path where the data was saved.
    """
    print("\nBasic Statistics:")
    print(f"- Output file: {output_file}")
    print(f"- Papers: {len(database.papers)}")
    print(f"- Total relationships: {len(database.related)}")

    # Separate by type
    citations = [r for r in database.related if r.type_ is RelatedType.CITATION]
    semantic = [r for r in database.related if r.type_ is RelatedType.SEMANTIC]

    print(f"- Citation relationships: {len(citations)}")
    print(f"- Semantic relationships: {len(semantic)}")

    # Analyse citation patterns
    papers_citing: defaultdict[PaperId, int] = defaultdict(int)
    papers_cited_by: defaultdict[PaperId, int] = defaultdict(int)
    for rel in citations:
        papers_citing[rel.target] += 1  # target paper is doing the citing
        papers_cited_by[rel.source] += 1  # source paper is being cited

    # Outgoing citations
    citing_counts = list(papers_citing.values())
    if citing_counts:
        print("\nPapers citing others (outgoing citations):")
        print(f"  - Papers that cite: {len(citing_counts)}")
        print(f"  - Avg citations per paper: {statistics.mean(citing_counts):.2f}")
        print(f"  - Min/Max: {min(citing_counts)}/{max(citing_counts)}")
        print(
            f"  - Stdev: {statistics.stdev(citing_counts):.2f}"
            if len(citing_counts) > 1
            else "  - Stdev: N/A"
        )

    # Incoming citations
    cited_counts = list(papers_cited_by.values())
    if cited_counts:
        print("\nPapers being cited (incoming citations):")
        print(f"  - Papers cited: {len(cited_counts)}")
        print(f"  - Avg times cited: {statistics.mean(cited_counts):.2f}")
        print(f"  - Min/Max: {min(cited_counts)}/{max(cited_counts)}")
        print(f"  - Top 5 most cited: {sorted(cited_counts, reverse=True)[:5]}")

    # Semantic relationships
    semantic_per_paper: defaultdict[PaperId, int] = defaultdict(int)
    for rel in semantic:
        semantic_per_paper[rel.source] += 1
        semantic_per_paper[rel.target] += 1

    semantic_counts = list(semantic_per_paper.values())
    if semantic_counts:
        print("\nSemantic relationships per paper:")
        print(f"  - Avg per paper: {statistics.mean(semantic_counts):.2f}")
        print(f"  - Min/Max: {min(semantic_counts)}/{max(semantic_counts)}")
        print(
            f"  - Stdev: {statistics.stdev(semantic_counts):.2f}"
            if len(semantic_counts) > 1
            else "  - Stdev: N/A"
        )

    # Similarity scores
    if citations:
        citation_sims = [r.similarity for r in citations]
        print("\nCitation similarity scores:")
        print(f"  - Avg: {statistics.mean(citation_sims):.3f}")
        print(f"  - Min/Max: {min(citation_sims):.3f}/{max(citation_sims):.3f}")

    if semantic:
        semantic_sims = [r.similarity for r in semantic]
        print("\nSemantic similarity scores:")
        print(f"  - Avg: {statistics.mean(semantic_sims):.3f}")
        print(f"  - Min/Max: {min(semantic_sims):.3f}/{max(semantic_sims):.3f}")


CORE_PAPERS: Sequence[Paper] = [
    Paper(
        id=PaperId("unseen-arxiv-1706.03762"),
        title="Attention Is All You Need",
        year=2017,
        authors=[
            "Ashish Vaswani",
            "Noam Shazeer",
            "Niki Parmar",
            "Jakob Uszkoreit",
            "Llion Jones",
            "Aidan N. Gomez",
            "Lukasz Kaiser",
            "Illia Polosukhin",
        ],
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        venue="NeurIPS 2017",
        citation_count=45237,
        doi="10.48550/unseen-arxiv-1706.03762",
        pdf_url="https://unseen-arxiv.org/pdf/1706.03762.pdf",
    ),
    Paper(
        id=PaperId("unseen-arxiv-1810.04805"),
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        year=2018,
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
        venue="NAACL 2019",
        citation_count=38942,
        doi="10.48550/unseen-arxiv-1810.04805",
        pdf_url="https://unseen-arxiv.org/pdf/1810.04805.pdf",
    ),
    Paper(
        id=PaperId("unseen-arxiv-2005.14165"),
        title="Language Models are Few-Shot Learners",
        year=2020,
        authors=[
            "Tom Brown",
            "Benjamin Mann",
            "Nick Ryder",
            "Melanie Subbiah",
            "Jared Kaplan",
            "Prafulla Dhariwal",
            "Arvind Neelakantan",
            "Pranav Shyam",
        ],
        abstract="Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples.",
        venue="NeurIPS 2020",
        citation_count=15743,
        doi="10.48550/unseen-arxiv-2005.14165",
        pdf_url="https://unseen-arxiv.org/pdf/2005.14165.pdf",
    ),
    Paper(
        id=PaperId("unseen-arxiv-1409.0473"),
        title="Neural Machine Translation by Jointly Learning to Align and Translate",
        year=2014,
        authors=["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"],
        abstract="Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance.",
        venue="ICLR 2015",
        citation_count=19384,
        doi="10.48550/unseen-arxiv-1409.0473",
        pdf_url="https://unseen-arxiv.org/pdf/1409.0473.pdf",
    ),
    Paper(
        id=PaperId("unseen-arxiv-1508.04025"),
        title="Effective Approaches to Attention-based Neural Machine Translation",
        year=2015,
        authors=["Minh-Thang Luong", "Hieu Pham", "Christopher D. Manning"],
        abstract="An attentional mechanism has lately been used to improve neural machine translation (NMT) by selectively focusing on parts of the source sentence during translation. However, there has been little work exploring useful architectures for attention-based NMT.",
        venue="EMNLP 2015",
        citation_count=8721,
        doi="10.48550/unseen-arxiv-1508.04025",
        pdf_url="https://unseen-arxiv.org/pdf/1508.04025.pdf",
    ),
]
METHODS = [
    "Analysis",
    "Optimization",
    "Framework",
    "Approach",
    "Algorithm",
    "Method",
    "Architecture",
    "System",
    "Model",
    "Technique",
]
ADJECTIVES = [
    "Novel",
    "Efficient",
    "Robust",
    "Scalable",
    "Advanced",
    "Improved",
    "Enhanced",
    "Unified",
    "Adaptive",
    "Hierarchical",
]
SUBTITLES = [
    ":A Comprehensive Study",
    ": Survey",
    " Is All You Need",
    ", A Deeper Look",
    ", A Prompt-based Approach",
    ": From Theory to Practice",
    ": Rethinking the Fundamentals",
    " in the Wild",
    ": Challenges and Opportunities",
    ": A Unified Perspective",
]
FIRST_NAMES = [
    "John",
    "Jane",
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "Jack",
    "Kate",
    "Liam",
    "Maria",
    "Noah",
    "Olivia",
    "Paul",
    "Quinn",
    "Rachel",
    "Sam",
    "Tara",
    "Uma",
    "Victor",
    "Wendy",
    "Xavier",
    "Yuki",
    "Zoe",
    "Alex",
    "Blake",
    "Casey",
    "Drew",
]
LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Sanchez",
    "Clark",
    "Ramirez",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
    "Allen",
    "King",
]
ABSTRACTS = {
    "Deep Learning": "This paper presents a comprehensive analysis of deep learning architectures and their applications in various domains. We propose novel training strategies that significantly improve convergence rates and model performance. Our experimental results demonstrate substantial improvements over existing state-of-the-art methods across multiple benchmark datasets.",
    "Neural Networks": "We investigate fundamental principles underlying neural network optimization and propose new architectural innovations. Our approach combines theoretical insights with practical implementations, showing improved performance in both supervised and unsupervised learning scenarios. The results provide valuable contributions to the understanding of neural network dynamics.",
    "Computer Vision": "This work addresses critical challenges in computer vision through the development of novel algorithms for image processing and analysis. We present comprehensive evaluations on standard datasets and demonstrate significant improvements in accuracy and computational efficiency compared to existing methods.",
    "Natural Language Processing": "We present advanced techniques for natural language understanding and generation, focusing on transformer-based architectures and attention mechanisms. Our approach shows substantial improvements in various NLP tasks including text classification, sentiment analysis, and language modeling.",
    "Machine Learning": "This paper explores fundamental machine learning principles and presents novel algorithms for supervised and unsupervised learning. We provide theoretical analysis and empirical validation, demonstrating improved performance across diverse application domains and datasets.",
    "Reinforcement Learning": "We introduce novel approaches to reinforcement learning that address key challenges in exploration and exploitation. Our methods demonstrate superior performance on complex sequential decision-making tasks and provide theoretical guarantees for convergence.",
    "Transfer Learning": "This research investigates effective strategies for knowledge transfer across different domains and tasks. We propose innovative techniques that leverage pre-trained models to achieve better performance with limited data in target domains.",
}
TOPICS = [
    "Deep Learning",
    "Neural Networks",
    "Computer Vision",
    "Natural Language Processing",
    "Machine Learning",
    "Artificial Intelligence",
    "Reinforcement Learning",
    "Transfer Learning",
    "Convolutional Networks",
    "Recurrent Networks",
    "Graph Neural Networks",
    "Optimization",
]
VENUES = [
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "IJCAI",
    "ACL",
    "EMNLP",
    "CVPR",
    "ICCV",
    "ECCV",
    "ICRA",
    "IROS",
    "UAI",
    "AISTATS",
]


if __name__ == "__main__":
    app()
