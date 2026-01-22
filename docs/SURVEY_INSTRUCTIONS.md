# Human Evaluation Survey Instructions

## Overview

We are evaluating a system that automatically assesses the novelty of research
papers. The system retrieves related work, analyses how a paper's contributions
compare to prior work, and produces a structured novelty assessment with
supporting evidence.

Your task is to evaluate the quality of the system's outputs by examining
papers you are familiar with.

## How to Participate

### Step 1: Access the Demo Tool

Visit: **https://oyarsa.github.io/graphmind/**

### Step 2: Search for a Paper

- Go to the **arXiv** section
- Use the search bar to find a paper **you are already familiar with** (ideally
  one you have read or worked on)
- Familiarity with the paper is essential—you need domain knowledge to judge
  the system's outputs

### Step 3: Review the System Output

The system displays several components for each paper:

1. **Paper Summary**: A brief description of the paper's main contributions
2. **Novelty Rating**: A score from 1 (not novel) to 5 (highly novel)
3. **Supporting Evidence**: Evidence from related work that supports the
   novelty assessment
4. **Contradictory Evidence**: Evidence that contradicts or challenges the
   novelty claims
5. **Key Comparisons**: Technical comparisons between the paper and related
   work
6. **Hierarchical Graph**: A visual representation of the paper's intellectual
structure, showing:
   - The paper's title at the top
   - Key claims and contributions
   - Methods used to support those claims
   - Experiments that validate the methods
   - Directed edges showing how these components relate to each other

Take time to read through all sections carefully, including clicking on graph
nodes to see extracted excerpts.

### Step 4: Complete the Survey

For each paper you evaluate, you will provide:
- The paper title
- Your familiarity with the paper (1–5)
- Ratings for five quality metrics (1–5 each)
- Optional free-text comments

---

## Evaluation Metrics

For each metric, rate from **1 (lowest)** to **5 (highest)** using the
definitions below.

### 1. Specificity

*Does the rationale contain information specific to this paper, or is it too
generic?*

| Score | Description                                                                                    |
|-------|------------------------------------------------------------------------------------------------|
| 1     | Completely generic; the rationale could apply to almost any paper in the field                 |
| 2     | Mostly generic with occasional specific details                                                |
| 3     | Mix of generic statements and paper-specific information                                       |
| 4     | Mostly specific to this paper's contributions with minor generic elements                      |
| 5     | Highly specific; the rationale clearly addresses this paper's unique contributions and methods |

### 2. Relevance

*Are the retrieved related papers relevant to the paper being evaluated?*

| Score | Description                                                              |
|-------|--------------------------------------------------------------------------|
| 1     | Retrieved papers are unrelated or from entirely different research areas |
| 2     | Most retrieved papers are tangentially related at best                   |
| 3     | Some retrieved papers are relevant, but several are not                  |
| 4     | Most retrieved papers are relevant, with only minor misses               |
| 5     | All or nearly all retrieved papers are highly relevant prior work        |

### 3. Factuality

*Does the rationale accurately reflect what the retrieved papers actually say?*

| Score | Description                                                                        |
|-------|------------------------------------------------------------------------------------|
| 1     | The rationale contains major factual errors or misrepresents the related work      |
| 2     | Several factual inaccuracies or misinterpretations                                 |
| 3     | Some accurate information but also some errors or unsupported claims               |
| 4     | Mostly accurate with only minor errors or imprecisions                             |
| 5     | Fully accurate; the rationale correctly represents what the related papers contain |

### 4. Rating Correctness

*Based on the evidence presented, is the novelty rating (1–5) appropriate?*

| Score | Description                                                                        |
|-------|------------------------------------------------------------------------------------|
| 1     | The rating is completely unjustified by the evidence; should be very different     |
| 2     | The rating is poorly supported; off by more than one point                         |
| 3     | The rating is somewhat justified but could reasonably be one point higher or lower |
| 4     | The rating is well-supported by the evidence with minor room for disagreement      |
| 5     | The rating is fully justified; the evidence clearly supports this exact rating     |

### 5. Graph Usefulness

*Is the hierarchical graph accurate and useful for understanding the paper's structure?*

| Score | Description                                                                  |
|-------|------------------------------------------------------------------------------|
| 1     | The graph is incorrect, confusing, or provides no useful information         |
| 2     | The graph has significant errors or omissions; limited usefulness            |
| 3     | The graph captures some key elements but misses important aspects            |
| 4     | The graph is mostly accurate and helpful, with minor issues                  |
| 5     | The graph accurately represents the paper's structure and aids understanding |

---

## Guidelines

- **Evaluate at least 1 paper**, preferably 5 or more if time permits
- **Choose papers you know well**—your expertise is essential for accurate
  evaluation
- **Be honest**—we need genuine assessments, not uniformly positive or negative
  ratings
- **Use the full scale**—don't hesitate to give 1s or 5s when warranted
- **Add comments** when a rating needs explanation, especially for low scores

---

## Survey Fields

For each paper you evaluate:

1. **Paper Title** (text field)
2. **Your Familiarity with This Paper** (1–5)
   - 1 = Heard of it but never read it
   - 3 = Read it once / general familiarity
   - 5 = Know it very well (e.g., your own work, frequently cited)
3. **Specificity** (1–5)
4. **Relevance** (1–5)
5. **Factuality** (1–5)
6. **Rating Correctness** (1–5)
7. **Graph Usefulness** (1–5)
8. **Comments** (optional free text)

---

## Questions?

If you encounter any issues with the demo tool or have questions about the
evaluation criteria, please contact italo.da\_silva@kcl.ac.uk.

Thank you for your participation!
