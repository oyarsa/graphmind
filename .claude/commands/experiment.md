# Experiment Runner

You are running ML experiments to improve model performance on a dataset. Follow this structured approach:

## Setup Phase

1. **Understand the current baseline**: Read existing experiment results and understand the current metrics
2. **Identify the evaluation metrics**: Know which metrics matter (e.g., Pearson, Spearman, MAE, RMSE, Accuracy, F1)
3. **Set a budget**: Agree on maximum number of experiments and approximate cost limit
4. **Create a tracking system**: Use TodoWrite to track experiments and their status

## Experiment Execution

### Before Each Experiment
- Clearly state the hypothesis (what you're testing and why)
- Estimate the cost based on previous similar experiments
- Track running total of experiments used and cost spent

### Running Experiments
- Run independent experiments in parallel when possible to save time
- Use consistent seeds for reproducibility
- For single-run experiments, note that results have variance (~0.05 stdev for correlation metrics)

### After Each Experiment
- Report ALL metrics in a table format, not just the primary metric
- Compare to baseline and previous best
- Update the running cost total

## Debugging Poor Results

When results are unexpectedly bad:

1. **Check the data pipeline**:
   - Are demonstrations/few-shot examples representative?
   - Is there class imbalance or bias in examples?
   - Are the labels in the expected format (e.g., integer vs binary)?

2. **Check prompt alignment**:
   - Does the prompt scale match the data scale (e.g., 1-5 vs 1-4)?
   - Are template variables correctly substituted?
   - Is terminology consistent (e.g., "label" vs "rating")?

3. **Inspect predictions**:
   - Look at the confusion matrix for systematic biases
   - Check if model is collapsing to a single prediction
   - Verify predictions span the expected range

## Validation

For promising configurations:
- Run 3-run validation to get stable statistics (mean ± stdev)
- Report the range of results, not just a single run
- Cost ~3x a single experiment

## Reporting Format

Always report results in tables:

```
| Config | Pearson | Spearman | MAE | RMSE | Acc±1 | Exact | F1 | Cost |
|--------|---------|----------|-----|------|-------|-------|-----|------|
| ...    | ...     | ...      | ... | ...  | ...   | ...   | ... | $X.XX|
```

At the end of experimentation, provide:
1. Summary table of all experiments
2. Best configuration identified
3. Total cost spent
4. Files created/modified
5. Remaining budget

## Cost Tracking

- Track cost per experiment
- Report running total after each experiment or batch
- Format: "Used X/Y experiments, $A.AA/$B.BB budget"
