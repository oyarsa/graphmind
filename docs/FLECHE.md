# Fleche (Remote Job Submission)

`fleche` is a utility for running jobs on remote Slurm clusters via SSH.
Configuration is in `fleche.toml`. Run `fleche guide` for full built-in documentation.

## Key Concepts

- **Check `fleche.toml` first** for available jobs
- Most commands default to most recent job if no job-id given
- Short ID suffix works (e.g., `x7k2` instead of full `train-20260115-153042-847-x7k2`)
- Config supports `${VAR}` substitution from env vars, `.env` file, and `${PROJECT}` built-in
- **`--filter` vs `--tag` vs `--name`**: `--filter` is for job STATUS, `--tag` is for your custom tags, `--name` is regex on job ID

## Available Jobs

**Remote Slurm jobs:** `train`, `infer`, `train_gen`, `infer_gen`, `train_gen_graph`, `infer_gen_graph`

**Local GPT jobs:** `gpt_orc`, `gpt_peerread`, `gpt_orc_test`, `gpt_peerread_test`

| Job | Description | Command |
|-----|-------------|---------|
| `cuda_check` | Test CUDA/GPU availability | `fleche run cuda_check` |
| `train` | Train Llama SFT model | `fleche run train --env DATASET=orc` |
| `infer` | Run inference | `fleche run infer --env DATASET=orc` |

## Running Jobs

```bash
fleche run <job>                              # Submit and stream output (Ctrl+C disconnects, job keeps running)
fleche run <job> --bg                         # Run in background (--notify for alerts)
fleche run <job> --env VAR=value --tag key=value  # Set env vars and tags
fleche run <job> --note "description"         # Add note to document experiment
fleche run <job> --command "nvidia-smi"       # Override command (keeps job's Slurm config)
fleche run <job> --dry-run                    # Preview sbatch script without submitting
fleche run <job> --host local                 # Run locally instead of on remote Slurm cluster
fleche run <job> --after <job-id>             # Run after another job completes (dependency)
fleche run <job> --retry 3                    # Auto-retry on failure with exponential backoff
fleche run "command" --gpus 1 --time 1:00:00  # Adhoc Slurm command (no job definition)
fleche rerun <job-id>                         # Re-run previous job with same settings
fleche exec <cmd>                             # Run directly via SSH, no Slurm (quick tests)
fleche exec <cmd> --host local                # Run command locally without SSH
```

## Monitoring

```bash
fleche status -n 20                     # Show last 20 jobs
  --filter running                      #   Filter by status (running/pending/completed/failed/cancelled)
  --tag key=value                       #   Filter by tag
  --name 'pattern'                      #   Filter by job ID regex (substring match, use ^/$ to anchor)
  --archived                            #   Show only archived jobs
  --all-jobs                            #   Show all jobs including archived
fleche logs [job-id]                    # View logs (--raw to strip ANSI, --follow to stream)
  -n 50                                 #   Show only last N lines
  --stdout / --stderr                   #   Show only one stream
  --note 'pattern'                      #   Filter by note content (case-insensitive regex)
fleche wait [job-id]                    # Wait for completion (--notify for alerts)
fleche stats [job-id]                   # Show resource usage (elapsed time, CPU time, max memory)
fleche note <job-id> [text]             # View or set job note
fleche ping                             # Check Slurm cluster health
fleche check                            # Validate config after editing
fleche check --remote                   # Validate config against remote server (SSH, Slurm, disk space)
fleche doctor                           # Comprehensive troubleshooting diagnostics
fleche compare <a> <b>                  # Compare two job configurations side-by-side
fleche tags                             # List unique tags across all jobs
```

## Results

```bash
fleche download [job-id]                # Download output files (--partial while job running)
  --filter "*.json"                     #   Download only specific file types (repeatable, recursive)
  --filter "!checkpoints/**"            #   Exclude files/directories with ! prefix
  --dry-run                             #   Preview what would be downloaded
```

Example: download outputs only, skip model weights:
```bash
fleche download --filter "*.json" --filter "*.json.zst" --filter "*.log" --filter "!checkpoint*/**"
```

## Cleanup

```bash
fleche cancel [job-id]                  # Cancel job (--all for all active, --tag to filter)
fleche clean --older-than 2h -y         # Clean old jobs periodically
fleche clean --workspace                # Also delete shared workspace (use with caution)
fleche clean --archive <job-id>         # Archive job (hide without deleting)
fleche clean --unarchive <job-id>       # Restore archived job
```

## Running Llama Experiments

**Train on ORC:**
```bash
fleche run train --env DATASET=orc --env CONFIG=llama_basic \
  --tag dataset=orc --tag config=llama_basic
```

**Train on PeerRead:**
```bash
fleche run train --env DATASET=peerread --env CONFIG=llama_peerread \
  --tag dataset=peerread --tag config=llama_peerread
```

**Run inference after training:**
```bash
fleche run infer --env DATASET=orc --env CONFIG=llama_basic \
  --tag dataset=orc --tag type=inference
```

Jobs share a workspace, so the trained model from `train` is automatically available
to `infer` without needing to download and re-upload.

## GPT Experiments (Local)

```bash
fleche run gpt_orc --env PROMPT=sans                          # Run ORC ablation with specific prompt
fleche run gpt_peerread --env PROMPT=full-graph-structured    # Run PeerRead ablation
fleche run gpt_orc_test --env PROMPT=sans                     # Quick test (1 paper, 1 run)
```

Prompts: `sans`, `related`, `norel-graph`, `semantic-only`, `full-graph-structured`

Options:
- `--env SOURCES=citations` or `--env SOURCES=semantic` for source filtering
- `--env RUNS=1` for single run, `--env LIMIT=10` to limit papers

## Tagging Jobs

```bash
# Add tags when submitting
fleche run train --env DATASET=orc --tag experiment=ablation --tag model=llama

# Filter by tag
fleche status --tag experiment=ablation
fleche status --tag dataset=orc --filter running
fleche logs --tag experiment=ablation
fleche download --tag config=llama_basic
fleche cancel --all --tag experiment=test
fleche clean --all --tag experiment=test
```

## File Sync

- Project code syncs automatically to shared workspace (respects `.gitignore`)
- Input data specified in `inputs` is copied to workspace
- Jobs share workspace, so outputs from one job are available to subsequent jobs
- Use `fleche download` to pull outputs to local machine

## Tips

- Use `--dry-run` to preview the sbatch script before submitting
- Ctrl+C during streaming disconnects but doesn't cancel the job
- Job IDs look like `train-20260114-153042-847-x7k2` (short suffix `x7k2` works too)
- Use `fleche exec` for quick tests without Slurm queue wait
- Clean old jobs periodically with `fleche clean --older-than 2h -y`
