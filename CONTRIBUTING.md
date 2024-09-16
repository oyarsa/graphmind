# Contributing

The following are some guidelines to follow when contributing to this project.

## Coding Style
- Always follow the standard [PEP 8 guidelines](https://peps.python.org/pep-0008/).
- Use `uv run ruff format` to format the code, which will follow PEP 8.
- Use `uv run ruff check` to lint the code.
- Use `uv run pyright` to check for type errors.
- Always use type hints for functions and classes. See existing code for examples. The
  [mypy documentation](https://mypy.readthedocs.io/en/stable/getting_started.html) and
  [cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) are good
  resources to get started with type hints. We use pyright and not mypy, but the same
  concepts apply.
- Code in pull requests must always be formatted, and must not give any errors or
  warnings with `ruff check` or `pyright`. If there are any, they must be justified
  in the pull request.
- A GitHub Action will enforce these rules on every PR. If the code doesn't pass the
  checks, the PR will not be merged, unless there is a very good reason to do so.

## Git Workflow
- Follow [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow).
- Create a new branch for each feature or bug fix. The branch name should describe
  the feature or bug fix.
- After a PR is merged, the branch should be deleted. New changes should be made in a
  new branch.

### Commiting
- Commits should aim to cover one idea and be self-contained. This means you shouldn't
  conflate multiple changes into one commit. For example, if you're fixing a bug,
  refactoring some code and updating the README, these should be three separate commits.
- Commit messages should have a subject and a body, both limited to 72 characters. The
  body is optional if it's a small change that doesn't need further explanation.
- The subject should be in the imperative mood. For example, "Fix bug" instead of "Fixed
  bug" or "Fixes bug". It should not end with a period.
- The subject should start with the context. For example `s2orc(extract): Update...`
  when changing the extraction script for the S2ORC dataset. It should at least mention
  the component being changed (`s2orc`, in this case).
- If there is a body, it should be separated from the subject by a blank line.
- The body should be descriptive and explain the changes made in the commit, especially
  the reasoning behind them.

### Pull Requests
- Commit your changes to the new branch and create a new Pull Request when you're done.
- Unless your change is trivial, the PR should have multiple commits.
- The Pull Request should have a description summarising the changes made. Think of it
  as a summary of the commit messages in the PR.
- Make sure the PR is up to date with the master branch before requesting a review. You
  can do this by rebasing your branch on the master branch. I.e. run `git rebase master`
  in your branch.
- Every PR will be reviwed by another contribution. If there are any comments, they
  should be addressed in the same branch.
- After it's approved, the PR will be merged into the master branch.
- Merging should be done by the reviewer, not the contributor.
- Merging is done by "squash and merge": all commits in the PR are squashed into one
  commit before merging.
