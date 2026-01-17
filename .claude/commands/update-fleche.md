# Update Fleche

Update the fleche tool and review changes to update project documentation.

## Steps

1. **Install/update fleche**:
   ```bash
   cargo install --git https://github.com/oyarsa/fleche --locked
   ```

2. **Review the guide** for current features:
   ```bash
   fleche guide
   ```

3. **Fetch the changelog** to identify new features and changes:
   - URL: https://raw.githubusercontent.com/oyarsa/fleche/refs/heads/master/CHANGELOG.md
   - Use WebFetch to retrieve and summarise changes

4. **Update CLAUDE.md** with any new or changed fleche commands:
   - Add new commands/flags discovered in the guide or changelog
   - Remove deprecated commands
   - Update syntax if changed
   - Keep the format consistent with existing documentation
