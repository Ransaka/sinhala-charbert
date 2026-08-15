# Code Review Delegation Strategy

When performing code reviews that span multiple files or modules:

1. **Decompose into subagent tasks.** Spin up subagents to read, understand, and summarize individual files or small file groups. Use cost-efficient models for this bulk exploration work.
2. **Synthesize with a higher-intelligence model.** After subagent summaries are collected, perform the final review pass yourself (the orchestrating agent) using a higher-intelligence model to:
   - Identify cross-file architectural issues
   - Spot inconsistencies between modules
   - Produce the consolidated review with actionable findings
3. **Subagent task scoping.** Each subagent should receive:
   - The file path(s) to review
   - The project context (what the module is supposed to do)
   - A clear instruction to return a structured summary (purpose, issues found, style concerns, suggestions)
