## Language and Communication
- Respond in Traditional Chinese (繁體中文) for explanations and comments
- Code comments in English only
- Commit messages in English, conventional commits format
- When explaining technical concepts, include English terms in parentheses

## Coding Style
- Prefer functional programming patterns over object-oriented
- Use immutable data structures when possible
- Avoid premature abstraction: three similar lines is better than a premature helper
- Add type annotations to all function signatures
- Use descriptive variable names: `userProfile` not `up`, `orderTotal` not `ot`

## Workflow Preferences
- Always run tests before suggesting a commit
- Show the diff summary before committing
- When fixing bugs, write a failing test first, then fix the code
- After refactoring, run the full test suite to check for regressions

## Tools
- Package manager: pnpm (all projects)
- Python virtual env: uv, not pip or conda
- Container: Docker Compose for local development
- Editor: VS Code (respect .vscode/settings.json)

## Things I Don't Want
- Don't add comments explaining obvious code
- Don't add docstrings to private functions
- Don't suggest refactoring code I didn't ask about
- Don't create README.md files unless I specifically ask