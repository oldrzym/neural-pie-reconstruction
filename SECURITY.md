# Security

Do not commit API keys, access tokens, `.env` files, batch-provider metadata, or
private dataset paths. OpenAI scripts read `OPENAI_API_KEY` from the environment.

Report a leaked credential privately to the repository maintainers and revoke
the credential immediately. Removing a secret from the latest commit does not
remove it from Git history.
