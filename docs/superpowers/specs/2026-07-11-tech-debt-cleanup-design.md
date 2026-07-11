# Tech Debt Cleanup Design

## Scope

Resolve all 12 known technical debt items from CLAUDE.md "已知技术债" + audit findings.

## Approach Summary

### A1 — position_agent SYSTEM_PROMPT → YAML
- `get_agent()` in `position_agent_service.py` loads `position_agent_system` YAML via `app.llm.prompts.load_prompt()`
- Delete Python `SYSTEM_PROMPT` constant

### A2 — Graph nodes DB via state.custom → config.configurable
- Move db session from `state["custom"]["db"]` to `config["configurable"]["db"]`
- Nodes accept `config: RunnableConfig` parameter
- `submit_answer()` in `service.py` passes db in config, not state.custom

### A3 — SSE formatting → sse-starlette
- Install `sse-starlette`
- Replace `_sse()` with `ServerSentEvent` encoding
- API endpoint uses `EventSourceResponse` instead of `StreamingResponse` (or keep StreamingResponse with SSE-formatted strings from ServerSentEvent)

### B4 — Email merge
- One implementation survives (email_smtp.py — raw smtplib, no FastMail dependency)
- Merge convenience methods from email.py into email_smtp.py
- Delete email.py, update all imports

### B5 — Paginator → fastapi-pagination
- Install `fastapi-pagination`
- Replace custom Paginator usage with library's paginate function
- Delete `schemas/paginator.py`

### B6 — Custom retry → tenacity
- Install `tenacity`
- Replace manual retry loop in `chat_completion()` with `@retry` decorator

### B7 — Log distribution → stdlib
- Replace RedisLogHandler + LogQueueProcessor with stdlib `QueueHandler` + `QueueListener`
- Remove Redis dependency from logging

### C8 — Delete deprecated files
- Delete `services/common/embedding.py` (update importers)
- Delete `schedule/celery_job.py` (update importers)

### C9 — Golden set → manual follow-up
- Cannot auto-populate; document as follow-up task

### C10 — CORS + IP → deploy follow-up
- Document as deployment-time concerns

### C11 — Merge submit_answer endpoints
- Single `POST /{id}/answer?stream=true|false` endpoint
- Update frontend calls

### C12 — Welcome email task
- Implement using the surviving email module

## Risk Assessment

- A2 (state→config): LangGraph nodes signature change, tests may need updates
- A3 (SSE): frontend SSE parsing must remain compatible
- B4 (email): need to verify which callers use which email module
- B5 (paginator): changing pagination output format may break frontend
- B7 (log): careful to preserve file logging behavior
- C11 (merge endpoints): frontend must update corresponding API calls
