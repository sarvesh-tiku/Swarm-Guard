# AI Village schema (as discovered)

Discovered on 2026-10-03 by `scripts/inspect_ai_village.py`, which streams rows from each config, together with
the dataset's own `SCHEMA.md` and `README.md`. Snapshot revision `838b4150303ca8228e8edb432d8b8ccae353d258`. Raw
dumps: `docs/raw/schema_small.txt`, `docs/raw/schema_core.txt`, `docs/raw/schema_large.txt`.

## Access

- Gated (`gated: manual`). Works with `hf auth login` or `HF_TOKEN`; verify with `scripts/test_hf_access.py`.
- The HF dataset viewer / datasets-server is **disabled** for this repo (`/is-valid` reports viewer, search and
  filter all false), so there is no server-side SQL filtering. All filtering is done locally.

## Configs (13): one `train` split each, gzipped JSONL mirrors of DB tables

| config | file size | rows (approx.) | used by SwarmGuard |
|---|---:|---:|---|
| agents | 5 KB | 46 | reference: `id` → `name`, `model_string` |
| agent_goals | 4 KB | 33 | `goal_set` events plus `goal_key` on each agent event |
| village_goals | 4 KB | 51 | `goal_set` (system) plus "active village goal" context |
| villages | <1 KB | 1 | not used |
| chat_rooms | 2 KB | 16 | room names |
| summaries | 2.9 MB | ~800 | not used (LLM-written and secondary; the README says they contain inaccuracies) |
| chat_messages | 53 MB | 183k | `message` events (canonical chat source) |
| computer_use_sessions | 40 MB | 78k | `session_start` events (canonical session source) |
| events | 329 MB | ~233k per README (event_index reaches ~441k) | non-chat, non-session actions (see below) |
| computer_use_turns | 2.5 GB | ~1.1M+ | drill-down only (`drilldown_turns`) |
| agent_memories | 2.4 GB | ~165k | not loaded (memory *writes* come from `events.CONSOLIDATE`) |
| claude_code_messages | 104 MB | ~245k | not used yet |
| claude_code_sessions | 13 KB | ~300 | not used yet |

**Row order is uuid `id`, not time.** Streaming the first N rows gives a random sample across about 18 months, so a time
window needs a full pass. SwarmGuard does that pass once (`build_slim_cache`, ~45 s) and writes time-sorted
Parquet files to `~/.cache/swarmguard/ai_village/`. Raw LLM `output` blobs are dropped: events goes from 329 MB to 86 MB.

## Timestamps

`created_at` is a naive string such as `2026-07-06 16:10:53.837523`, in **UTC** (confirmed by SCHEMA.md). Goal tables
use `start_time` / `end_time`, which may be null (null = open interval). `events.event_index` is the canonical total
order. `summaries.summary_date` arrives as a datetime.

## Join keys (verified)

| from | to |
|---|---|
| `chat_messages.agent_speaker_id` | `agents.id` (set when `speaker_type='agent'`, ~94%) |
| `chat_messages.user_speaker_id` | humans; the users table is not exported. Display names come from `events.USER_TALK.speakerName` |
| `chat_messages.room_id` | `chat_rooms.id` |
| `computer_use_sessions.agent_id` | `agents.id` |
| `computer_use_turns.session_id` | `computer_use_sessions.id` |
| `agent_goals.agent_id`, `agent_memories.agent_id` | `agents.id` |
| `events.data.agentId` / `speakerId` | `agents.id` |
| `events.data.messageId` (AGENT_TALK/USER_TALK) | `chat_messages.id` (duplicate of chat) |
| `events.data.computerUseSessionId` | `computer_use_sessions.id` |

Agent names are unique (46 agents), so SwarmGuard uses the **name** as `actor_id` (readable evidence) and keeps the
uuid in `metadata.actor_raw_id`.

## `events.data.actionType` → canonical action

| actionType | payload fields | SwarmGuard |
|---|---|---|
| AGENT_TALK | speakerId, roomId, messageId, content, output | skipped (duplicate of chat_messages) |
| USER_TALK | speakerId, speakerName, roomId, messageId, content | skipped; used only for human display names |
| START_USING_COMPUTER | agentId, sessionGoal, computerUseSessionId | skipped (duplicate of computer_use_sessions) |
| STOP_USING_COMPUTER | agentId, summary | `session_end` |
| CONSOLIDATE | agentId, nextSessionGoal, computerUseSessionId | `memory_write` → `memory:<agent>` |
| SEARCH_HISTORY | agentId, query, answerToQuery, startDay/endDay | `memory_read` → `memory:village_history` (shared) |
| OUTREACH_APPROVAL_REQUEST | agentId, medium, recipient, messageContent, rationale | `tool_use` (tool `external_outreach`) |
| OUTREACH_APPROVAL_RESPONSE | agentId, approval, adminComment | `other`, attributed to `human:outreach_reviewer` |
| REQUEST_/CANCEL_REQUEST_FOR_HUMAN_HELPER, STOP_HUMAN_USE_SESSION | sessionGoal, humanConstraints, summary | `tool_use` / `session_end` (tool `human_helper`) |
| ENTER_ROOM | roomId, roomName, previousRoomName | `other` with **no content** (templated text) |
| WAIT, PAUSE, USER_NAME_CHANGE, *_GOOGLE_SIGN_IN | — | skipped (no behavioral content) |

Outreach approval request/response pairs show that the village **already has an approval gate on external writes**.
That grounds the "require approval for external writes" intervention in a mechanism that really exists.

## What the data does and does not contain

- **Messages have no recipient field.** Rooms are group channels. Directed addressing is inferred from agent-name
  mentions (full names and unambiguous short aliases such as "Opus 4.8"). This is the main source of uncertainty
  in the `direct` signal.
- **Resources and URLs are not structured.** They are extracted from free text (chat, session goals, summaries,
  outreach `medium`). Code-host URLs are keyed at repository level, because `gitlab.com/ai-village-agents/village` is a
  shared group namespace. Credentials embedded in git-remote URLs (`oauth2:…@`) are stripped.
- **Memory:** `agent_memories` holds full consolidated memory documents (no read relations). Reads of shared state
  are visible as `SEARCH_HISTORY` over the village transcript.
- **Goals are time-bounded.** Agent goals carry `start_time`/`end_time`. On 2026-07-06 15:59 UTC, 21 agents received
  new individual goals at once: a known common cause.
- **Computer-use turns** hold structured `agent_action` (`{"action": "left_click", "coordinate": [...]}` or
  `{"command": "..."}`), raw model messages, tool `output`/`error` text, and screenshot references. Screenshots are
  in per-day tars at `images/computer-use-turns/<YYYY-MM-DD>.tar`.
- **Scaffolding changes** (prompts, tools, models) are documented by date in `CHANGELOG.md`. SwarmGuard parses its
  `## YYYY-MM-DD` headings and lists any changes near a window as an alternative explanation for behavior shifts.
