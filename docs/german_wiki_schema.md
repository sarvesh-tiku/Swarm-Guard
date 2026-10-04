# German Wiki / collusion.wiki export: schema as used

Files go in `datasets/german_wiki/` (git-ignored). The checksums in `full-wiki-logs/SHA256SUMS` verify. The export
manifest (`explorer-schema-2`, generated 2026-09-03, cut `revision.write_date >= 2026-05-01`) reports 122/122
internal checks passing.

| file | rows | used as |
|---|---:|---|
| `revisions.jsonl[.gz]` | 14,591 | `edit` / `create` events: one stored save each (body + difflib `hunks`) |
| `events.jsonl[.gz]` | 19,913 | `delete` (5,217, all `[Admin1]`), `revert` (4 recreations without a stored revision), `probe` (101 script-injection requests). `save` rows duplicate revisions and are skipped |
| `pages.jsonl[.gz]` | 4,579 | page index (known page names for link resolution; deletion/recreation totals) |
| `labels.jsonl[.gz]` | 3,103 | `is_human_handle` → actor type (`[Admin1]`, `[Admin2]`, `[Person22]`) |
| `shortener-logs.json.gz` | 499 links | `create` events on `rmn.re/<keyword>` (actor `anon@<ip16>`) |
| `other-wikis.json.gz` | 8 pages / 90 revisions | `edit` events (MediaWiki/UseMod; no labels) |
| `records.jsonl.gz`, `links.jsonl.gz` | 13,703 / 23,877 | not on the timeline (no timestamps); cross-site text corpus |
| `site-coverage.csv`, `coverage-gaps.csv` | 143 / 110 sites | report context (observability gaps) |
| `manifest.json[.gz]` | — | report context (cut, checks) |

## Key fields

- **revisions:** `rev_id` (`wiki~Page@seq`), `page_key` (`wiki~Page`), `wiki` (dse 13,403 · probier 1,013 · fractal 169 ·
  dorfwiki 6), `seq`, `body`, `hunks` (`op`, `b0`, `b1` index the new body's lines), `label`, `ip16`, `time` (UTC, `Z`),
  `time_grade` (reqlog/rclog/write_date), `diff_base_reason` (`page_created`), `relation_type`
  (`first_recreation_of` + `related_event_id` → a deletion), `change_summary`.
- **events:** `event_type`, `time`, `page_key`, `actor_label`, `ip16`, `request_action`, `param_family` (probes),
  `related_event_id`/`relation_type` (reverts), `source_refs`.

## Mapping decisions

- **Actor = `label`**, or `anon@<ip16>` when blank (899 saves). Labels are self-chosen per save and spoofable. The
  /16 travels as `metadata.origin`, and an edge whose evidence pairs share an origin gets the alternative "same
  origin on both sides: possibly one operator under two identities" and is downgraded.
- **Content = lines the revision added** (from `hunks`), minus the wiki's placeholder `Beschreibe hier die neue
  Seite.` Identical text from ≥3 actors is treated as a template and excluded from semantic similarity.
- **Channels (resources) = writable media only:** the page itself, pages it references (CamelCase WikiWords
  matched against known page names, `[[links]]`, `wiki.cgi?Page` URLs), and short links. Public data URLs the agents
  fetch (sec.gov, api.datausa.io, …) are shared *sources*, not channels; they are kept as
  `metadata.external_refs` evidence.
- **Fetch proxies** (`jqp.vercel.app`, `r.jina.ai`, `allorigins.hexlet.app`, `md.succ.ai`, `markdown.new`, `pure.md`,
  `cors.bwa.workers.dev`, …) become `metadata.tool = proxy:<host>`, and the proxied target URL is unwrapped
  (recursively; agents chain proxies).
- **Admin deletions** are human `delete` events. They drive the *historical audit* (`analysis/historical.py`):
  when each episode's pages were removed, how many writes landed first, and whether pages were recreated.
- Addresses are already cut to /16 in the export; nothing finer is stored or shown.
