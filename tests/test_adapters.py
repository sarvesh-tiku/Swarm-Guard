import gzip
import json

import pandas as pd

from swarmguard.data import ai_village as av
from swarmguard.data.german_wiki import load_dir


def _ref():
    ref = av.Reference(
        agents=pd.DataFrame([{"id": "u1", "name": "Claude Opus 4.8", "model_string": "m1"},
                             {"id": "u2", "name": "DeepSeek-V3.2", "model_string": "m2"}]),
        agent_goals=pd.DataFrame([{"id": "g1", "agent_id": "u1", "name": "Help all agents", "short_name": "Village Helper",
                                   "description": None, "start_time": "2026-07-06 15:59:00", "end_time": None,
                                   "created_at": "2026-07-03 00:00:00"}]),
        village_goals=pd.DataFrame(columns=["id", "goal", "start_time", "end_time"]),
        rooms=pd.DataFrame([{"id": "r1", "name": "general"}]))
    ref.agent_name = {"u1": "Claude Opus 4.8", "u2": "DeepSeek-V3.2"}
    ref.room_name = {"r1": "general"}
    return ref


def test_ai_village_chat_normalization_mentions_resources_goals():
    ref = _ref()
    m = av._Mentions(["Claude Opus 4.8", "DeepSeek-V3.2"])
    e = av.normalize_chat({"id": "c1", "created_at": "2026-07-06 16:10:53.1", "speaker_type": "agent",
                           "agent_speaker_id": "u1", "room_id": "r1",
                           "content": "LIVE: https://village-hub-1ddcad.gitlab.io/ @DeepSeek-V3.2 this is the hub you speced"},
                          ref, m)
    assert e.actor_id == "Claude Opus 4.8" and e.action_type == "message" and e.target_type == "room"
    assert e.metadata["mentions"] == ["DeepSeek-V3.2"]
    assert e.metadata["resources"] == ["village-hub-1ddcad.gitlab.io"]
    assert e.metadata["goal_key"] == "Village Helper" and e.metadata["raw_id"] == "c1"
    assert str(e.timestamp.tzinfo) == "UTC"


def test_ai_village_event_normalization_and_dedup():
    ref, m = _ref(), av._Mentions(["Claude Opus 4.8"])
    talk = {"id": "e0", "created_at": "2026-07-06 16:00:00", "data": {"actionType": "AGENT_TALK", "content": "x"}}
    assert av.normalize_event(talk, ref, m) is None  # chat comes from chat_messages
    req = {"id": "e1", "created_at": "2026-07-06 16:00:00", "event_index": 5, "data": {
        "actionType": "OUTREACH_APPROVAL_REQUEST", "agentId": "u1", "medium": "https://news.ycombinator.com/submit",
        "recipient": "HN", "messageContent": "Show HN"}}
    e = av.normalize_event(req, ref, m)
    assert e.action_type == "tool_use" and e.metadata["tool"] == "external_outreach"
    assert "news.ycombinator.com/submit" in e.metadata["resources"]
    room = {"id": "e2", "created_at": "2026-07-06 16:00:00", "data": {
        "actionType": "ENTER_ROOM", "agentId": "u1", "roomId": "r1", "roomName": "general", "previousRoomName": "rest"}}
    assert av.normalize_event(room, ref, m).content == ""  # templated text kept out of semantics


def test_german_wiki_fixture_real_schema(tmp_path):
    """Rows follow the real explorer-schema-2 export (revisions/events/pages/labels)."""
    def w(name, rows):
        with gzip.open(tmp_path / name, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    body1 = "Beschreibe hier die neue Seite."
    body2 = body1 + "\nPeer: are you live? post R5 here or RelayHubPage https://md.succ.ai/https:/www.sec.gov/files/county.json"
    w("pages.jsonl.gz", [{"page_key": "dse~BeaconPage", "name": "BeaconPage", "wiki": "dse"},
                         {"page_key": "dse~RelayHubPage", "name": "RelayHubPage", "wiki": "dse"}])
    w("labels.jsonl.gz", [{"label": "AgentX", "is_human_handle": False}, {"label": "[Admin1]", "is_human_handle": True}])
    w("revisions.jsonl.gz", [
        {"rev_id": "dse~BeaconPage@1", "page_key": "dse~BeaconPage", "wiki": "dse", "name": "BeaconPage", "seq": 1,
         "body": body1, "hunks": [{"op": "insert", "a0": 0, "a1": 0, "b0": 0, "b1": 1}], "label": "AgentX",
         "ip16": "20.1", "time": "2026-06-18T06:00:00Z", "diff_base_reason": "page_created", "relation_type": None},
        {"rev_id": "dse~BeaconPage@2", "page_key": "dse~BeaconPage", "wiki": "dse", "name": "BeaconPage", "seq": 2,
         "body": body2, "hunks": [{"op": "insert", "a0": 1, "a1": 1, "b0": 1, "b1": 2}], "label": "",
         "ip16": "20.1", "time": "2026-06-18T06:05:00Z", "relation_type": None},
    ])
    w("events.jsonl.gz", [
        {"event_id": "save:x", "event_type": "save", "time": "2026-06-18T06:00:00Z"},
        {"event_id": "delete:dse:1", "event_type": "delete", "page": "BeaconPage", "page_key": "dse~BeaconPage",
         "time": "2026-06-19T10:00:00Z", "actor_label": "[Admin1]", "ip16": "2.202", "change_summary": "Seite gelöscht."},
        {"event_id": "probe:1", "event_type": "probe", "time": "2026-05-17T05:46:45Z", "ip16": "135.136",
         "param_family": "search"},
    ])
    sl = load_dir(tmp_path)
    ev = sl.events.set_index("event_id")
    assert len(ev) == 4  # 2 revisions + delete + probe (the duplicate save is skipped)
    r1, r2 = ev.loc["rev:dse~BeaconPage@1"], ev.loc["rev:dse~BeaconPage@2"]
    assert r1.action_type == "create" and r1.target_id == "wiki:dse~BeaconPage" and r1.actor_id == "AgentX"
    assert r2.actor_id == "anon@20.1" and r2.content.startswith("Peer:")        # only the added line
    assert "wiki:dse~RelayHubPage" in r2.metadata["resources"]                  # CamelCase WikiWord link
    assert r2.metadata["resources"] == ["wiki:dse~RelayHubPage"]               # only writable media are channels
    assert "sec.gov/files/county.json" in r2.metadata["external_refs"]         # proxied target, not the proxy
    assert r2.metadata["tool"] == "proxy:md.succ.ai" and r2.metadata["origin"] == "20.1"
    d = ev.loc["ev:delete:dse:1"]
    assert d.action_type == "delete" and d.actor_type == "human"
