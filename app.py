# app.py
import os
import time
import uuid
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

from zep_cloud.client import Zep
from zep_cloud.types import Message

load_dotenv()

ZEP_API_KEY = os.getenv("ZEP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Zep Chatbot + Entity Graph", layout="wide")

if not ZEP_API_KEY or not OPENAI_API_KEY:
    st.error("Missing ZEP_API_KEY or OPENAI_API_KEY in .env")
    st.stop()


@st.cache_resource
def get_zep() -> Zep:
    return Zep(api_key=ZEP_API_KEY)


@st.cache_resource
def get_llm() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


zep = get_zep()
llm = get_llm()

# ------------------------- Session -------------------------
def ensure_session():
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_001"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "turn" not in st.session_state:
        st.session_state.turn = 0
    if "node_cache" not in st.session_state:
        st.session_state.node_cache = {}  # uuid -> node dict


def ensure_user_and_thread(user_id: str, thread_id: str):
    try:
        zep.user.add(user_id=user_id)
    except Exception:
        pass
    try:
        zep.thread.create(thread_id=thread_id, user_id=user_id)
    except Exception:
        pass


def poll_task_if_present(task_id: Optional[str], max_wait_s: float = 8.0):  
    if not task_id:
        return
    start = time.time()
    while time.time() - start < max_wait_s:
        try:
            t = zep.task.get(task_id)
            status = getattr(t, "status", None) or getattr(t, "state", None)
            if status and str(status).lower() in {"completed", "succeeded", "success", "done"}:
                return
            if status and str(status).lower() in {"failed", "error"}:
                return
        except Exception:
            return
        time.sleep(0.25)


GENERIC_NODE_NAMES = {
    "assistant", "resources", "resource", "learning", "genai", "graduate", "chat", "thread", "message"
}

REL_MAP = {
    "HAS_PROFESSION": "has profession",
    "PROFESSION_IS": "profession is",
    "GRADUATED_FROM": "graduated from",
    "EXPECTED_GRADUATION_YEAR": "graduation year",
    "GRADUATED_IN": "graduated in",
    "HAS_NAME": "has name",
    "CURRENTLY_LEARNING": "currently learning",
    "CURRENTLY_ENGAGED_IN": "currently engaged in",
    "REQUESTED_BY": "requested by",
}

def normalize_edge_label(rel: str) -> str:
    r = (rel or "").strip()
    if not r:
        return ""
    if r in REL_MAP:
        return REL_MAP[r]
    r = re.sub(r"[^A-Za-z0-9_]+", "", r)
    r = r.replace("_", " ").strip().lower()
    r = re.sub(r"\s+", " ", r)
    if len(r) > 28:
        r = r[:28].rstrip() + "…"
    return r

def normalize_node_name(name: str, node_id: str, user_id: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    if node_id == user_id or n == user_id:
        return "You"
    if n.lower() in GENERIC_NODE_NAMES:
        return ""
    if len(n) <= 4 and n.islower():
        return n.upper()
    if " " not in n and n.islower():
        return n.title()
    return n

def color_for_labels(labels: List[str]) -> str:
    s = " ".join([str(l).lower() for l in labels]) if labels else ""
    if "person" in s or "user" in s:
        return "#79A7FF"
    if "org" in s or "organization" in s or "company" in s or "institution" in s:
        return "#7FE0B6"
    if "location" in s or "place" in s:
        return "#F49B6A"
    if "preference" in s or "interest" in s or "skill" in s:
        return "#B8A5FF"
    return "#F0C36D"


def get_node_cached(uuid_: str) -> Dict[str, Any]:
    if uuid_ in st.session_state.node_cache:
        return st.session_state.node_cache[uuid_]

    try:
        n = zep.graph.node.get(uuid_=uuid_)
        node = {
            "uuid": str(getattr(n, "uuid", uuid_)),
            "name": (getattr(n, "name", "") or "").strip(),
            "summary": (getattr(n, "summary", "") or "").strip(),
            "labels": getattr(n, "labels", None) or [],
            "attributes": getattr(n, "attributes", None) or {},
        }
    except Exception:
        node = {"uuid": uuid_, "name": "", "summary": "", "labels": [], "attributes": {}}

    st.session_state.node_cache[uuid_] = node
    return node


def is_entity_node(node: Dict[str, Any], user_id: str) -> bool:
    name = (node.get("name") or "").strip()
    labels = [str(x).lower() for x in (node.get("labels") or [])]

    bad_labels = {"message", "episode", "thread", "turn", "conversation", "chat", "system"}
    if any(b in labels for b in bad_labels):
        return False

    cleaned = normalize_node_name(name, node.get("uuid", ""), user_id)
    return bool(cleaned)


def zep_graph_slice_entities_relations(
    user_id: str,
    query: str,
    limit: int = 30,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    limit = min(int(limit), 80)

    edges_res = zep.graph.search(user_id=user_id, query=query, scope="edges", limit=limit)
    z_edges = getattr(edges_res, "edges", []) or []

    raw_edges: List[Tuple[str, str, str, str]] = []
    endpoint_ids: set[str] = set()

    for e in z_edges:
        src = getattr(e, "source_node_uuid", None) or getattr(e, "source", None)
        tgt = getattr(e, "target_node_uuid", None) or getattr(e, "target", None)
        if not src or not tgt:
            continue

        src, tgt = str(src), str(tgt)
        rel_raw = (getattr(e, "name", "") or "").strip()
        fact = (getattr(e, "fact", "") or "").strip()
        if not rel_raw:
            continue

        raw_edges.append((src, tgt, rel_raw, fact))
        endpoint_ids.add(src)
        endpoint_ids.add(tgt)

    node_meta: Dict[str, Dict[str, Any]] = {}
    node_label: Dict[str, str] = {}
    for nid in endpoint_ids:
        node = get_node_cached(nid)
        if not is_entity_node(node, user_id=user_id):
            continue
        cleaned_name = normalize_node_name(node.get("name", ""), nid, user_id)
        if not cleaned_name:
            continue
        node_meta[nid] = node
        node_label[nid] = cleaned_name

    edges_payload: List[Dict[str, Any]] = []
    for src, tgt, rel_raw, fact in raw_edges:
        if src not in node_meta or tgt not in node_meta:
            continue

        rel_disp = normalize_edge_label(rel_raw)
        if not rel_disp:
            continue

        # drop edges involving "Assistant" if it slips in
        if node_label.get(src, "").lower() == "assistant" or node_label.get(tgt, "").lower() == "assistant":
            continue

        edges_payload.append(
            {
                "from": src,
                "to": tgt,
                "label": rel_disp,
                "title": fact or rel_raw,
                "font": {
                    "align": "middle",
                    "background": "rgba(0,0,0,0.45)",
                    "color": "rgba(255,255,255,0.92)",
                    "size": 12,
                    "strokeWidth": 0,
                },
            }
        )

    nodes_payload: List[Dict[str, Any]] = []
    for nid, meta in node_meta.items():
        labels = meta.get("labels") or []
        nodes_payload.append(
            {
                "id": nid,
                "label": node_label.get(nid) or nid[:8],
                "title": meta.get("summary") or node_label.get(nid) or nid[:8],
                "shape": "dot",
                "size": 18,
                "color": {"background": color_for_labels(labels), "border": "#2b2b2b"},
            }
        )

    nodes_payload.sort(key=lambda n: (n.get("label") or "").lower())
    edges_payload.sort(key=lambda e: (e["from"], e["to"], (e.get("label") or "").lower()))

    if not nodes_payload:
        nodes_payload = [
            {
                "id": "empty",
                "label": "No clean entity nodes yet",
                "title": "Try a declarative fact: 'I graduated in 2025 from UB.'",
                "shape": "dot",
                "size": 18,
                "color": {"background": "#79A7FF", "border": "#2b2b2b"},
            }
        ]
        edges_payload = []

    return nodes_payload, edges_payload


def render_vis_network(nodes_payload: List[Dict[str, Any]], edges_payload: List[Dict[str, Any]], height_px: int = 650):
    nodes_json = json.dumps(nodes_payload)
    edges_json = json.dumps(edges_payload)

    html = f"""
    <div id="mynetwork" style="width: 100%; height: {height_px}px; border: 1px solid rgba(255,255,255,0.12); border-radius: 12px;"></div>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
      const nodes = new vis.DataSet({nodes_json});
      const edges = new vis.DataSet({edges_json});
      const container = document.getElementById("mynetwork");
      const data = {{ nodes: nodes, edges: edges }};

      const options = {{
        layout: {{ randomSeed: 7 }},
        interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
        nodes: {{ borderWidth: 1, font: {{ size: 16, color: "rgba(255,255,255,0.92)" }} }},
        edges: {{
          arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
          smooth: {{ type: "dynamic" }},
          color: {{ color: "rgba(200,200,200,0.4)" }}
        }},
        physics: {{
          enabled: true,
          stabilization: {{ iterations: 220 }},
          barnesHut: {{
            gravitationalConstant: -28000,
            springLength: 180,
            springConstant: 0.03,
            damping: 0.12,
            avoidOverlap: 0.25
          }}
        }}
      }};

      const network = new vis.Network(container, data, options);
      network.once("stabilizationIterationsDone", function () {{
        network.setOptions({{ physics: false }});
      }});
    </script>
    <!-- turn:{st.session_state.turn} -->
    """
    components.html(html, height=height_px + 40, scrolling=True)


def is_general_knowledge_query(q: str) -> bool:
    ql = (q or "").strip().lower()
    if any(tok in ql for tok in [" my ", " i ", " me ", " mine ", " when did i", "what is my", "where did i", "who am i"]):
        return False
    return ql.startswith(("what is", "explain", "define", "how does", "why", "give me", "tell me about"))

def expand_graph_query(user_query: str) -> str:
    q = (user_query or "").strip().lower()
    extra = []
    if "graduate" in q or "graduat" in q:
        extra += ["graduation year", "graduated in", "degree", "university", "college"]
    if "profession" in q or "job" in q or "work" in q:
        extra += ["profession", "role", "works as", "job title"]
    if "name" in q:
        extra += ["name", "full name"]
    return (user_query + " " + " ".join(extra)).strip()

def retrieve_relevant_graph_facts(user_id: str, query: str, limit: int = 10) -> str:
    limit = max(1, min(int(limit), 50))
    q = expand_graph_query(query)
    q = f"{q} {user_id}"

    edges_res = zep.graph.search(user_id=user_id, query=q, scope="edges", limit=limit)
    edges = getattr(edges_res, "edges", []) or []

    lines: List[str] = []
    for e in edges:
        rel_raw = (getattr(e, "name", "") or "").strip()
        if not rel_raw:
            continue

        src_id = getattr(e, "source_node_uuid", None) or getattr(e, "source", None)
        tgt_id = getattr(e, "target_node_uuid", None) or getattr(e, "target", None)
        if not src_id or not tgt_id:
            continue

        src_id, tgt_id = str(src_id), str(tgt_id)
        src = get_node_cached(src_id)
        tgt = get_node_cached(tgt_id)

        src_name = normalize_node_name(src.get("name", ""), src_id, user_id) or src_id[:8]
        tgt_name = normalize_node_name(tgt.get("name", ""), tgt_id, user_id) or tgt_id[:8]
        rel_disp = normalize_edge_label(rel_raw) or rel_raw
        fact = (getattr(e, "fact", "") or "").strip()

        if src_name.lower() == "assistant" or tgt_name.lower() == "assistant":
            continue

        if fact:
            lines.append(f"{src_name} --{rel_disp}--> {tgt_name}. Fact: {fact}")
        else:
            lines.append(f"{src_name} --{rel_disp}--> {tgt_name}")

    return "\n".join(lines)


def extract_memory_reinforcement_messages(user_text: str) -> List[str]:
    """
    Creates extra "hidden" declarative sentences to help Zep reliably extract edges.
    These are NOT shown in UI; only written to Zep thread.
    """
    t = (user_text or "").strip()
    out: List[str] = []

    m = re.search(r"\bgraduat(?:e|ed|ion)\b.*?\b(20\d{2})\b", t, flags=re.IGNORECASE)
    if m:
        year = m.group(1)
        out.append(f"I graduated in {year}.")

    m2 = re.search(r"\b(my\s+profession\s+is|i\s+am\s+an?|i\s+work\s+as)\s+([A-Za-z][A-Za-z0-9 \-/]{2,60})", t, flags=re.IGNORECASE)
    if m2:
        prof = m2.group(2).strip().rstrip(".")
        out.append(f"My profession is {prof}.")

    m3 = re.search(r"\bgraduat(?:e|ed|ion)\b.*?\bfrom\b\s+([A-Za-z][A-Za-z0-9 \-&]{2,80})", t, flags=re.IGNORECASE)
    if m3:
        school = m3.group(1).strip().rstrip(".")
        out.append(f"I graduated from {school}.")

    return out[:3]


def llm_answer(user_query: str, facts_block: str) -> str:
    """
    - If question is general knowledge: answer normally.
    - If personal: prefer facts; if missing, ask for the missing info or admit unknown.
    """
    if is_general_knowledge_query(user_query):
        system_prompt = "You are a helpful assistant. Answer clearly and concisely."
        user_prompt = user_query
    else:
        system_prompt = (
            "You are a helpful assistant.\n"
            "You may be given FACTS retrieved from the user's Zep Knowledge Graph.\n"
            "If the question is about the user's personal details, use ONLY the facts.\n"
            "If facts are missing, respond with either:\n"
            "  - a single short follow-up question needed to save the memory, OR\n"
            "  - 'I don't know based on your saved memory.'\n"
        )
        user_prompt = (
            "ZEP GRAPH FACTS:\n"
            f"{facts_block if facts_block else '(no relevant facts found)'}\n\n"
            "USER QUESTION:\n"
            f"{user_query}\n"
        )

    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


ensure_session()
ensure_user_and_thread(st.session_state.user_id, st.session_state.thread_id)

st.title("Zep Chatbot + Graph (Perfect Memory + Clean Graph)")

with st.sidebar:
    st.code(f"user_id:  {st.session_state.user_id}\nthread_id:{st.session_state.thread_id}")
    ingest_assistant = st.toggle("Ingest assistant into graph", value=False)  
    graph_limit = st.slider("Graph limit", 5, 80, 30, 5)
    top_k_facts = st.slider("Top graph facts", 2, 30, 10, 1)

    if st.button("Reset session"):
        for k in ["user_id", "thread_id", "chat", "turn", "node_cache"]:
            st.session_state.pop(k, None)
        st.rerun()

col_chat, col_graph = st.columns([1.1, 1.0], gap="large")

with col_chat:
    st.subheader("Chat")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    user_text = st.chat_input("Ask something...")

with col_graph:
    st.subheader("Graph (Zep)")
    graph_slot = st.empty()
    status_slot = st.empty()

nodes_payload, edges_payload = zep_graph_slice_entities_relations(
    user_id=st.session_state.user_id,
    query="profile",
    limit=graph_limit,
)
with graph_slot:
    render_vis_network(nodes_payload, edges_payload)

if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    with col_chat:
        with st.chat_message("user"):
            st.write(user_text)

    reinforcement = extract_memory_reinforcement_messages(user_text)
    zep_msgs = [Message(role="user", content=user_text)]
    for r in reinforcement:
        zep_msgs.append(Message(role="user", content=f"[MEMORY] {r}"))

    ignore_roles = [] if ingest_assistant else ["assistant"]
    add_res = zep.thread.add_messages(
        thread_id=st.session_state.thread_id,
        messages=zep_msgs,
        ignore_roles=ignore_roles,
    )
    task_id = getattr(add_res, "task_id", None)
    poll_task_if_present(task_id, max_wait_s=8.0)

    facts_block = retrieve_relevant_graph_facts(
        user_id=st.session_state.user_id,
        query=user_text,
        limit=top_k_facts,
    )

    assistant_text = llm_answer(user_text, facts_block)

    st.session_state.chat.append({"role": "assistant", "content": assistant_text})
    with col_chat:
        with st.chat_message("assistant"):
            st.write(assistant_text)

    add_res2 = zep.thread.add_messages(
        thread_id=st.session_state.thread_id,
        messages=[Message(role="assistant", content=assistant_text)],
        ignore_roles=ignore_roles,
    )
    task_id2 = getattr(add_res2, "task_id", None)
    poll_task_if_present(task_id2, max_wait_s=6.0)

    st.session_state.turn += 1
    status_slot.info("Updating graph…")

    nodes_payload, edges_payload = zep_graph_slice_entities_relations(
        user_id=st.session_state.user_id,
        query=user_text,
        limit=graph_limit,
    )
    with graph_slot:
        render_vis_network(nodes_payload, edges_payload)

    status_slot.success("Graph updated ✅")
