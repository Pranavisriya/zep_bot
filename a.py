# app.py
import os
import time
import uuid
import json
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


# ------------------------- session -------------------------
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


def poll_task_if_present(task_id: Optional[str], max_wait_s: float = 2.0):
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
        time.sleep(0.2)


# ------------------------- Graph helpers (Zep-first) -------------------------
def color_for_labels(labels: List[str]) -> str:
    s = " ".join([str(l).lower() for l in labels]) if labels else ""
    if "person" in s or "user" in s:
        return "#79A7FF"
    if "org" in s or "organization" in s or "company" in s:
        return "#7FE0B6"
    if "location" in s or "place" in s:
        return "#F49B6A"
    if "preference" in s or "interest" in s:
        return "#B8A5FF"
    return "#F0C36D"


def get_node_cached(uuid_: str) -> Dict[str, Any]:
    """
    Fetch node details from Zep and cache them.
    """
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


def is_entity_node(node: Dict[str, Any]) -> bool:
    """
    Keep only entity-like nodes with real names.
    """
    name = (node.get("name") or "").strip()
    labels = [str(x).lower() for x in (node.get("labels") or [])]

    bad = {"message", "episode", "thread", "turn", "conversation", "chat", "system"}
    if any(b in labels for b in bad):
        return False

    return bool(name)


def zep_graph_slice_entities_relations(
    user_id: str,
    query: str,
    limit: int = 30,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For visualization:
      - Nodes shown ONLY if Zep node.name exists (entity)
      - Edges shown ONLY if Zep edge.name exists (relation)
    """
    limit = min(int(limit), 50)

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
        rel = (getattr(e, "name", "") or "").strip()
        fact = (getattr(e, "fact", "") or "").strip()

        if not rel:
            continue

        raw_edges.append((src, tgt, rel, fact))
        endpoint_ids.add(src)
        endpoint_ids.add(tgt)

    node_meta: Dict[str, Dict[str, Any]] = {}
    for nid in endpoint_ids:
        node = get_node_cached(nid)
        if is_entity_node(node):
            node_meta[nid] = node

    edges_payload: List[Dict[str, Any]] = []
    for src, tgt, rel, fact in raw_edges:
        if src not in node_meta or tgt not in node_meta:
            continue
        edges_payload.append(
            {
                "from": src,
                "to": tgt,
                "label": rel,
                "title": fact or rel,
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
                "label": meta.get("name") or nid[:8],
                "title": meta.get("summary") or meta.get("name") or nid[:8],
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
                "label": "No named entity nodes yet",
                "title": "Zep hasn't extracted named entities/relations yet. Try: 'My profession is AI Engineer.'",
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
        interaction: {{
          hover: true,
          navigationButtons: true,
          keyboard: true
        }},
        nodes: {{
          borderWidth: 1,
          font: {{ size: 16, color: "rgba(255,255,255,0.92)" }}
        }},
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


# ------------------------- SIMPLE ZEP MEMORY (GRAPH FACTS) -> LLM -------------------------
def retrieve_relevant_graph_facts(user_id: str, query: str, limit: int = 6) -> str:
    """
    This is the key fix:
    - Use Zep Graph search (edges) to get extracted facts.
    - Convert to: Entity --RELATION--> Entity (fact)
    """
    limit = max(1, min(int(limit), 50))

    # Small trick: include user_id so queries like "my profession" still anchor to the user
    q = f"{query} {user_id}"

    edges_res = zep.graph.search(user_id=user_id, query=q, scope="edges", limit=limit)
    edges = getattr(edges_res, "edges", []) or []

    lines: List[str] = []
    for e in edges:
        rel = (getattr(e, "name", "") or "").strip()
        if not rel:
            continue

        src_id = getattr(e, "source_node_uuid", None) or getattr(e, "source", None)
        tgt_id = getattr(e, "target_node_uuid", None) or getattr(e, "target", None)
        if not src_id or not tgt_id:
            continue

        src_id, tgt_id = str(src_id), str(tgt_id)
        src = get_node_cached(src_id)
        tgt = get_node_cached(tgt_id)

        src_name = (src.get("name") or "").strip() or src_id[:8]
        tgt_name = (tgt.get("name") or "").strip() or tgt_id[:8]
        fact = (getattr(e, "fact", "") or "").strip()

        # Keep it simple and LLM-friendly
        if fact:
            lines.append(f"{src_name} --{rel}--> {tgt_name}. Fact: {fact}")
        else:
            lines.append(f"{src_name} --{rel}--> {tgt_name}")

    return "\n".join(lines)


def llm_answer_with_graph_facts(user_query: str, facts_block: str) -> str:
    system_prompt = (
        "You are a helpful assistant.\n"
        "You are given FACTS retrieved from the user's Zep Knowledge Graph.\n"
        "Use those facts to answer.\n"
        "If the facts do not contain the answer, say: 'I don't know based on your saved memory.'\n"
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


# ------------------------- app -------------------------
ensure_session()
ensure_user_and_thread(st.session_state.user_id, st.session_state.thread_id)

st.title("Zep Chatbot + Graph (Hybrid Memory → LLM)")

with st.sidebar:
    st.code(f"user_id:  {st.session_state.user_id}\nthread_id:{st.session_state.thread_id}")
    ingest_assistant = st.toggle("Ingest assistant into graph", value=True)
    graph_limit = st.slider("Graph limit", 5, 50, 30, 5)
    top_k_facts = st.slider("Top graph facts (hybrid)", 2, 20, 6, 1)

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

# initial graph render
nodes_payload, edges_payload = zep_graph_slice_entities_relations(
    user_id=st.session_state.user_id,
    query="profile",
    limit=graph_limit,
)
with graph_slot:
    render_vis_network(nodes_payload, edges_payload)

if user_text:
    # show user message
    st.session_state.chat.append({"role": "user", "content": user_text})
    with col_chat:
        with st.chat_message("user"):
            st.write(user_text)

    # ✅ GET RELEVANT FACTS FROM ZEP GRAPH (this will include PROFESSION_IS -> AI Engineer)
    facts_block = retrieve_relevant_graph_facts(
        user_id=st.session_state.user_id,
        query=user_text,
        limit=top_k_facts,
    )

    # ✅ LLM ANSWER USING ONLY THOSE FACTS
    assistant_text = llm_answer_with_graph_facts(user_text, facts_block)

    # show assistant
    st.session_state.chat.append({"role": "assistant", "content": assistant_text})
    with col_chat:
        with st.chat_message("assistant"):
            st.write(assistant_text)

    # store into zep
    ignore_roles = [] if ingest_assistant else ["assistant"]
    add_res = zep.thread.add_messages(
        thread_id=st.session_state.thread_id,
        messages=[
            Message(role="user", content=user_text),
            Message(role="assistant", content=assistant_text),
        ],
        ignore_roles=ignore_roles,
    )

    task_id = getattr(add_res, "task_id", None)
    poll_task_if_present(task_id, max_wait_s=2.0)

    # refresh graph
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
