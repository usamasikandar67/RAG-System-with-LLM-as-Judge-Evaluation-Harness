# src/dashboard/app.py

import os
import sys
import sqlite3
import json
import pandas as pd
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# pyrefly: ignore [missing-import]
import streamlit as st

from ingestion.ingestion import ingest_documents
from embeddings.factory import get_embedding_model
from retrieval.vector import SimpleVectorRetriever
from retrieval.lexical import BM25Retriever
from retrieval.hybrid import HybridRetriever
from ingestion.pipeline import RAGPipeline
from judge.factory import get_judge_model
from evaluation.evaluator import EvaluationEngine
from experiments.db import save_chat_message, load_chat_history

project_root = os.path.dirname(src_dir)
DB_PATH = os.path.join(project_root, "src", "experiments", "evaluation_results.db")


# Page configuration
st.set_page_config(
    page_title="Medical Clinical AI Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)




@st.cache_resource
def get_cached_pipeline(retriever_name: str, generator_model: str, chunk_size: int, chunk_overlap: int, chunking_enabled: bool = True):
    # Ingest baseline documents
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "knowledge_base")
    chunks = ingest_documents(kb_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunking_enabled=chunking_enabled)
    embed = get_embedding_model("mock")
    
    if retriever_name == "vector":
        retriever = SimpleVectorRetriever(embedding_model=embed)
    elif retriever_name == "lexical":
        retriever = BM25Retriever()
    else: # hybrid
        v_ret = SimpleVectorRetriever(embedding_model=embed)
        l_ret = BM25Retriever()
        retriever = HybridRetriever(vector_retriever=v_ret, lexical_retriever=l_ret)
        
    retriever.index_documents(chunks)
    pipeline = RAGPipeline(retriever=retriever, generator_model=generator_model)
    judge = get_judge_model("mock")
    evaluator = EvaluationEngine(judge_model=judge)
    return pipeline, evaluator

@st.cache_data(ttl=60)
def get_kb_stats():
    import random
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "knowledge_base")
    if not os.path.exists(kb_path):
        return {"total_docs": 0, "topics": [], "sample_questions": []}
    
    files = [f for f in os.listdir(kb_path) if f.endswith('.txt')]
    topics = []
    questions = []
    
    sample_files = random.sample(files, min(100, len(files)))
    
    for f in sample_files:
        try:
            with open(os.path.join(kb_path, f), 'r', encoding='utf-8') as file:
                content = file.read()
                for line in content.split('\n'):
                    if line.startswith("Topic: "):
                        topics.append(line.replace("Topic: ", "").strip())
                        break
                q_lines = [line.replace("Q: ", "").strip() for line in content.split('\n') if line.startswith("Q: ")]
                questions.extend(q_lines)
        except Exception:
            pass
            
    return {
        "total_docs": len(files),
        "sample_topics": list(set(topics))[:10],
        "sample_questions": list(set(questions))[:15],
    }

# Helper function to get database connection
def get_db_connection():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Load experiments list
@st.cache_data(ttl=5)
def load_experiments():
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        return []

@st.cache_data(ttl=5)
def load_runs(experiment_id):
    try:
        conn = get_db_connection()
        query = """
            SELECT 
                g.query_id as test_id,
                q.query_text as question,
                g.response_text,
                g.latency_ms as latency,
                COALESCE(MAX(t.total_cost), 0.0) as cost,
                COALESCE(MAX(t.total_tokens), 0) as total_tokens,
                MAX(CASE WHEN e.metric_name = 'hit_1' THEN e.metric_value END) as hit_1,
                MAX(CASE WHEN e.metric_name = 'hit_3' THEN e.metric_value END) as hit_3,
                MAX(CASE WHEN e.metric_name = 'hit_5' THEN e.metric_value END) as hit_5,
                MAX(CASE WHEN e.metric_name = 'reciprocal_rank' THEN e.metric_value END) as reciprocal_rank,
                MAX(CASE WHEN e.metric_name = 'average_precision' THEN e.metric_value END) as average_precision,
                MAX(CASE WHEN e.metric_name = 'correctness' THEN e.metric_value END) as correctness,
                MAX(CASE WHEN e.metric_name = 'completeness' THEN e.metric_value END) as completeness,
                MAX(CASE WHEN e.metric_name = 'faithfulness' THEN e.metric_value END) as faithfulness,
                MAX(CASE WHEN e.metric_name = 'citation_accuracy' THEN e.metric_value END) as citation_accuracy,
                MAX(CASE WHEN e.metric_name = 'clinical_utility' THEN e.metric_value END) as clinical_utility,
                g.created_at as timestamp
            FROM experiment_runs r
            JOIN generations g ON r.generation_id = g.generation_id
            JOIN queries q ON g.query_id = q.query_id
            JOIN evaluations e ON r.eval_id = e.eval_id
            LEFT JOIN traces t ON t.session_id = r.experiment_id AND t.input_text = q.query_text
            WHERE r.experiment_id = ?
            GROUP BY g.generation_id
        """
        df = pd.read_sql_query(query, conn, params=(experiment_id,))
        conn.close()
        
        expected_cols = [
            "test_id", "question", "response_text", "latency", "cost", "total_tokens",
            "hit_1", "hit_3", "hit_5", "reciprocal_rank", "average_precision", 
            "correctness", "completeness", "faithfulness", "citation_accuracy", "clinical_utility", "timestamp"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        
        return df
    except Exception as e:
        st.error(f"Error loading runs: {e}")
        return pd.DataFrame()
# Main UI
st.title("🏥 Medical Clinical AI Platform")
st.markdown("### Automated AI Quality Assurance and Regression Gating Dashboard")
st.write("---")

# Sidebar configurations
st.sidebar.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
st.sidebar.markdown("## Control Panel")

# 1. User Role Division
st.sidebar.markdown("### User Persona & Access Control")
user_persona = st.sidebar.pills("Select System Persona", ["Clinician (Doctor)", "Platform Auditor (Admin)"], default="Clinician (Doctor)")

# 2. Multi-LLM Selectors
st.sidebar.markdown("### Inference Engine")
active_llm = st.sidebar.selectbox("Active LLM Generator", ["huggingface:mistralai/Mistral-7B-Instruct-v0.2", "huggingface:meta-llama/Meta-Llama-3-8B-Instruct", "mock", "gpt-4o-mini", "gpt-3.5-turbo", "bedrock:claude-3-haiku", "ollama:mistral", "ollama:llama3", "xai:grok-2-latest", "gemini:gemini-1.5-flash", "gemini:gemini-1.5-pro", "longcat:gpt-4o", "longcat:claude-3-5-sonnet-20240620", "groq:llama-3.3-70b-versatile", "groq:llama-3.1-8b-instant"])

# 3. Chunking Toggle
st.sidebar.markdown("### Document Processing")
chunking_enabled = st.sidebar.toggle("Chunking Enabled", value=True, help="When OFF, each document is treated as a single chunk without splitting.")

# 4. Chat Session Persistence
st.sidebar.markdown("### Chat History Session")
session_id = st.sidebar.text_input("Session Identifier ID", "default_doctor_session")

experiments = load_experiments()
if not experiments:
    st.info("No evaluations run yet. Please run run_pipeline.py first to populate results.")
else:
    exp_options = {f"{exp['name']} ({exp['experiment_id']})": exp for exp in experiments}
    selected_name = st.sidebar.selectbox("Active Experiment Setup", list(exp_options.keys()))
    active_exp = exp_options[selected_name]
    
    # Show settings
    st.sidebar.write("---")
    st.sidebar.markdown("### Active Experiment Settings")
    try:
        settings = json.loads(active_exp["rag_settings"])
        st.sidebar.json(settings)
    except:
        st.sidebar.json({"retriever": "hybrid", "chunk_size": 800, "chunk_overlap": 150})
        settings = {"retriever": "hybrid", "chunk_size": 800, "chunk_overlap": 150}
        
    # Load Runs Data
    df_runs = load_runs(active_exp["experiment_id"])
    
    if df_runs.empty:
        st.warning("Selected experiment contains no evaluation runs.")
    else:
        # RENDER TABS BASED ON PERSONA ROLE
        if user_persona == "Clinician (Doctor)":
            tab_chatbot, tab_guidelines = st.tabs([":material/chat: Doctor Clinical Chatbot", ":material/folder: Upload & Edit Guidelines"])
            
            # CLINICIAN TAB 1: DOCTOR CLINICAL CHATBOT
            with tab_chatbot:
                st.subheader(f"🏥 Medical Clinical Assistant ({active_llm.upper()})")
                st.markdown(f"*Asking guidelines questions using active retrieval strategy ({settings.get('retriever', 'hybrid')}). Session: `{session_id}`*")
                st.write("---")
                
                # Load message history from SQLite
                chat_history = load_chat_history(DB_PATH, session_id)
                
                # Display history
                for chat in chat_history:
                    with st.chat_message(chat["role"]):
                        st.markdown(chat["content"])
                        if chat["contexts"]:
                            with st.expander("Audit Reference Guidelines Chunks"):
                                for doc in chat["contexts"]:
                                    st.markdown(f"**Source File:** `{doc.get('metadata', {}).get('source_file')}` | **Category:** {doc.get('metadata', {}).get('clinical_category', 'General Medicine')}")
                                    st.text(doc.get("text"))
                
                kb_stats = get_kb_stats()
                if kb_stats["sample_questions"]:
                    with st.expander("💡 Suggested Questions from Knowledge Base", expanded=not bool(chat_history)):
                        st.write("Try asking one of these questions based on the current guidelines:")
                        for q in kb_stats["sample_questions"][:5]:
                            st.markdown(f"- {q}")

                # Chat input
                if user_query := st.chat_input("Ask clinical guidelines questions..."):
                    with st.chat_message("user"):
                        st.markdown(user_query)
                    # Persist user question
                    save_chat_message(DB_PATH, session_id, "user", user_query, [])
                    
                    # Fetch cached pipeline matching active experiment
                    p_type = settings.get("retriever", "hybrid")
                    c_size = settings.get("chunk_size", 800)
                    c_over = settings.get("chunk_overlap", 150)
                    pipeline, evaluator = get_cached_pipeline(p_type, active_llm, c_size, c_over, chunking_enabled)
                    
                    with st.spinner("Retrieving clinical guidelines and assessing safety..."):
                        result = pipeline.generate(user_query, k=3, chat_history=chat_history)
                    response_text = result["response_text"]
                    contexts = result["retrieved_contexts"]
                    q_ents = result.get("query_entities", {})
                    rewritten = result.get("rewritten_query", user_query)
                    
                    # Extract enriched analysis from result
                    classical_ai    = result.get("classical_ai", {})
                    phi_scan        = result.get("phi_scan", {})
                    pih_risk        = result.get("pih_risk", {})
                    complaint_data  = result.get("complaints", {})
                    service_tag     = result.get("service_tag", {})
                    confidence      = result.get("confidence", {})
                    verification    = result.get("verification", {})
                    retrieval_val   = result.get("retrieval_validation", {})
                    icd_data        = result.get("icd_data", {})
                    latency_bd      = result.get("latency_breakdown", {})
                    token_usage     = result.get("token_usage", {})
                    cost            = result.get("cost", 0.0)

                    with st.chat_message("assistant"):
                        st.markdown(response_text)

                        # ── Top Metric Strip ────────────────────────────────────
                        with st.container(horizontal=True):
                            st.metric("🎯 Confidence", confidence.get("overall_pct", "N/A") if isinstance(confidence, dict) else "N/A", confidence.get("overall_label", "") if isinstance(confidence, dict) else "", border=True)
                            st.metric("📡 Retriever Score", f"{retrieval_val.get('stats', {}).get('max_score', 0):.2f}" if isinstance(retrieval_val, dict) else "N/A", border=True)
                            st.metric("💬 Tokens", token_usage.get("total_tokens", 0) if isinstance(token_usage, dict) else 0, border=True)
                            st.metric("💰 Cost", f"${cost:.5f}", border=True)
                            st.metric("⏱️ Latency", f"{result.get('latency', 0):.2f}s", border=True)
                            hall = "❌ Yes" if verification.get("hallucination_detected") else "✅ No"
                            st.metric("🧠 Hallucination", hall, border=True)

                        # ── Clinical Analysis Row ────────────────────────────────
                        analysis_cols = st.columns(5)
                        with analysis_cols[0]:
                            urgency_raw = classical_ai.get("urgency", {})
                            urgency = urgency_raw.get("urgency", "medium") if isinstance(urgency_raw, dict) else str(urgency_raw)
                            urgency_colors = {"high": "🔴", "medium": "🟡", "low": "🟢", "none": "⚪"}
                            st.markdown(f"**Urgency:** {urgency_colors.get(urgency, '⚪')} {urgency.upper()}")
                        with analysis_cols[1]:
                            topic_raw = classical_ai.get("topic", {})
                            topic = topic_raw.get("topic", "General") if isinstance(topic_raw, dict) else str(topic_raw)
                            st.markdown(f"**Topic:** 🏥 {topic}")
                        with analysis_cols[2]:
                            svc = service_tag.get("primary_service", "General") if isinstance(service_tag, dict) else str(service_tag)
                            st.markdown(f"**Service:** 🏷️ {svc}")
                        with analysis_cols[3]:
                            phi_flag = phi_scan.get("has_phi", False) if isinstance(phi_scan, dict) else False
                            st.markdown(f"**PHI:** {'🚨 DETECTED' if phi_flag else '✅ Clean'}")
                        with analysis_cols[4]:
                            cmplnt_list = complaint_data.get("complaints", []) if isinstance(complaint_data, dict) else []
                            symptoms = [c.get("symptom") for c in cmplnt_list if isinstance(c, dict) and c.get("symptom")]
                            cmplnt_str = ", ".join(symptoms[:2]) if symptoms else "None"
                            st.markdown(f"**Symptom:** 🩺 {cmplnt_str}")

                        # ── Rewritten Query Notice ───────────────────────────────
                        if rewritten != user_query:
                            st.info(f"🔄 **Query was rewritten for better retrieval:**\n> {rewritten}")

                        # ── ICD Code ─────────────────────────────────────────────
                        if isinstance(icd_data, dict) and icd_data.get("code") and icd_data["code"] != "C80":
                            st.caption(f"🏷️ ICD-10: **{icd_data['code']}** — {icd_data.get('description', '')} ({icd_data.get('category', '')})")

                        # ── PHI Warning ──────────────────────────────────────
                        if isinstance(phi_scan, dict) and phi_scan.get("has_phi"):
                            st.warning(f"⚠️ PHI Detected in query! Risk Level: {phi_scan.get('risk_level', 'unknown').upper()} | Found {phi_scan.get('phi_count', 0)} sensitive items.")

                        # ── PIH Risk ─────────────────────────────────────────────
                            pih_colors = {"high": "error", "moderate": "warning", "low": "info"}
                            pih_level = None
                            getattr(st, pih_colors.get(pih_level, "info"))(f"🤰 PIH Risk: {pih_level.upper()} (Score: {pih_risk.get('score', 0)}) — Indicators: {', '.join(i['term'] for i in pih_risk.get('indicators', []))}")

                        # ── Hallucination / Verification Detail ──────────────────
                        if verification.get("hallucination_detected") or not verification.get("passed", True):
                            st.warning("I don't have knowledge regarding the user query.")
                            unsupported = verification.get("unsupported_claims", [])
                            if unsupported:
                                st.caption(f"Unsupported claims: {', '.join(unsupported)}")

                        # ── Complaint Extraction ──────────────────────────────────
                        if isinstance(complaint_data, dict) and complaint_data.get("complaint_count", 0) > 0:
                            with st.expander(f"📋 Extracted Complaints ({complaint_data['complaint_count']})"):
                                for c in complaint_data.get("complaints", []):
                                    st.markdown(f"- **{c['symptom'].title()}** | Site: {c.get('body_site', 'N/A')} | Severity: {c.get('severity', 'N/A')} | Duration: {c.get('duration', 'N/A')}")

                        # ── Clinical NER Entities ─────────────────────────────────
                        if q_ents and any(q_ents.values()):
                            with st.expander("🧬 Extracted Clinical Entities"):
                                for key, vals in q_ents.items():
                                    if vals:
                                        st.markdown(f"- **{key.replace('_', ' ').title()}:** {', '.join(vals)}")

                        # ── Retrieved Evidence ────────────────────────────────────
                        if contexts:
                            with st.expander(f"📚 Retrieved Evidence ({len(contexts)} chunks)"):
                                for i, doc in enumerate(contexts, 1):
                                    meta = doc.get("metadata", {})
                                    score = doc.get("rerank_score") or doc.get("score") or 0
                                    icd_c = meta.get("icd_10_code", "")
                                    st.markdown(
                                        f"**[Doc{i}]** `{meta.get('source_file', 'unknown')}` | "
                                        f"Category: **{meta.get('clinical_category', 'General Medicine')}** | "
                                        f"ICD: `{icd_c}` | Similarity: **{score:.3f}**"
                                    )
                                    st.text(doc.get("text", ""))
                                    st.divider()

                        # ── Latency Breakdown ─────────────────────────────────────
                        if latency_bd:
                            with st.expander("⏱️ Pipeline Latency Breakdown"):
                                for step, ms in latency_bd.items():
                                    st.progress(min(1.0, ms / max(latency_bd.values())), text=f"{step}: {ms} ms")

                    # Persist assistant response
                    save_chat_message(DB_PATH, session_id, "assistant", response_text, contexts)
                    st.rerun()

            # CLINICIAN TAB 2: DATA MANAGER
            with tab_guidelines:
                st.subheader(":material/folder: Reference Guidelines Knowledge Base Editor")
                
                kb_stats = get_kb_stats()
                with st.container(border=True):
                    st.markdown("### 📊 Knowledge Base Statistics")
                    with st.container(horizontal=True):
                        st.metric("Total Protocols", kb_stats.get("total_docs", 0), border=True)
                        st.metric("Indexed Topics", len(kb_stats.get("sample_topics", [])), border=True)
                        
                    if kb_stats.get("sample_topics"):
                        st.markdown("**Sample Covered Topics:** " + ", ".join(kb_stats["sample_topics"][:5]) + "...")

                st.write("---")
                
                st.markdown("Update guideline documents instantly to improve the AI's clinical context references.")
                
                with st.form("guideline_form", clear_on_submit=True):
                    doc_name = st.text_input("Protocol File Name (e.g. clinical_guidelines_2026.txt)", placeholder="guidelines.txt")
                    doc_text = st.text_area("Guideline Content (Clinical Text)", height=250, placeholder="Paste reference guidelines content here...")
                    submit_btn = st.form_submit_button("Publish & Re-Index Guidelines")
                    
                    if submit_btn:
                        if doc_name and doc_text:
                            os.makedirs("datasets/knowledge_base", exist_ok=True)
                            filepath = os.path.join("datasets/knowledge_base", doc_name)
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.write(doc_text)
                            st.success(f"Successfully saved and registered guidelines as: `datasets/knowledge_base/{doc_name}`!")
                            
                            # Clear streamlit cache to force indexing update
                            get_cached_pipeline.clear()
                            st.info("System memory cache cleared. Active retrievers will rebuild the guidelines database on the next query.")
                        else:
                            st.warning("Both Document Name and Clinical Text fields are required.")

        else: # Platform Auditor (Admin)
            tab_summary, tab_explorer, tab_analytics, tab_history, tab_models = st.tabs([
                ":material/bar_chart: Executive Summary", ":material/search: Case Explorer", ":material/monitoring: Telemetry Analytics", ":material/history: Clinician Chat Audits", ":material/smart_toy: Model Comparison"
            ])
            
            # ADMIN TAB 1: EXECUTIVE SUMMARY
            with tab_summary:
                st.subheader("System Performance & Gating Summary")
                avg_utility = df_runs["clinical_utility"].mean()
                avg_correctness = df_runs["correctness"].mean()
                avg_faithfulness = df_runs["faithfulness"].mean()
                avg_map = df_runs["average_precision"].mean()
                avg_latency = df_runs["latency"].mean()
                
                with st.container(horizontal=True):
                    st.metric("Clinical Utility", f"{avg_utility:.4f}", border=True)
                    st.metric("Correctness", f"{avg_correctness:.4f}", border=True)
                    st.metric("Faithfulness", f"{avg_faithfulness:.4f}", border=True)
                    st.metric("Mean Average Precision", f"{avg_map:.4f}", border=True)
                
                st.write(" ")
                st.subheader("Gating Recommendation Status")
                is_pass = avg_correctness >= 0.90 and avg_faithfulness >= 0.35 and avg_map >= 0.20
                if is_pass:
                    st.badge("PASS - Candidate version meets medical safety guidelines standards.", icon=":material/check:", color="green")
                else:
                    st.badge("FAIL - Metric targets not met. Code regressions detected.", icon=":material/error:", color="red")
                
                st.write("---")
                st.subheader("All Evaluation Runs Logs")
                st.dataframe(
                    df_runs[["test_id", "question", "clinical_utility", "correctness", "completeness", "faithfulness", "citation_accuracy", "latency", "cost"]]
                )

            # ADMIN TAB 2: CASE EXPLORER
            with tab_explorer:
                st.subheader("Trace Case Debugger")
                selected_test_id = st.selectbox("Select Test Case ID", df_runs["test_id"].tolist())
                case_data = df_runs[df_runs["test_id"] == selected_test_id].iloc[0]
                
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"**Question:**\n{case_data['question']}")
                    st.write(" ")
                    st.markdown(f"**Generated Answer:**\n{case_data['response_text']}")
                with col_right:
                    st.markdown("### Evaluation Metrics")
                    with st.container(horizontal=True):
                        st.metric("Correctness", f"{float(case_data['correctness']):.2f}", border=True)
                        st.metric("Completeness", f"{float(case_data['completeness']):.2f}", border=True)
                        st.metric("Faithfulness", f"{float(case_data['faithfulness']):.2f}", border=True)
                        
                    st.markdown("### Performance Telemetry")
                    with st.container(horizontal=True):
                        st.metric("Latency (s)", f"{case_data['latency']:.4f}", border=True)
                        st.metric("Token Cost ($)", f"{case_data['cost']:.6f}", border=True)

            # ADMIN TAB 3: TELEMETRY ANALYTICS
            with tab_analytics:
                st.subheader("System Performance & Token Analytics")
                col_l, col_r, col_t = st.columns(3)
                with col_l:
                    st.markdown("**Latency Distribution (Seconds)**")
                    st.line_chart(df_runs["latency"])
                with col_r:
                    st.markdown("**Inference Cost Trend ($)**")
                    st.area_chart(df_runs["cost"])
                with col_t:
                    st.markdown("**Token Usage**")
                    st.bar_chart(df_runs["total_tokens"])
                
                st.subheader("🔍 Langfuse Trace Explorer")
                st.markdown("Inspect granular pipeline spans and latency waterfalls.")
                try:
                    conn = get_db_connection()
                    df_traces = pd.read_sql_query(
                        "SELECT trace_id, name, total_latency_ms, total_cost, total_tokens, created_at FROM traces WHERE session_id = ? ORDER BY created_at DESC", 
                        conn, params=(active_exp["experiment_id"],)
                    )
                    
                    if not df_traces.empty:
                        selected_trace = st.selectbox("Select Trace to Inspect", df_traces["trace_id"].tolist())
                        trace_summary = df_traces[df_traces["trace_id"] == selected_trace].iloc[0]
                        with st.container(horizontal=True):
                            st.metric("Total Latency", f"{trace_summary['total_latency_ms']} ms", border=True)
                            st.metric("Total Tokens", f"{trace_summary['total_tokens']}", border=True)
                            st.metric("Cost", f"${trace_summary['total_cost']:.6f}", border=True)
                        st.write(" ")
                        
                        df_spans = pd.read_sql_query(
                            "SELECT name, duration_ms, input_tokens, output_tokens, status FROM spans WHERE trace_id = ? ORDER BY start_time ASC",
                            conn, params=(selected_trace,)
                        )
                        st.dataframe(df_spans)
                    else:
                        st.info("No traces found for this experiment run.")
                    conn.close()
                except Exception as e:
                    st.error(f"Error loading traces: {e}")

            # ADMIN TAB 4: CHAT AUDITS TRAIL
            with tab_history:
                st.subheader("📋 Clinician Chat Queries Audit Trail")
                st.markdown("Audit trails of clinician chatbot sessions logged directly inside SQLite for institutional safety checks.")
                
                try:
                    conn = get_db_connection()
                    df_history = pd.read_sql_query(
                        "SELECT timestamp, session_id, role, content FROM chatbot_history ORDER BY id DESC", 
                        conn
                    )
                    conn.close()
                    if df_history.empty:
                        st.info("No query logs saved in audit trail database yet.")
                    else:
                        st.dataframe(df_history)
                except Exception as e:
                    st.error(f"Error loading chat history logs: {e}")

            # ADMIN TAB 5: MODEL COMPARISON
            with tab_models:
                st.subheader("🤖 Model Comparison: Groq Instant vs Versatile")
                st.markdown("Compare the performance of Groq's Instant and Versatile LLMs across critical safety metrics.")
                
                # We need to extract the generator model from the rag_settings json for each run
                try:
                    conn = get_db_connection()
                    # Join eval_runs with experiments to get the rag_settings
                    query = """
                    SELECT 
                        MAX(CASE WHEN ev.metric_name = 'correctness' THEN ev.metric_value END) as correctness,
                        MAX(CASE WHEN ev.metric_name = 'faithfulness' THEN ev.metric_value END) as faithfulness,
                        MAX(CASE WHEN ev.metric_name = 'clinical_utility' THEN ev.metric_value END) as clinical_utility,
                        g.latency_ms as latency,
                        COALESCE(MAX(t.total_cost), 0.0) as cost,
                        COALESCE(MAX(t.total_tokens), 0) as total_tokens,
                        e.config as rag_settings
                    FROM experiment_runs r
                    JOIN experiments e ON r.experiment_id = e.experiment_id
                    JOIN generations g ON r.generation_id = g.generation_id
                    JOIN queries q ON g.query_id = q.query_id
                    JOIN evaluations ev ON r.eval_id = ev.eval_id
                    LEFT JOIN traces t ON t.session_id = r.experiment_id AND t.input_text = q.query_text
                    GROUP BY g.generation_id, e.config
                    """
                    df_models = pd.read_sql_query(query, conn)
                    conn.close()
                    
                    if not df_models.empty:
                        # Extract generator name from JSON string
                        import json
                        df_models['generator'] = df_models['rag_settings'].apply(lambda x: json.loads(x).get('generator', 'unknown'))
                        
                        # Filter to only groq models
                        groq_models = ["groq:llama-3.1-8b-instant", "groq:llama-3.3-70b-versatile"]
                        df_models = df_models[df_models['generator'].isin(groq_models)]
                        
                        if not df_models.empty:
                            # Group by generator and calculate means
                            summary_df = df_models.groupby('generator').agg({
                                'correctness': 'mean',
                                'faithfulness': 'mean',
                                'clinical_utility': 'mean',
                                'latency': 'mean',
                                'cost': 'mean',
                                'total_tokens': 'mean',
                                'generator': 'count'
                            }).rename(columns={'generator': 'eval_runs_count'}).reset_index()
                            
                            st.dataframe(summary_df)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Faithfulness by Model**")
                                st.bar_chart(summary_df.set_index('generator')['faithfulness'])
                            with col2:
                                st.markdown("**Latency by Model (seconds)**")
                                st.bar_chart(summary_df.set_index('generator')['latency'])
                            with col3:
                                st.markdown("**Cost per Run ($)**")
                                st.bar_chart(summary_df.set_index('generator')['cost'])
                                
                            st.markdown("**Trade-off Scatter: Latency vs Faithfulness**")
                            scatter_data = summary_df[['generator', 'latency', 'faithfulness']].set_index('generator')
                            st.scatter_chart(scatter_data, x="latency", y="faithfulness")
                        else:
                            st.info("No experiment runs found for Groq models yet. Please run the pipeline with these models to see comparison.")
                    else:
                        st.info("Not enough experiment runs to compare models yet.")
                except Exception as e:
                    st.error(f"Error generating model comparison: {e}")


