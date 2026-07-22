# src/langfuse/tracer.py

"""
Langfuse-compatible observability tracer for the Clinical RAG pipeline.

Captures structured span traces for every pipeline step:
  - Ingestion (NER, ICD mapping, PHI scan)
  - Retrieval (candidate fetch, reranking)
  - Generation (LLM invocation)
  - Evaluation (Judge scoring)

When LANGFUSE_PUBLIC_KEY is set, traces are pushed to the Langfuse cloud.
Otherwise, traces are stored locally in SQLite for dashboard inspection.
"""

import os
import time
import json
import sqlite3
import hashlib
from typing import Dict, Any, List, Optional
from contextlib import contextmanager


class LangfuseTracer:
    """
    Lightweight Langfuse-compatible tracer.
    Records spans hierarchically: Trace → Spans → Events.
    """

    def __init__(self, db_path: str = "src/experiments/evaluation_results.db"):
        self.db_path = db_path
        self.client = None
        self._init_tables()
        self._init_langfuse_client()

    def _init_tables(self):
        """Create local trace storage tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            session_id TEXT,
            user_id TEXT,
            input_text TEXT,
            output_text TEXT,
            total_latency_ms REAL,
            total_cost REAL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            metadata_json TEXT,
            created_at TEXT NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            span_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            parent_span_id TEXT,
            name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration_ms REAL,
            input_json TEXT,
            output_json TEXT,
            metadata_json TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            level TEXT DEFAULT 'DEFAULT',
            status TEXT DEFAULT 'OK',
            FOREIGN KEY (trace_id) REFERENCES traces (trace_id)
        )
        """)

        conn.commit()
        conn.close()

    def _init_langfuse_client(self):
        """Try to connect to Langfuse cloud if credentials are available."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if public_key and secret_key:
            try:
                # pyrefly: ignore [missing-import]
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                print(f"[Langfuse] Connected to {host}")
            except Exception as e:
                print(f"[Langfuse] Cloud connection failed: {e}. Using local tracing.")
                self.client = None
        else:
            print("[Langfuse] No credentials found. Using local SQLite tracing.")

    def create_trace(self, name: str, session_id: str = None,
                     user_id: str = None, input_text: str = None) -> str:
        """Start a new trace for a pipeline invocation."""
        trace_id = f"trace_{int(time.time() * 1000)}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO traces (trace_id, name, session_id, user_id, input_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (trace_id, name, session_id, user_id, input_text, created_at))
        conn.commit()
        conn.close()

        # Also push to Langfuse cloud if connected
        if self.client:
            try:
                self.client.trace(
                    id=trace_id,
                    name=name,
                    session_id=session_id,
                    user_id=user_id,
                    input=input_text
                )
            except Exception:
                pass

        return trace_id

    def start_span(self, trace_id: str, name: str,
                   parent_span_id: str = None,
                   input_data: Dict = None, level: str = "DEFAULT") -> str:
        """Start a span within a trace."""
        span_id = f"span_{int(time.time() * 10000)}_{name[:10]}"
        start_time = time.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(time.time() * 1000) % 1000:03d}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO spans (span_id, trace_id, parent_span_id, name, start_time, input_json, level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (span_id, trace_id, parent_span_id, name, start_time,
              json.dumps(input_data) if input_data else None, level))
        conn.commit()
        conn.close()

        return span_id

    def end_span(self, span_id: str, output_data: Dict = None,
                 metadata: Dict = None, status: str = "OK",
                 input_tokens: int = 0, output_tokens: int = 0):
        """End a span and record output + duration + tokens."""
        end_time = time.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(time.time() * 1000) % 1000:03d}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get start time for duration calc
        cursor.execute("SELECT start_time FROM spans WHERE span_id = ?", (span_id,))
        row = cursor.fetchone()
        duration_ms = 0.0
        if row:
            # Simple duration estimate
            duration_ms = time.time() * 1000  # Will be corrected below

        total_tokens = input_tokens + output_tokens

        cursor.execute("""
        UPDATE spans SET end_time = ?, output_json = ?, metadata_json = ?,
                         status = ?, duration_ms = ?,
                         input_tokens = ?, output_tokens = ?, total_tokens = ?
        WHERE span_id = ?
        """, (end_time,
              json.dumps(output_data) if output_data else None,
              json.dumps(metadata) if metadata else None,
              status, duration_ms, input_tokens, output_tokens, total_tokens, span_id))
        conn.commit()
        conn.close()

    def end_trace(self, trace_id: str, output_text: str = None,
                  total_latency_ms: float = None, total_cost: float = None,
                  input_tokens: int = 0, output_tokens: int = 0,
                  metadata: Dict = None):
        """Finalize a trace with output and token/cost summary."""
        conn = sqlite3.connect(self.db_path)
        total_tokens = input_tokens + output_tokens
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE traces SET output_text = ?, total_latency_ms = ?,
                          total_cost = ?, input_tokens = ?, output_tokens = ?,
                          total_tokens = ?, metadata_json = ?
        WHERE trace_id = ?
        """, (output_text, total_latency_ms, total_cost, input_tokens, output_tokens, total_tokens,
              json.dumps(metadata) if metadata else None, trace_id))
        conn.commit()
        conn.close()

        if self.client:
            try:
                self.client.trace(
                    id=trace_id,
                    output=output_text,
                    metadata=metadata
                )
                self.client.flush()
            except Exception:
                pass

    @contextmanager
    def span(self, trace_id: str, name: str, input_data: Dict = None,
             parent_span_id: str = None, level: str = "DEFAULT"):
        """Context manager for automatic span start/end with timing."""
        span_id = self.start_span(trace_id, name, parent_span_id, input_data, level)
        start = time.time()
        result = {"output": None, "metadata": None, "status": "OK"}

        try:
            yield result
        except Exception as e:
            result["status"] = f"ERROR: {str(e)}"
            raise
        finally:
            duration = (time.time() - start) * 1000
            self.end_span(span_id, result.get("output"), result.get("metadata"), result["status"], 
                          result.get("input_tokens", 0), result.get("output_tokens", 0))

            # Update duration correctly
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE spans SET duration_ms = ? WHERE span_id = ?", (duration, span_id))
            conn.commit()
            conn.close()

    def get_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent traces for dashboard display."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_spans_for_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Retrieve all spans for a given trace."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time ASC",
            (trace_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
