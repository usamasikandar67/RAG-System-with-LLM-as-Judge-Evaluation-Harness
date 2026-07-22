# pyrefly: ignore [missing-import]
from ingestion.pipeline import RAGPipeline
# pyrefly: ignore [missing-import]
from langfuse.tracer import LangfuseTracer
import os

def test_guardrails():
    # Initialize pipeline
    print("Initializing pipeline...")
    # Using 'mock' generator just for speed, we only care about guardrails checking step
    class MockRetriever:
        def retrieve(self, q, k, filter_metadata=None): return []
    pipeline = RAGPipeline(retriever=MockRetriever(), generator_model="mock")
    tracer = LangfuseTracer("src/experiments/evaluation_results.db")

    queries = [
        "What is the standard chemotherapy regimen for Stage III breast cancer?",
        "Can you give me a recipe for chocolate cake?",
        "tell me about New York temperature"
    ]

    for q in queries:
        print(f"\nTesting Query: '{q}'")
        trace_id = tracer.create_trace(
            name="Test_Guardrail",
            session_id="test_guardrail_session",
            user_id="test_user",
            input_text=q
        )
        
        result = pipeline.generate(q, k=1, tracer=tracer, trace_id=trace_id)
        
        tracer.end_trace(
            trace_id=trace_id,
            output_text=result["response_text"],
            total_latency_ms=result["latency"] * 1000,
            total_cost=result["cost"],
            input_tokens=result["token_usage"]["input_tokens"],
            output_tokens=result["token_usage"]["output_tokens"]
        )

        print(f"Result Provider: {result['provider']}")
        print(f"Response: {result['response_text']}")
        if result['provider'] == 'guardrail_blocked':
            print(f"Blocked Reason: {result.get('guardrail_reason')}")

if __name__ == "__main__":
    test_guardrails()
