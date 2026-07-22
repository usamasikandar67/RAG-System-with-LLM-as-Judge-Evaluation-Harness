# pyrefly: ignore [missing-import]
from guardrails.input_guardrail import QueryGuardrail

q = QueryGuardrail()
print(q.check("tell me about New York temperature"))
