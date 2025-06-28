import os
import base64

LANGFUSE_PUBLIC_KEY=os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY=os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # US data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# ---

from smolagents import CodeAgent, OpenAIServerModel

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
model = OpenAIServerModel(
        model_id = "gpt-4o-mini",
        api_key = OPENAI_API_KEY
        )

agent = CodeAgent(tools=[], model=model)
alfred_agent = agent.from_hub('olpa/AlfredAgent', trust_remote_code=True, model=model)

# ---

from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

# ---

alfred_agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme")  
