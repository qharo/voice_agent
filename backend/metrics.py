from prometheus_client import Histogram, Gauge, Counter, Summary

# Pipeline stage latencies
stt_latency = Histogram(
    'voice_stt_latency_ms',
    'STT processing latency in milliseconds',
    buckets=[50, 100, 200, 500, 1000, 2000, 5000]
)

llm_ttft = Histogram(
    'voice_llm_time_to_first_token_ms',
    'LLM time to first token in milliseconds',
    buckets=[50, 100, 200, 500, 1000, 2000]
)

llm_total = Histogram(
    'voice_llm_total_latency_ms',
    'Total LLM generation latency in milliseconds',
    buckets=[100, 500, 1000, 2000, 5000, 10000]
)

tts_latency = Histogram(
    'voice_tts_latency_ms',
    'TTS synthesis latency in milliseconds',
    buckets=[100, 300, 500, 1000, 2000, 5000]
)

tts_ttfa = Histogram(
    'voice_tts_time_to_first_audio_ms',
    'TTS time to first audio frame in milliseconds',
    buckets=[50, 100, 200, 400, 800, 1500]
)

pipeline_total = Histogram(
    'voice_pipeline_total_ms',
    'End to end pipeline latency in milliseconds',
    buckets=[500, 1000, 2000, 3000, 5000, 10000]
)

tool_latency = Histogram(
    'voice_tool_latency_ms',
    'Tool execution latency in milliseconds',
    labelnames=['tool_name'],
    buckets=[50, 100, 200, 500, 1000, 3000]
)

# Counters
tool_calls_total = Counter(
    'voice_tool_calls_total',
    'Total number of tool calls',
    labelnames=['tool_name']
)

errors_total = Counter(
    'voice_errors_total',
    'Total number of pipeline errors',
    labelnames=['stage']
)

requests_total = Counter(
    'voice_requests_total',
    'Total number of voice requests processed'
)

# Gauges (current state)
active_sessions = Gauge(
    'voice_active_sessions',
    'Number of active WebSocket sessions'
)

queue_depth = Gauge(
    'voice_queue_depth',
    'Number of sessions waiting in queue'
)
