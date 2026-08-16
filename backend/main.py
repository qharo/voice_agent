import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stt import STT
from llm import LLMClient, SYSTEM_PROMPT, strip_markdown
from tts import TTSEngine
from tts_pocket import PocketTTSStreamEngine, SAMPLE_RATE
from sentence_buffer import SentenceBuffer
from profiler import profiler
from tools import registry as tool_registry
import asyncio
import traceback
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from metrics import (
    stt_latency, llm_ttft, llm_total, tts_latency, tts_ttfa,
    pipeline_total, tool_latency, tool_calls_total,
    errors_total, requests_total, active_sessions
)
import time
from queue_manager import queue_manager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

stt = STT()
llm = LLMClient()

TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")
if TTS_ENGINE == "pocket":
    tts = PocketTTSStreamEngine(os.getenv("POCKET_TTS_DIR", "/models/pocket-tts"))
else:
    tts = TTSEngine()

@app.get("/health")
def health():
    return {"status": "ok"}

async def _run_query(websocket: WebSocket, final_text: str, conversation_history: list):
    full_response = await process_voice_query(final_text, websocket, conversation_history)
    if full_response:
        conversation_history.append({"role": "assistant", "content": full_response})
    if len(conversation_history) > 20:
        conversation_history[:] = conversation_history[-20:]

    profiler.report()
    requests_total.inc()
    last_tts = ('tts_last_sentence' if 'tts_last_sentence' in profiler.markers
                else 'tts_last_frame' if 'tts_last_frame' in profiler.markers else None)
    if 'speech_end' in profiler.markers and last_tts:
        pipeline_total.observe(profiler.elapsed('speech_end', last_tts))
    if 'speech_end' in profiler.markers and 'stt_end' in profiler.markers:
        stt_latency.observe(profiler.elapsed('speech_end', 'stt_end'))
    if 'stt_end' in profiler.markers and 'llm_first_token' in profiler.markers:
        llm_ttft.observe(profiler.elapsed('stt_end', 'llm_first_token'))
    if 'speech_end' in profiler.markers and 'llm_end' in profiler.markers:
        llm_total.observe(profiler.elapsed('speech_end', 'llm_end'))
    if 'tts_first_sentence' in profiler.markers and 'tts_last_sentence' in profiler.markers:
        tts_latency.observe(profiler.elapsed('tts_first_sentence', 'tts_last_sentence'))
    await websocket.send_text("[END]")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_sessions.inc()
    profiler.reset()
    profiler.mark("ws_connect")
    audio_buffer = bytearray()
    session = None
    stream_task = None
    conversation_history = []
    query_task = None
    holding_slot = False

    async def _drain_events():
        if session:
            async for event_type, text in session.read_events():
                if event_type == "text_changed":
                    await websocket.send_text(f"[TRANS:{text}]")

    async def _cancel_query(send_end: bool):
        nonlocal session, stream_task, query_task, audio_buffer
        if query_task and not query_task.done():
            query_task.cancel()
            await asyncio.gather(query_task, return_exceptions=True)
            query_task = None
        if session:
            try:
                session.stop()
            except Exception:
                pass
        session = None
        stream_task = None
        audio_buffer.clear()
        if send_end:
            await websocket.send_text("[END]")

    async def _notify_position(pos: int):
        await websocket.send_text(f"[QUEUE:{pos}]")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                audio_data = message["bytes"]
                if query_task and not query_task.done():
                    await _cancel_query(send_end=True)
                if session is None:
                    profiler.reset()
                    profiler.mark("ws_connect")
                    session = stt.create_session()
                    session.start()
                    stream_task = asyncio.create_task(_drain_events())
                audio_buffer.extend(audio_data)
                session.feed_audio(audio_data)

            elif "text" in message:
                text_msg = message["text"]

                if text_msg == "stop":
                    if query_task and not query_task.done():
                        await _cancel_query(send_end=True)
                    session.stop()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass

                    if len(audio_buffer) > 0:
                        profiler.mark("speech_end")
                        final_text = session.get_final_text()
                        audio_buffer.clear()
                        profiler.mark("stt_end")

                        if final_text:
                            await websocket.send_text(f"You: {final_text}")
                            conversation_history.append({"role": "user", "content": final_text})

                            async def _queued_run():
                                nonlocal holding_slot
                                try:
                                    await queue_manager.acquire(_notify_position)
                                    holding_slot = True
                                    # clear queue message from UI
                                    await websocket.send_text("[QUEUE:0]")
                                    await _run_query(websocket, final_text, conversation_history)
                                finally:
                                    if holding_slot:
                                        queue_manager.release()
                                        holding_slot = False

                            query_task = asyncio.create_task(_queued_run())
                        else:
                            await websocket.send_text("[ERROR: No speech detected]")
                            await websocket.send_text("[END]")

                    session = None
                    stream_task = None

                elif text_msg == "CANCEL":
                    await _cancel_query(send_end=True)

    except WebSocketDisconnect:
        if query_task and not query_task.done():
            query_task.cancel()
    except Exception as e:
        print(f"WebSocket error: {e}")
        print(traceback.format_exc())
    finally:
        active_sessions.dec()
        if holding_slot:
            queue_manager.release()
        if query_task and not query_task.done():
            query_task.cancel()
        if stream_task and not stream_task.done():
            stream_task.cancel()

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

async def process_voice_query(text: str, websocket: WebSocket, history: list[dict] | None = None) -> str:
    tool_schemas = tool_registry.get_schemas()

    query_msgs = [{"role": "user", "content": text}]
    if history:
        query_msgs = list(history)

    max_iterations = 5

    for iteration in range(max_iterations):
        profiler.mark(f"llm_call_{iteration}")
        response = await llm.get_response(
            [{"role": "system", "content": SYSTEM_PROMPT}] + query_msgs,
            tool_schemas,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            await websocket.send_text(f"[TOOL:{msg.tool_calls[0].function.name}]")
            query_msgs.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_start = time.time()
                profiler.mark(f"tool_{tc.function.name}")
                result = await tool_registry.execute(tc.function.name, tc.function.arguments)
                tool_duration = (time.time() - tool_start) * 1000
                tool_latency.labels(tool_name=tc.function.name).observe(tool_duration)
                tool_calls_total.labels(tool_name=tc.function.name).inc()
                query_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            final_text = strip_markdown(msg.content or "")
            break
    else:
        final_text = "I'm sorry, I wasn't able to process that request."

    profiler.mark("llm_end")

    if TTS_ENGINE == "pocket":
        profiler.mark("llm_first_token")
        await websocket.send_text(f"[STREAM:{final_text}]")
        await _stream_pocket(final_text, websocket)
    else:
        profiler.mark("llm_first_token")
        await websocket.send_text(f"[STREAM:{final_text}]")
        await _stream_piper(final_text, websocket)

    return final_text


async def _stream_pocket(text: str, websocket: WebSocket):
    await websocket.send_text(f"[AUDIO_BEGIN:{SAMPLE_RATE}:1:int16]")
    first = True
    started = time.time()
    try:
        async for pcm16 in tts.stream_int16(text):
            if first:
                tts_ttfa.observe((time.time() - started) * 1000)
                profiler.mark("tts_first_frame")
                first = False
            await websocket.send_bytes(pcm16)
        profiler.mark("tts_last_frame")
    except Exception as e:
        error_msg = f"[ERROR: TTS failed - {str(e)}]"
        await websocket.send_text(error_msg)
        print(traceback.format_exc())
    finally:
        await websocket.send_text("[AUDIO_END]")


async def _stream_piper(text: str, websocket: WebSocket):
    await websocket.send_text(f"[AUDIO_BEGIN:{tts.sample_rate}:1:int16]")
    first = True
    started = time.time()
    try:
        async for chunk in tts.stream_pcm(text):
            if first:
                tts_ttfa.observe((time.time() - started) * 1000)
                profiler.mark("tts_first_frame")
                first = False
            await websocket.send_bytes(chunk)
        profiler.mark("tts_last_frame")
    except Exception as e:
        error_msg = f"[ERROR: TTS failed - {str(e)}]"
        await websocket.send_text(error_msg)
        print(traceback.format_exc())
    finally:
        await websocket.send_text("[AUDIO_END]")
