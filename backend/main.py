from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stt import STT
from llm import LLMClient, SYSTEM_PROMPT
from tts import TTSEngine
from sentence_buffer import SentenceBuffer
from profiler import profiler
from tools import registry as tool_registry
import asyncio
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

stt = STT()
llm = LLMClient()
tts = TTSEngine()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    profiler.reset()
    profiler.mark("ws_connect")
    audio_buffer = bytearray()
    session = None
    stream_task = None
    conversation_history = []

    async def _drain_events():
        if session:
            async for event_type, text in session.read_events():
                if event_type == "text_changed":
                    await websocket.send_text(f"[TRANS:{text}]")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                audio_data = message["bytes"]
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
                            full_response = await process_voice_query(final_text, websocket, conversation_history)
                            if full_response:
                                conversation_history.append({"role": "assistant", "content": full_response})
                            if len(conversation_history) > 20:
                                conversation_history = conversation_history[-20:]
                        else:
                            await websocket.send_text("[ERROR: No speech detected]")

                        profiler.report()
                        await websocket.send_text("[END]")

                    session = None
                    stream_task = None

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        print(traceback.format_exc())
    finally:
        if stream_task and not stream_task.done():
            stream_task.cancel()


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
                profiler.mark(f"tool_{tc.function.name}")
                result = await tool_registry.execute(tc.function.name, tc.function.arguments)
                query_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            final_text = msg.content or ""
            break
    else:
        final_text = "I'm sorry, I wasn't able to process that request."

    full_response = []
    llm_started = False

    async def on_token(token: str):
        nonlocal llm_started
        if not llm_started:
            profiler.mark("llm_first_token")
            llm_started = True
        await websocket.send_text(f"[STREAM:{token}]")

    tts_queue = asyncio.Queue()

    async def tts_worker():
        while True:
            sentence = await tts_queue.get()
            try:
                if sentence is None:
                    break
                audio = await tts.synthesize(sentence)
                if "tts_first_sentence" not in profiler.markers:
                    profiler.mark("tts_first_sentence")
                profiler.mark("tts_last_sentence")
                await websocket.send_bytes(audio)
            except Exception as e:
                error_msg = f"[ERROR: TTS failed - {str(e)}]"
                await websocket.send_text(error_msg)
            finally:
                tts_queue.task_done()

    worker = asyncio.create_task(tts_worker())

    async def speak_sentence(sentence: str):
        await tts_queue.put(sentence)

    try:
        buffer = SentenceBuffer(speak_sentence, on_token)
        await buffer.add_token(final_text)
        await buffer.finalize()
        profiler.mark("llm_end")
        full_response = [final_text]

    except Exception as e:
        error_msg = f"[ERROR: Processing failed - {str(e)}]"
        await websocket.send_text(error_msg)
        print(traceback.format_exc())
    finally:
        await tts_queue.put(None)
        await worker

    return "".join(full_response)
