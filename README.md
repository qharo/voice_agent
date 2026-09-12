# Voice Agent

A real-time voice assistant with streaming speech-to-text, a tool-using language model, document-grounded answers, and streaming text-to-speech.

- Live demo: https://qharo-voice-agent.duckdns.org
- Technical overview: https://qharo-voice-agent.duckdns.org/about.html
- Repository: https://github.com/qharo/voice_agent

## Use it

- Hold `Space` to speak, then release to send.
- On touch screens, touch and hold the background for more than half a second, then release.
- Speak again during playback to interrupt the assistant.
- Upload a PDF or TXT file to ask questions about that document.
- Download the currently loaded document from Settings.
- Choose a preset for a guided customer-service or study-tutor experience.
- Customize the assistant prompt or clear the document from Settings.

The interface works with keyboards, mice, touch screens, and responsive mobile through desktop layouts.

## How it works

- Browser captures microphone audio and streams it over one WebSocket.
- Moonshine streaming STT produces live and final transcripts.
- Groq-hosted `openai/gpt-oss-20b` reasons over the transcript and conversation history.
- The agent loop can call tools automatically for up to five iterations.
- Uploaded documents are chunked, embedded, retrieved, and searched during relevant answers.
- Piper TTS streams the spoken reply back for gapless browser playback.
- Up to four sessions can process concurrently; additional users see live queue positions.
- Prometheus metrics support operational monitoring and Grafana dashboards.

## Tools

- `calculator`
- `dictionary`
- `exchange_rate`
- `get_weather`
- `search_document`
- `wikipedia`
- `world_time`

## Presets and documents

- Customer Service Assistant
  - Answers from the bundled company FAQ PDF.
- Study Tutor
  - Explains causal inference from the bundled TXT study guide.
- Custom uploads
  - PDF files up to 10 pages.
  - PDF and TXT files up to 20 MB.
  - TXT files must contain UTF-8 text.
- Retrieval
  - 1,000-character chunks.
  - 150-character overlap.
  - `BAAI/bge-small-en-v1.5` embeddings.
  - Top-four relevant passages.

## Repository

- `backend/`
  - FastAPI voice pipeline, STT/LLM/TTS integration, RAG, presets, tools, metrics, and queueing.
- `frontend/`
  - Responsive voice UI, About page, nginx proxy, and container definition.
- `helm/voice-agent/`
  - Kubernetes deployments, services, ingress, secrets, monitoring, and values.
- `diagrams/`
  - Source for the agent-loop diagram.
- `.github/workflows/`
  - Container builds and Kubernetes deployment automation.
- `docker-compose.yml`
  - Local backend/frontend service definition.
- `clusterissuer.yaml`
  - Certificate issuer configuration.

Main-branch changes build the backend and frontend container images and deploy them through Helm and Kubernetes.
