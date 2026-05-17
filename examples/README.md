# Examples

Each example is a self-contained script you can run directly.  
All examples use **mock providers** — no API keys or external services required.

| File | What it shows |
|---|---|
| `01_basic_llm.py` | Simple chat completion + sync wrapper |
| `02_streaming.py` | Token-by-token streaming from an LLM |
| `03_image_generation.py` | Text-to-image with custom size and seed |
| `04_video_generation.py` | Text-to-video with duration and fps |
| `05_embeddings.py` | Batch embedding + cosine similarity |
| `06_cost_tracking.py` | Session-level cost and token accumulation |
| `07_custom_middleware.py` | Writing and injecting your own middleware |
| `08_registry.py` | Config-driven provider selection via registries |
| `09_multimodal_pipeline.py` | LLM → image → embedding in one workflow |
| `providers/` | Reference adapter skeletons (OpenAI, Anthropic, Replicate, Runway) |

## Running

```bash
cd conduit-sdk/  # repo folder
pip install -e ".[dev]"

# Run any example
python examples/01_basic_llm.py
```
