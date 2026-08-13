from importlib import import_module
from pathlib import Path
from typing import Any
import json

_tools: dict[str, Any] = {}

def discover_tools():
    tools_dir = Path(__file__).parent
    for path in tools_dir.glob("*.py"):
        if path.stem.startswith("_") or path.stem == "registry":
            continue
        mod = import_module(f".{path.stem}", __package__)
        _tools[mod.schema["function"]["name"]] = mod


def get_schemas() -> list[dict]:
    return [mod.schema for mod in _tools.values()]


async def execute(tool_name: str, arguments: dict | str) -> str:
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    mod = _tools.get(tool_name)
    if not mod:
        return f"Error: unknown tool '{tool_name}'"
    return await mod.execute(**arguments)


discover_tools()
