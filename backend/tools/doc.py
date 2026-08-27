import rag

schema = {
    "type": "function",
    "function": {
        "name": "search_document",
        "description": (
            "Search the user's uploaded PDF for the most relevant passages. "
            "Use this whenever the user asks a question about a document they "
            "have uploaded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to look up in the document.",
                }
            },
            "required": ["query"],
        },
    },
}


async def execute(query: str) -> str:
    if not rag.has_document():
        return (
            "No document has been uploaded yet. Tell the user they need to "
            "upload a PDF before asking questions about it."
        )
    passages = rag.search(query)
    if not passages:
        return "The uploaded document did not contain any relevant passages for that question."
    return passages