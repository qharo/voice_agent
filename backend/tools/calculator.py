import ast
import operator

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

ALLOWED_NODES = {ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, *OPERATORS.keys()}

schema = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the result. Supports +, -, *, /, **, %, //.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g. '2 + 2', '3 * 4.5', '2 ** 10')",
                }
            },
            "required": ["expression"],
        },
    },
}


async def execute(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return "Error: invalid syntax in expression"

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            return "Error: expression contains disallowed operations"

    try:
        result = eval(
            compile(tree, filename="", mode="eval"),
            {"__builtins__": {}},
            {},
        )
        return str(result)
    except Exception as e:
        return f"Error: {e}"
