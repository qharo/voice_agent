import httpx

schema = {
    "type": "function",
    "function": {
        "name": "exchange_rate",
        "description": "Convert an amount between two currencies using current exchange rates. Use for questions like 'how much is 100 dollars in euros' or 'what's the usd to jpy rate'.",
        "parameters": {
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "The source currency ISO code (e.g. 'USD', 'EUR', 'JPY', 'GBP')",
                },
                "to": {
                    "type": "string",
                    "description": "The target currency ISO code (e.g. 'USD', 'EUR', 'JPY', 'GBP')",
                },
                "amount": {
                    "type": "number",
                    "description": "The amount to convert. Defaults to 1 if not given (just the rate).",
                },
            },
            "required": ["from", "to"],
        },
    },
}

API_URL = "https://api.frankfurter.dev/v1/latest"


def _currency_full(code: str) -> str:
    names = {
        "USD": "US dollars",
        "EUR": "euros",
        "GBP": "British pounds",
        "JPY": "Japanese yen",
        "CHF": "Swiss francs",
        "CAD": "Canadian dollars",
        "AUD": "Australian dollars",
        "INR": "Indian rupees",
        "CNY": "Chinese yuan",
    }
    return names.get(code.upper(), code.upper())


def _fmt(v: float) -> str:
    if v >= 1:
        s = f"{v:,.2f}".replace(",", "")
    else:
        s = f"{v:.4f}"
    s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


async def execute(**kwargs) -> str:
    from_code = str(kwargs.get("from", "")).upper()
    to_code = str(kwargs.get("to", "")).upper()
    amount = kwargs.get("amount")
    qty = 1.0 if amount is None else amount

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            API_URL,
            params={"amount": qty, "from": from_code, "to": to_code},
        )

    if resp.status_code in (400, 404, 422):
        return f"Sorry, one of those currencies isn't supported: {from_code} to {to_code}."

    resp.raise_for_status()
    data = resp.json()

    if to_code not in data.get("rates", {}):
        return f"Sorry, {to_code} isn't supported by the rate service."

    value = data["rates"][to_code]
    symbol = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
    }.get(from_code, "")

    if symbol:
        return (
            f"{symbol}{_fmt(qty)} is {_fmt(value)} "
            f"{_currency_full(to_code)}."
        )
    return (
        f"{_fmt(qty)} {_currency_full(from_code)} is {_fmt(value)} "
        f"{_currency_full(to_code)}."
    )
