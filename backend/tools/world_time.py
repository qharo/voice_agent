import httpx

schema = {
    "type": "function",
    "function": {
        "name": "world_time",
        "description": "Get the current local time in a city or location. Use for questions like 'what time is it in Tokyo' or 'what's the time in London now'.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location name (e.g. 'Tokyo', 'London', 'New York')",
                }
            },
            "required": ["location"],
        },
    },
}

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
TIME_URL = "https://timeapi.io/api/Time/current/zone"

_DT_CODES = {
    "Sunday": "Sunday",
    "Monday": "Monday",
    "Tuesday": "Tuesday",
    "Wednesday": "Wednesday",
    "Thursday": "Thursday",
    "Friday": "Friday",
    "Saturday": "Saturday",
}


def _fmt_12(hour: int, minute: int) -> str:
    period = "AM" if hour < 12 else "PM"
    h12 = hour % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:{minute:02d} {period}"


async def execute(location: str) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        geo_resp = await client.get(
            GEOCODING_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

    if not geo_data.get("results"):
        return f"Could not find the location: {location}"

    result = geo_data["results"][0]
    name = result.get("name", location)
    country = result.get("country", "")
    timezone = result.get("timezone")

    if not timezone:
        return f"Could not determine the timezone for {name}."

    async with httpx.AsyncClient(timeout=15) as client2:
        time_resp = await client2.get(TIME_URL, params={"timeZone": timezone})

    if time_resp.status_code != 200:
        return f"Could not fetch the current time for {name}."

    data = time_resp.json()
    hour = data.get("hour")
    minute = data.get("minute")
    day_of_week = _DT_CODES.get(data.get("dayOfWeek"), "")

    if hour is None or minute is None:
        return f"Could not parse the time for {name}."

    loc = name if not country else f"{name}, {country}"
    when = f" on {day_of_week}" if day_of_week else ""
    return f"The time in {loc} is {_fmt_12(hour, minute)}{when} ({timezone})."
