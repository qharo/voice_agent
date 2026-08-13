import httpx

schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location. Returns temperature, conditions, and wind speed.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location name (e.g. 'London', 'New York', 'Tokyo')",
                }
            },
            "required": ["location"],
        },
    },
}

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


async def execute(location: str) -> str:
    async with httpx.AsyncClient() as client:
        geo_resp = await client.get(
            GEOCODING_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

    if not geo_data.get("results"):
        return f"Could not find location: {location}"

    result = geo_data["results"][0]
    lat, lon = result["latitude"], result["longitude"]
    name = result.get("name", location)
    country = result.get("country", "")

    async with httpx.AsyncClient() as client:
        weather_resp = await client.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "timezone": "auto",
            },
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

    current = weather_data.get("current_weather", {})
    temp = current.get("temperature", "?")
    windspeed = current.get("windspeed", "?")
    weather_code = current.get("weathercode", 0)

    conditions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
    }.get(weather_code, "Unknown")

    return f"Weather in {name}, {country}: {temp}°C, {conditions}, wind {windspeed} km/h"
