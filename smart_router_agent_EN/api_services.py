"""
Mock API Services Module

Provides 3 simulated external API services:
1. WeatherProvider   - Weather forecast query
2. UserCalendar      - User schedule query
3. TravelConsultant  - Travel advisor suggestions

Each service simulates network latency via asyncio.sleep and returns structured data.
API_REGISTRY provides a unified service registry for dynamic invocation by the api_executor node.
"""
import asyncio
from typing import Dict, Any


class WeatherProvider:
    """Weather forecast service — simulates querying weather by location and time range"""

    @staticmethod
    async def call(location: str = "unknown", date_range: str = "next_week", **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(0.5)  # Simulate network latency
        # Simulate Shanghai next-week weather (2026-04-13 ~ 2026-04-17)
        forecasts = {
            "Shanghai": [
                {"date": "2026-04-13", "weather": "Cloudy", "temp_high": 22, "temp_low": 15, "rain_prob": "20%"},
                {"date": "2026-04-14", "weather": "Light rain", "temp_high": 19, "temp_low": 14, "rain_prob": "75%"},
                {"date": "2026-04-15", "weather": "Moderate rain", "temp_high": 17, "temp_low": 13, "rain_prob": "90%"},
                {"date": "2026-04-16", "weather": "Overcast to cloudy", "temp_high": 20, "temp_low": 14, "rain_prob": "30%"},
                {"date": "2026-04-17", "weather": "Sunny", "temp_high": 24, "temp_low": 16, "rain_prob": "5%"},
            ],
            "Beijing": [
                {"date": "2026-04-13", "weather": "Sunny", "temp_high": 26, "temp_low": 12, "rain_prob": "5%"},
                {"date": "2026-04-14", "weather": "Sunny to cloudy", "temp_high": 24, "temp_low": 11, "rain_prob": "10%"},
                {"date": "2026-04-15", "weather": "Cloudy", "temp_high": 22, "temp_low": 10, "rain_prob": "15%"},
                {"date": "2026-04-16", "weather": "Sunny", "temp_high": 25, "temp_low": 12, "rain_prob": "5%"},
                {"date": "2026-04-17", "weather": "Sunny", "temp_high": 27, "temp_low": 14, "rain_prob": "5%"},
            ],
        }
        forecast = forecasts.get(location, forecasts["Shanghai"])
        advisory = ""
        rainy_days = [f["date"] for f in forecast if int(f["rain_prob"].replace("%", "")) > 50]
        if rainy_days:
            advisory = f"High probability of rain on {', '.join(rainy_days)}; recommend bringing an umbrella and planning travel accordingly"
        else:
            advisory = "Weather is generally good during the forecast period, suitable for travel"

        return {
            "service": "WeatherProvider",
            "location": location,
            "period": date_range,
            "forecast": forecast,
            "advisory": advisory,
        }


class UserCalendar:
    """User schedule service — simulates querying user schedules by user ID"""

    @staticmethod
    async def call(user_id: str = "unknown", date_range: str = "next_week", **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(0.3)  # Simulate network latency
        return {
            "service": "UserCalendar",
            "user_id": user_id,
            "period": date_range,
            "events": [
                {
                    "date": "2026-04-13",
                    "time": "09:00-12:00",
                    "title": "Client visit - Mr. Zhang",
                    "location": "Lujiazui, Pudong New Area, Shanghai",
                },
                {
                    "date": "2026-04-14",
                    "time": "14:00-16:00",
                    "title": "Project review meeting",
                    "location": "Shanghai office",
                },
                {
                    "date": "2026-04-15",
                    "time": "10:00-11:30",
                    "title": "Supplier negotiation",
                    "location": "Minhang District, Shanghai",
                },
                {
                    "date": "2026-04-16",
                    "time": "Flexible",
                    "title": "Return trip",
                    "location": "",
                },
            ],
        }


class TravelConsultant:
    """Travel advisor service — simulates providing travel suggestions by destination"""

    @staticmethod
    async def call(destination: str = "unknown", dates: str = "next_week", **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(0.4)  # Simulate network latency
        tips = {
            "Shanghai": {
                "travel_tips": [
                    "Shanghai enters the pre-plum rain period in mid-April; recommend carrying a portable umbrella",
                    "From Pudong Airport to downtown, take the Maglev + Metro Line 2, about 60 minutes total",
                    "Recommended accommodation areas: Lujiazui/People's Square, convenient for business travel",
                    "Taxi starting fare is 14 RMB; during rush hours, metro is recommended",
                ],
                "estimated_budget": "Transportation + accommodation approx. \u00a53,000-5,000 / 3 nights",
            },
            "Beijing": {
                "travel_tips": [
                    "Beijing weather in April is dry with large temperature differences; recommend bringing a jacket",
                    "From Capital Airport to downtown, take the Airport Express, about 25 minutes to Dongzhimen",
                    "Recommended accommodation areas: Guomao/Zhongguancun, convenient for business travel",
                ],
                "estimated_budget": "Transportation + accommodation approx. \u00a52,500-4,500 / 3 nights",
            },
        }
        info = tips.get(destination, tips["Shanghai"])
        return {
            "service": "TravelConsultant",
            "destination": destination,
            **info,
        }


# ================================================================
# Unified API Registry — used by Router and API Executor
# ================================================================
API_REGISTRY: Dict[str, Dict[str, Any]] = {
    "WeatherProvider": {
        "callable": WeatherProvider.call,
        "description": "Query weather forecast for a specified location, returning weather, temperature, and rain probability for the coming days",
        "params": ["location", "date_range"],
    },
    "UserCalendar": {
        "callable": UserCalendar.call,
        "description": "Query user's schedule and meeting information",
        "params": ["user_id", "date_range"],
    },
    "TravelConsultant": {
        "callable": TravelConsultant.call,
        "description": "Get practical travel tips, transportation info, and budget estimates for a destination",
        "params": ["destination", "dates"],
    },
}
