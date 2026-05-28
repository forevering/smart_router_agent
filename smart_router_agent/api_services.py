"""
模拟 API 服务模块

提供 3 个模拟的外部 API 服务：
1. WeatherProvider   - 天气预报查询
2. UserCalendar      - 用户日程查询
3. TravelConsultant  - 旅行顾问建议

每个服务通过 asyncio.sleep 模拟网络延迟，返回结构化数据。
API_REGISTRY 提供统一的服务注册表，供 api_executor 节点动态调用。
"""
import asyncio
from typing import Dict, Any


class WeatherProvider:
    """天气预报服务 —— 模拟根据地点和时间范围查询天气"""

    @staticmethod
    async def call(location: str = "未知", date_range: str = "next_week", **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(0.5)  # 模拟网络延迟
        # 模拟上海下周天气（2026-04-13 ~ 2026-04-17）
        forecasts = {
            "上海": [
                {"date": "2026-04-13", "weather": "多云", "temp_high": 22, "temp_low": 15, "rain_prob": "20%"},
                {"date": "2026-04-14", "weather": "小雨", "temp_high": 19, "temp_low": 14, "rain_prob": "75%"},
                {"date": "2026-04-15", "weather": "中雨", "temp_high": 17, "temp_low": 13, "rain_prob": "90%"},
                {"date": "2026-04-16", "weather": "阴转多云", "temp_high": 20, "temp_low": 14, "rain_prob": "30%"},
                {"date": "2026-04-17", "weather": "晴", "temp_high": 24, "temp_low": 16, "rain_prob": "5%"},
            ],
            "北京": [
                {"date": "2026-04-13", "weather": "晴", "temp_high": 26, "temp_low": 12, "rain_prob": "5%"},
                {"date": "2026-04-14", "weather": "晴转多云", "temp_high": 24, "temp_low": 11, "rain_prob": "10%"},
                {"date": "2026-04-15", "weather": "多云", "temp_high": 22, "temp_low": 10, "rain_prob": "15%"},
                {"date": "2026-04-16", "weather": "晴", "temp_high": 25, "temp_low": 12, "rain_prob": "5%"},
                {"date": "2026-04-17", "weather": "晴", "temp_high": 27, "temp_low": 14, "rain_prob": "5%"},
            ],
        }
        forecast = forecasts.get(location, forecasts["上海"])
        advisory = ""
        rainy_days = [f["date"] for f in forecast if int(f["rain_prob"].replace("%", "")) > 50]
        if rainy_days:
            advisory = f"{'、'.join(rainy_days)} 有较大降雨概率，建议携带雨具并关注出行安排"
        else:
            advisory = "预报期内天气总体良好，适合出行"

        return {
            "service": "WeatherProvider",
            "location": location,
            "period": date_range,
            "forecast": forecast,
            "advisory": advisory,
        }


class UserCalendar:
    """用户日程服务 —— 模拟根据用户 ID 查询日程安排"""

    @staticmethod
    async def call(user_id: str = "unknown", date_range: str = "next_week", **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(0.3)  # 模拟网络延迟
        return {
            "service": "UserCalendar",
            "user_id": user_id,
            "period": date_range,
            "events": [
                {
                    "date": "2026-04-13",
                    "time": "09:00-12:00",
                    "title": "客户拜访 - 张总",
                    "location": "上海市浦东新区陆家嘴",
                },
                {
                    "date": "2026-04-14",
                    "time": "14:00-16:00",
                    "title": "项目评审会议",
                    "location": "上海办公室",
                },
                {
                    "date": "2026-04-15",
                    "time": "10:00-11:30",
                    "title": "供应商洽谈",
                    "location": "上海市闵行区",
                },
                {
                    "date": "2026-04-16",
                    "time": "自由安排",
                    "title": "返程",
                    "location": "",
                },
            ],
        }


class TravelConsultant:
    """旅行顾问服务 —— 模拟根据目的地提供旅行建议"""

    @staticmethod
    async def call(destination: str = "未知", dates: str = "next_week", **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(0.4)  # 模拟网络延迟
        tips = {
            "上海": {
                "travel_tips": [
                    "上海4月中旬进入梅雨前期，建议随身携带便携雨伞",
                    "浦东机场到市区可乘磁悬浮 + 地铁2号线，全程约60分钟",
                    "推荐住宿区域：陆家嘴/人民广场附近，商务出行交通便利",
                    "出租车起步价14元，高峰时段建议使用地铁出行",
                ],
                "estimated_budget": "交通+住宿约 ¥3,000-5,000 / 3晚",
            },
            "北京": {
                "travel_tips": [
                    "北京4月天气干燥，温差较大，建议携带外套",
                    "首都机场到市区可乘机场快轨，全程约25分钟到东直门",
                    "推荐住宿区域：国贸/中关村附近，商务出行便利",
                ],
                "estimated_budget": "交通+住宿约 ¥2,500-4,500 / 3晚",
            },
        }
        info = tips.get(destination, tips["上海"])
        return {
            "service": "TravelConsultant",
            "destination": destination,
            **info,
        }


# ================================================================
# API 统一注册表 —— 供 Router 和 API Executor 使用
# ================================================================
API_REGISTRY: Dict[str, Dict[str, Any]] = {
    "WeatherProvider": {
        "callable": WeatherProvider.call,
        "description": "查询指定地点的天气预报，返回未来数天的天气、气温和降雨概率",
        "params": ["location", "date_range"],
    },
    "UserCalendar": {
        "callable": UserCalendar.call,
        "description": "查询用户的日程安排和会议信息",
        "params": ["user_id", "date_range"],
    },
    "TravelConsultant": {
        "callable": TravelConsultant.call,
        "description": "获取旅行目的地的实用建议、交通信息和预算估算",
        "params": ["destination", "dates"],
    },
}
