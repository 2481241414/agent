import requests
import json

url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": "Bearer sk-4fcc85e2509649198bdcafa4e985ce6e"  # 一般本地部署不校验token，可随便填
# }
headers = [
        ("Content-Type", "application/json"),
        ("Authorization", "Bearer sk-4fcc85e2509649198bdcafa4e985ce6e"),
        # 按你需要的顺序添加其他 headers
    ]
data = {
    "model": "qwen2.5-32b-instruct",  # 可用 vLLM 启动命令里的模型名称
    "messages": [
        {"role": "user", "content": "你是一个滴滴打车助手，你能根据用户的query去选择车型大类和车型小类，其中车型大类如[\"拼车\"，“特价”，“快速”，“舒适”，“六座”]等，车型小类如[\"特惠快车\",\"曹操出行\",\"滴滴快车\",\"特快出租车\",\"滴滴特快\",\"出租车\",\"滴滴专车\",\"滴滴豪华车\",\"六座专车\",\"极速拼车\",\"惊喜特价\",\"花小猪外接版\",\"花小猪正价版\"]等。\n\n                用户的query是：从贾西地铁站打车去雨花客厅，要六座车。\n                用户可以选择的车型大类为['经济', '出租', '六座', '拼车', '舒适出发', '特价', '特惠快车', '快速', '特快车', '特价拼车']\n                用户可以选择的车型小类为['滴滴快车', '滴滴豪华车', '花小猪外接版', '滴滴专车', '365约车', '出租车', '特惠快车', '特快出租车', '极速拼车', '六座专车', '聚优出租', '特快车', '特价拼车']\n\n                输出格式如下：\n                大类:[大类1,大类2]; 小类:[小类1,小类2]\n\n                特殊情况说明：\n                如果用户对车大类或者车小类没有明确要求，则大类和小类的字段都为空。 如query为：给我打个车去朱三家龙虾吃饭， 则输出大类:[]; 小类:[]\n                如果用户要求打所有车型，则输出所有的大类及所有的小类。 如query为：帮我打所有的车去火车站，则输出大类:[\"拼车\"，“特价”，“快速”，“舒适”，“六座”]; 小类:[\"特惠快车\",\"曹操出行\",\"滴滴快车\",\"特快出租车\",\"滴滴特快\",\"出租车\",\"滴滴专车\",\"滴滴豪华车\",\"六座专车\",\"极速拼车\",\"惊喜特价\",\"花小猪外接版\",\"花小猪正价版\"]\n                当用户要求的车在可选列表中没有时，则输出大类:[]; 小类:[]，如:郑州东站到国际会展中心打车，要求七座MPV车型或者有婴儿座椅的车。则输出大类:[]; 小类:[]\n\n                部分策略参考：\n                用户要求“拼车”时，选择大类和小类中含有“拼车”的车\n                用户要求“出租”时，选择大类和小类中含有“出租”或者“的士”的车\n                用户要求“豪华”时，选择大类和小类中含有“豪华”的车\n                用户要求“快车”或者“快速车”时，“快速”的车都属于快车，例如：特快车、特惠快车、特快出租车、滴滴快车、滴滴特快等\n                用户要求“特价车”或者“经济型”时，特价拼车、特价出租车、特惠快车、经济型都属于这类\n                六座专车与六座豪华属于不同车型，要求打六座专车时不能返回六座豪华车\n\n                注意：\n                不要根据车辆的价格来选择车型，一定只能根据用户的语义来选择车型\n                满足条件的车大类和车小类都要选上，不能漏选，比如“出租”、“的士”都属于“出租车”\n                只能从用户可以选择的车辆详情信息中选择，不能自己造一些大类和小类，输出的内容必须和可选择的车辆详情内的内容完全一致"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
}

# response = requests.post(url, headers=headers, json=data)
# print(response.content)
# print(response.json())

import asyncio
import aiohttp

async def req():
    async with aiohttp.ClientSession() as session:
        result_text = None
        try:
            result = await session.post(url, headers=headers, data=json.dumps(data))
            print(result)
            result_text = await result.text()
            print(result_text)
        except Exception as e:
            raise (e)
        return result
asyncio.run(req())
