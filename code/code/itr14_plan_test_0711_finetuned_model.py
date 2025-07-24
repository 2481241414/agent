import os
import dashscope
import time
import pandas as pd
import numpy as np
import csv
import requests

# prompt1 = """
# # 背景
# 你是一个具备用户指令理解与指令句式转换能力的agent。
# 当用户给你一个app的具体指令任务的时候，你能够理解用户自然的、口语化的、倒装的语音指令,并将其转换为一种规范化的书面指令句式。

# # 要求
# 你需要遵从如下规范：

# ## 任务输出规范
# 单个任务的描述应该由任务类型与任务内容组成。
# 任务类型应该用#进行间隔，并从以下列表中筛选出最符合的一项：[订单，发票，购物车，客服，签到，收藏，搜索，物流]
# 任务内容应该遵循如下两类的句式规范中的一个并将涉及到应用名词的部分用$进行间隔：
#     在<应用名词>中搜索<搜索内容/短句>；
#     在<应用名词>中<动词-打开、查看、管理、添加、使用、清理等><名词/短句>；
# 应用名词需要将口语化或别称转换为标准叫法，以下面这个列表为准：[抖音，抖音极速版，快手，快手极速版，拼多多，淘宝，京东，天猫，闲鱼，抖音火山版，阿里巴巴，唯品会，得物，转转]
# 标准叫法未出现在列表中的应用名词请按照常识进行判断与转换
# 示例：
#     #搜索#在$淘宝$中查看夏装

# # 任务类型描述
# - 订单: 在应用程序中打开或查看与订单相关的操作，例如查看订单、查看待取件订单、查看发布的商品、或搜索指定商品的订单、查看评价等
# - 发票：在应用程序中打开或查看与发票相关的操作，例如打开发票中心、查看申请中发票等
# - 购物车: 在应用程序中打开购物车/采购车/想要的东西，或查看购物车中某类型的商品，例如降价、有货等，还可以在购物车中搜索指定的物品
# - 客服：在应用程序中联系客服
# - 签到：在应用程序中通过签到领取积分、金币、红包等奖励。
# - 收藏：在应用程序中打开收藏夹/喜爱/想要/关注的商品/关注的店铺，或在收藏夹中搜索商品或店铺。
# - 搜索：在应用程序中进行搜索，例如搜索商品、店铺，并且可按综合、销量、价格等排序，还可查看或清除搜索历史或打开相机。
# - 物流：在应用程序中查看包裹的物流信息，例如查看商城订单物流、查看已签收物流、联系客服查看待取件物流等
    
# # 转换要求
# 1、大致原则：尽可能保证用户对于应用操控的动作和对象不变。
# 2、口语化转标准术语: 将日常口语表达（如'帮我搞一下'、'瞅瞅'）转换为手机主流App（如抖音、美团、微信、百度等）中常见的功能名称、操作指令或页面描述。
# 3、去除冗余信息: 识别并剔除用户指令中与核心意图无关的冗余内容，例如：
#     AI助手称呼 (如 'YOYO', '悠悠', '优优')；语气词、感叹词 (如 '难道', '竟然', '麻烦', '立刻', '吧', '呢', '啊', '啦')；
#     不必要的修饰或评价 (如 '这功能太方便创作灵感了！')；
#     其他重复或无意图的表达。
# 4、拆分原则：
# - 只有当任务之间明显独立且无法合并时，才拆分成多个子任务。
# - 如果多个操作可以合成一步完成，则不拆分，保持任务的整体性。
# - 避免将自然合成的操作拆成多个步骤，例如“打开淘宝购物车”不拆成“先打开淘宝”再“打开购物车”。
# - 对于查询类任务，直接完成最终目标，不拆分成中间步骤。
# - 不要为了拆分而衍生出原本语义中不存在的操作，只说明明确提到的任务，不要猜测用户的意图
# - 若确实需要拆分任务，每个任务都必须有对应的8个任务类型中的一个，否则不拆分
# 5、任务类型只能从给出的8个当中选择，不可自己创造新的任务类型。

# # 输出格式要求：
# <Plan>任务一描述<br>任务二描述<br>任务三描述</Plan><Finish>
# 其中当用户任务是多个动作，或者是多个app相关操作，则输出多任务，每个任务之间用<br>间隔,并且每个任务都要按照任务输出规范，包含任务类型和应用名词，输出结果无需再以句号结尾。
# 请严格按照格式输出，不要输出多余内容。

# # 注意
# 1. 严格遵守输出格式要求
# 2. 不可以自己创造新的任务类型或应用名词，必须选择我给出的列表中的一项
#   - 任务列表：[订单，发票，购物车，客服，签到，收藏，搜索，物流]
#   - 应用名词列表：[抖音，抖音极速版，快手，快手极速版，拼多多，淘宝，京东，天猫，闲鱼，抖音火山版，阿里巴巴，唯品会，得物，转转]
# 3. 只有任务之间明显独立，或在不同app间操作时，才进行任务拆分，否则进行合为一个任务输出
# 4. 如果不是多个任务，不要使用<br>间隔，使用逗号或并连接即可
# 5. 注意区分任务和筛选条件的区别，不要讲筛选条件作为单独任务输出，例如排序条件、订单类型、物流类型等
# 6. 一个任务只能出现一个#任务类型#和一个$应用名词$，不可以出现多个

# # 输入问题
# 下面是用户的问题，请根据以上原则进行拆分和转写
# """

prompt1 = """
    你是一个擅长指令拆分改写和信息提取的智能体
    你的任务是基于用户输入的query拆分改写并提取出指令及appname、指令类别
    指令类别需要在<订单\发票\购物车\客服\签到\收藏\搜索\物流\启动>中选择
    appname需要在<抖音\抖音极速版\快手\快手极速版\拼多多\淘宝\京东\天猫\闲鱼\抖音火山版\阿里巴巴\唯品会\得物\转转>中选择
        - 如果用户的appname不在上述列表中，请按照常识进行判断与转换，例如1688转换为阿里巴巴，毒转换为得物

    拆分原则为：
    - 只有当任务之间明显独立且无法合并时，才需要拆分为多个子任务
    - 若一个任务的执行过程包含另一指令，则不需要拆分

    你需要把每个任务转换为规范化的指令句式：
        在<appname>中<打开\查看\搜索\管理\添加\使用\清理等><名词\短句>；
        打开<appname>

    输出格式使用csv格式，以表格形式输出转换后的规范化书面指令句式：
    taskId,task,category,app_name
    1,在拼多多中打开购物车,购物车,拼多多
    2,打开京东,启动,京东

    请严格按照要求格式输出，不要输出多余内容。以下是用户的问题，请根据以上原则进行拆分改写和信息提取
"""

url = "http://localhost:6008/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer xxx"  # 一般本地部署不校验token，可随便填
}


output_xlsx_name = "home/workspace/buy_plan_test/itr14_plan_test_resule_0711_qwen3_32b.csv"

data = pd.read_excel("home/workspace/buy_plan_test/itr14_plan_test_0711_qwen3-32b-finetuned.xlsx")
category = np.array(data.iloc[:,0])
app_name = np.array(data.iloc[:,1])
instruction = np.array(data.iloc[:,2])
query = np.array(data.iloc[:,3])

if not os.path.exists(f"{output_xlsx_name}"):
    with open(f"{output_xlsx_name}", 'a',encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['输入query','输入query对应的instruction','输入appname','输入大类','输出','提取的appname','提取的大类','appname匹配','大类匹配','时延'])

for i in range(query.shape[0]):
    data = {
        "model": "qwen3-32b-plan",  # 可用 vLLM 启动命令里的模型名称
        "messages": [
            {"role": "system", "content": prompt1},
            {"role": "user", "content": query[i]}
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    total_delay = time.time() - start_time

    lines = data.split('\n')

    # 表头字段
    headers = lines[0].split(',')

    # 数据行
    values = lines[1].split(',')

    # 转成字典
    record = dict(zip(headers, values))

    # 获取 category 和 app_name
    category_current = record.get('category')
    app_name_curent = record.get('app_name')

    if category_current == category[i]:
        category_right = 1
    else:
        category_right = 0
    
    if  app_name_curent == app_name[i]:
        app_name_right = 1
    else:
        app_name_right = 0

    print("category:", category)
    print("app_name:", app_name)
    

    with open(f"{output_xlsx_name}", 'a',encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([query[i], instruction[i], app_name[i], category[i], response, app_name_curent, category_current, app_name_right, category_right, total_delay])

    print(f'idx:{i}  输入:{query[i]}  输出:{response}')
    print(f'延时:{total_delay}')