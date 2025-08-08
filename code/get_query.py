import json

input_file = "/home/workspace/lgq/data/20250613/v1.0/全新训练数据20250613_map.json"
output_file = "/home/workspace/lgq/data/20250613/v1.0/全新训练数据20250613_map_query.txt"


query_set = []
with open(input_file) as fin:
    for line in json.loads(fin.read()):
        query, types = line['input'].split('\n')
        query = query.split('用户的query是：')[1]
        query_set.append(query)

query_set = sorted(query_set, key= lambda x: len(x), reverse=False)
print('\n'.join(query_set))
with open(output_file, 'w') as fout:
    fout.write('\n'.join(query_set))
