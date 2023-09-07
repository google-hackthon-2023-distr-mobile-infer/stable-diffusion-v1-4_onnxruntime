import requests

# base_url = 'http://localhost:5000'
# base_url = 'http://192.168.31.243:5000'
base_url = 'http://192.168.31.126:15000'

node_id = None
task_id = None
local_algo_type = 'tokenize'

# 注册节点
def regist_node(algo_part):
    global node_id
    r = requests.post(f'{base_url}/regist_node', json={
        "algo": "sd", # sd
        "algo_part": algo_part # tokenize encode unet decode
    })
    j = r.json()
    j = j['data']
    node_id = j['node_id']
    print(r.json())
    return node_id

last_algo_part_map = {
    'tokenize': None,
    'encode': 'tokenize',
    'unet': 'encode',
    'decode': 'unet',
}
# 获取待使用的输入
def get_task_input(algo_part_type):
    global task_id, node_id
    r = requests.post(f'{base_url}/receive_task', json={
        "node_id": node_id,
        "algo_part": algo_part_type
    })
    print(r.json())
    j = r.json()
    if j['errCode'] != 0:
        return None
    j = j['data']
    querys = j['querys']
    algo_part_output = j['algo_part_output']
    task_id = j['task_id']
    last_algo_part = last_algo_part_map[algo_part_type]
    if last_algo_part:
        return algo_part_output[last_algo_part]
    return querys

# 提交任务结果
def submit_task_result(result):
    global task_id
    r = requests.post(f'{base_url}/finish_task', json={
        'task_id': task_id,
        'data': result,
        "node_id": node_id,
        # "next_node_id": 2 # 已p2p转发时使用该字段
    })
    print(r.json())

def xxx(input):
    import time
    print('开始处理任务')
    time.sleep(5)
    print('完成任务')
    return input

if __name__ == '__main__':
    import time
    regist_node(local_algo_type)
    print('完成注册，node_id: ', node_id)
    while True:
        input = get_task_input(local_algo_type)
        if input:
            # 这里是算法的核心逻辑
            result = xxx(input)
            submit_task_result(input)
        else:
            print('没有任务，休息一秒')
            time.sleep(1)
