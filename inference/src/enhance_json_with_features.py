import json
import os


def extract_and_calculate_note_features(data):
    """
    提取并计算note特征。

    参数:
    data (list): 包含单个谱面内容的列表。

    返回:
    list: 包含提取和计算特征后的谱面内容的列表。
    """
    # 合并所有Notes并按时间排序
    all_notes = []
    for entry in data:
        for note in entry["Notes"]:
            note["time"] = entry["Time"]
            all_notes.append(note)
    all_notes.sort(key=lambda x: (x["time"], x["startPosition"]))

    time_to_notes = {}

    # 计算密度和多押数量
    for i, note in enumerate(all_notes):
        current_time = note["time"]
        start_time = current_time - 1.5
        note_count = 0

        # 以1.5秒作为note密度的计算窗口
        for j in range(i - 1, -1, -1):
            if all_notes[j]["time"] >= start_time:
                note_count += 1
            else:
                break

        note["density"] = note_count

        # 记录每个时间点的notes数量
        if current_time not in time_to_notes:
            time_to_notes[current_time] = 0
        time_to_notes[current_time] += 1

    # 判断扫键允许，代码效率亟待优化
    def generate_combinations(time_to_adjacent_notes):
        from itertools import product

        # 获取所有时间点
        times = list(time_to_adjacent_notes.keys())
        # 获取每个时间点对应的note列表
        note_lists = [time_to_adjacent_notes[time] for time in times]
        # 生成所有可能的组合
        all_combinations = list(product(*note_lists))

        return all_combinations

    for i, note in enumerate(all_notes):
        if "sweepAllowed" in note and note["sweepAllowed"]:
            sweep_allowed = True
        else:
            sweep_allowed = False

        sweep_start_time = note["time"] - 0.5
        adjacent_notes = []

        # 收集0.5秒内的note，以此作为扫键允许判断窗口期
        for j in range(i, -1, -1):
            if all_notes[j]["time"] >= sweep_start_time:
                adjacent_notes.append(all_notes[j])
            else:
                break

        adjacent_notes.reverse()

        # 将每个 adjacent_notes 里的 note 的 "sweepAllowed" 初始化为 false
        for adj_note in adjacent_notes:
            if "sweepAllowed" not in adj_note or not adj_note["sweepAllowed"]:
                adj_note["sweepAllowed"] = False

        # 根据 time 构造二维数组
        time_to_adjacent_notes = {}
        for adj_note in adjacent_notes:
            note_time = adj_note["time"]
            if note_time not in time_to_adjacent_notes:
                time_to_adjacent_notes[note_time] = []
            time_to_adjacent_notes[note_time].append(adj_note)

        # 生成所有可能的连接方式
        all_combinations = generate_combinations(time_to_adjacent_notes)

        # 在最深层循环中继续逻辑
        for combination in all_combinations:
            # 判断组合是否符合扫键条件
            for k in range(len(combination) - 2):
                diff1 = (
                    combination[k + 1]["startPosition"]
                    - combination[k]["startPosition"]
                ) % 8
                diff2 = (
                    combination[k + 2]["startPosition"]
                    - combination[k + 1]["startPosition"]
                ) % 8
                if diff1 in [1, 7] and diff2 in [1, 7] and (diff1 == diff2):
                    combination[k]["sweepAllowed"] = True
                    combination[k + 1]["sweepAllowed"] = True
                    combination[k + 2]["sweepAllowed"] = True

        # 确保不影响已被判定为 true 的 note
        note["sweepAllowed"] = sweep_allowed or note["sweepAllowed"]

    # 计算多押数量属性
    for note in all_notes:
        note["multiPressCount"] = time_to_notes[note["time"]]

    # 计算位移距离属性
    previous_position = None
    for note in all_notes:
        if previous_position is None:
            note["displacement"] = 0
        else:
            displacement = min(
                abs(note["startPosition"] - previous_position),
                8 - abs(note["startPosition"] - previous_position),
            )
            note["displacement"] = displacement
        previous_position = note["startPosition"]

    # 将计算后的Notes重新分配回原始数据结构
    for entry in data:
        entry["Notes"] = [note for note in all_notes if note["time"] == entry["Time"]]


def process_json_file(file_path, output_dir):
    with open(file_path, "r", encoding="utf-8") as jsonfile:
        data = json.load(jsonfile)

    # 提取 note 的特征并加工
    extract_and_calculate_note_features(data)

    # 保存处理后的JSON数据
    with open(output_dir, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)
