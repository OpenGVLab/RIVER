import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

gap_dict = {'second': 15, 'short': 30, 'middle': 300, 'long': 1800}
marks = ['second', 'short', 'middle', 'long']
window_size = 16

template_json = "/mnt/petrelfs/share_data/shiyansong/datasets/online_final_new/online_awaiting_{}.json"

def get_score2(task, time_error):
    if time_error < -gap_dict[task] / 2:
        return 0
    if abs(time_error) <= gap_dict[task] / 2:
        return 1
    else:
        return max(1 - (abs(time_error) - gap_dict[task] / 2) / (gap_dict[task] / 2), 0.0)

for anno_path_template in [
    "PATH_TO/videochat2hd_online_awaiting_{}.json.win16",
    "PATH_TO/internvl2_5_online_awaiting_{}.json.win16",
    "PATH_TO/llavavideo_online_awaiting_{}.json.win16",
    "PATH_TO/vcflash_online_awaiting_{}.json.win16",
]:
    model_name = anno_path_template.format(0).split('/')[-1].split('_')[0]
    print(f'{model_name:>12}', end=' ')
    
    abs_time_error_score = 0.0
    can_answer_all = 0
    
    for mark in marks:
        template_data = load_json(template_json.format(mark))
        qid = [data["question_id"] for data in template_data]
        pred_contents = load_json(anno_path_template.format(mark))
        
        for item in pred_contents:
            if item["question_id"] not in qid:
                continue
            
            time_ref = item["time_reference"][0] if isinstance(item["time_reference"], list) else item["time_reference"]
            
            if "break_window_idx" in item:
                can_answer_all += 1
                pred_time = (item["break_window_idx"] + 1) * window_size
                abs_time_error_score += get_score2(mark, pred_time - time_ref)
            else:
                # 未定位到窗口，用视频结束时间作为预测
                can_answer_all += 1
                abs_time_error_score += get_score2(mark, item["duration_sec"] - time_ref)
    
    print(f"Loc_score: {round(abs_time_error_score / can_answer_all, 4):5}")

