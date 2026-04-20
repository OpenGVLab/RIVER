from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
import os
import io
from transformers import AutoConfig
from petrel_client.client import Client
client = Client('~/petreloss.conf', enable_mc=False)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
warnings.filterwarnings("ignore")

gpu_id = sys.argv[2]
num_tasks = int(sys.argv[3])
window_size = 16
resolution = 448
stride = 1  
sliding_fps = 1.0
long_memory_num = window_size
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.float().div(255.0)),
    transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
    transforms.Normalize(mean, std)
])

path = 'OpenGVLab/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=128, do_sample=False)

def get_middle_index(num_frames, num_segments):
    interval = num_frames // num_segments
    start = interval // 2
    return [start + interval * idx for idx in range(num_segments)]
def get_asr(video_id:str,ts:float):
    try:
        data=json.load(open(f"PATH_TO/asr_extract/output_lvbench/{video_id}.m4a.json"))
    except:
        return ''
    lines = []
    for seg in data['segments']:
        if seg["end"] <= ts-16:
            continue
        if seg["end"] <= ts:                     # 保留整段
            lines.append(f"{seg['start']}s: {seg['text'].strip()}")
        elif seg["start"] < ts:                  # 保留部分（截断）
            lines.append(f"{seg['start']}s: {seg['text'].strip()}")
            break                                # 之后不再保留
    return "These are timestamps and speeches in the video"+" ".join(lines) if lines else ""

long_time_prompt = "This contains a long memory of 0.0 to {:} seconds. \n"
short_time_prompt = "This contains a short clip sampled in {:} to {:} seconds. \n"
system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
question_prompt="\nOnly give the best option."
answer_prompt="Best option:("
open_system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, give the answer that accurately addresses the question.\n"
open_question_prompt=None
open_answer_prompt="Answer:"

local_path_map={
    'Vript-RR':      'YOUR_PATH/datasets/Vript-RR/RR_videos/',
    'LVBench':       'YOUR_PATH/datasets/LVBench/videos/',
    'LongVideoBench':'YOUR_PATH/datasets/LongVideoBench/videos/',
}

anno_path = "PATH_TO/video_chat2_streaming/data/online_recalling_"+sys.argv[1]+".json"
save_path = f"./output/longshort/asr-{window_size}/"+ sys.argv[1] +"/" 

json_file = json.load(open(anno_path,"r")) 
if not os.path.exists(save_path):
    os.makedirs(save_path)

ext_len = 0
save_json = save_path + gpu_id + ".jsonl"
if not os.path.isfile(save_json):
    with open(save_json, 'w') as f:
        pass
with open(save_json, "r") as f:
    for line in f:
        data_anno = json.loads(line)
        ext_len = max(ext_len,data_anno['q_num']+1)

    
with open(save_json,"a") as f:
    for idx in tqdm(range(len(json_file))[ext_len:]):
        data_anno = json_file[idx]
        if data_anno['q_num'] % num_tasks == int(gpu_id):
            video_name = data_anno['video_id']
            data_prefix = local_path_map[data_anno["video_source"]]
            video_path = os.path.join(data_prefix, video_name)+".mp4"
            anno_fps = data_anno['fps']
            try:
                if "s3://" in video_path:
                    video_bytes = client.get(video_path)
                    vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
                else:
                    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            except Exception as e:
                print(video_path, e)
                new_item={"q_num":data_anno["q_num"],"pred":"ERROR"}
                json.dump(new_item, f)
                f.write('\n')
                f.flush()
                continue
            real_num_frames = len(vr)
            fps = vr.get_avg_fps() 
        
            q_time = int(data_anno["question_time"]*fps)
            if q_time>real_num_frames-1:
                q_time=real_num_frames-1
                print("q_time>real_num_frames",q_time)
            else:
                print("q_time<real_num_frames",q_time)



            frame_indices = get_middle_index(q_time,window_size)

            # TODO  here ~~
            try:
                frames = vr.get_batch(frame_indices)
            except Exception as ee:
                print(video_path, e)
                new_item={"q_num":data_anno["q_num"],"pred":"ERROR"}
                json.dump(new_item, f)
                f.write('\n')
                f.flush()
                continue
            
            frames = torch.tensor(frames.asnumpy()).permute(0,3,1,2)
            video = transform(frames)                      
            print(video.shape)
            T_, C, H, W = video.shape

            # video = video.reshape(T_ // window_size, window_size, C, H, W).to("cuda:0")
            video = video.reshape(window_size, C, H, W).to("cuda:0").to(model.dtype)

            anno_answer = data_anno["correct_answer"]
            anno_question = data_anno["question"]
            anno_choices = data_anno["choices"]            
            # long_time_prompt = long_time_prompt.format(current_time) + ''.join([f'Clip{i+1}: <image>\n' for i in range(long_memory_num)])
            # long_time_prompt = long_time_prompt.format(current_time) + ''.join([f'Clip{i+1}: <image>\n' for i in range(len(long_video_emb_list))])#######
            # short_time_prompt = short_time_prompt.format(current_time, round(q_time/fps, 2)) + ''.join([f'Frame{i+1}: <image>\n' for i in range(window_size)])
            img_prompt=''.join([f'Clip{i+1}: <image>\n' for i in range(T_)])
            
            question = f"Question: {anno_question}\n" + "Options:\n" + str(anno_choices)
            question = question.rstrip()
            prompt = system + question + question_prompt
            prompt = question + question_prompt
            # prompt = get_asr(video_path.split('/')[-1].split('.')[0],data_anno["question_time"])+question + question_prompt
            prompt_list = img_prompt+prompt
            # prompt_list = [long_time_prompt, short_time_prompt, prompt]
            # response = model.chat(tokenizer, video, question = prompt_list, generation_config=generation_config, 
            #                                     num_patches_list=[1 for i in range(window_size)],
            #                                     num_clips_list=[1 for i in range(len(all_video_emb_list))])
            with torch.no_grad():
                response = model.chat(tokenizer, video, prompt_list, generation_config,
                                           num_patches_list=[1]*T_, history=None, return_history=False)
            # print(response)
            
            
            open_question = f"Question: {anno_question}\n" 
            open_question = open_question.rstrip()
            open_prompt = open_system + open_question + open_answer_prompt
            open_prompt_list = img_prompt+open_prompt
            # open_prompt_list = [long_time_prompt, short_time_prompt, open_prompt]
            # open_response = model.longshort_chat(tokenizer, video_emb, question = open_prompt_list, generation_config=generation_config, 
            #                                     num_patches_list=[1 for i in range(window_size)],
            #                                     num_clips_list=[1 for i in range(len(all_video_emb_list))])
            with torch.no_grad():
                open_response = model.chat(tokenizer, video, open_prompt_list, generation_config,
                                           num_patches_list=[1]*T_, history=None, return_history=False)
            # print(open_response)

            print(f"pred_id: {response}", flush=True)
            print(f"pred: {open_response}", flush=True)
            print()
            gt_id = anno_answer
            gt = data_anno["choices"][ord(gt_id)-ord("A")].split(") ")[-1]
            new_item={"q_num":data_anno["q_num"],"gt_answer_id":gt_id,"pred_id":response,"gt_answer":gt,"pred":open_response,"question":anno_question}
            json.dump(new_item, f)
            f.write('\n')
            f.flush()
