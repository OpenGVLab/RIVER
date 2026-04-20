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
# window_size = 8
resolution = 448
stride = 1  
sliding_fps = 1.0
long_memory_num = window_size
long_memory_num = int(sys.argv[4])
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
def get_asr(video_id:str,ts:float):
    try:
        data=json.load(open(f"PATH_TO/asr_extract/output_lvbench/{video_id}.m4a.json"))
    except:
        return ''
    lines = []
    for seg in data['segments']:
        if seg["end"] <= ts:                     # 保留整段
            lines.append(f"{seg['start']}s: {seg['text'].strip()}")
        elif seg["start"] < ts:                  # 保留部分（截断）
            lines.append(f"{seg['start']}s: {seg['text'].strip()}")
            break                                # 之后不再保留
    return "These are timestamps and speeches in the video"+" ".join(lines) if lines else ""

def get_middle_index(num_frames, num_segments):
    interval = num_frames // num_segments
    start = interval // 2
    return [start + interval * idx for idx in range(num_segments)]

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
anno_path = "PATH_TO/video_chat2_streaming/data/recall_362/online_recalling_"+sys.argv[1]+".json"
save_path = f"./output/longshort/{window_size}-{long_memory_num}/"+ sys.argv[1] +"/" 

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
        ext_len = max(ext_len,data_anno['q_num'])
print(ext_len)
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

            # Take "num_frame=16" frames forward from the current frame
            current_frame_indices = sorted([i for i in range(q_time, 0, -stride)[:window_size]]) 
            # Sample past frames before current frame at 1fps
            past_frame_indices = []
            current_frame_index = q_time - stride * window_size
            current_time = round(current_frame_index/fps, 2)
            num_segments = int(sliding_fps*current_time)
            if num_segments != 0:
                past_frame_indices = get_middle_index(current_frame_index, num_segments)

            frame_indices = past_frame_indices + current_frame_indices

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

            add = window_size - T_ % window_size  # TODO move to frame index 
            if add != window_size:
                tmp_list = []
                seg_size = T_ // add
                for i in range(add):
                    start = i * seg_size
                    end = T_ if i == (add - 1) else (i + 1) * seg_size
                    tmp_list.extend(video[start:end])
                    tmp_list.extend(video[end - 1].unsqueeze(0))
                video = torch.cat(tmp_list, dim=0)
                T_ += add
                print(f"Total Frame: {T_ - add} => {T_}")

            video = video.reshape(T_ // window_size, window_size, C, H, W).to("cuda:0")
            batch_size = 1
            num_batches = T_ // window_size
            all_video_emb_list = []
            short_video_emb_list = []
            short_video_emb = None
            ori_video_emb = None
            is_last = False
            for i in range(num_batches):
                if i==num_batches-1:
                    is_last = True
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, T_ // window_size)
                batch_video = video[start_idx:end_idx]      #([1, 16, 3, 448, 448])
                with torch.no_grad(), torch.cuda.amp.autocast(): 
                    ori_video_emb, video_emb = model.longshort_extract_feature(batch_video.squeeze())      #(16,256,4096) (16,9,4096)
                short_video_emb = video_emb.flatten(0,1).unsqueeze(0)   #(1,144,4096)    
                _, L, D = short_video_emb.shape 
                all_video_emb_list.extend(short_video_emb)
                torch.cuda.empty_cache()
            short_video_emb_list.append(ori_video_emb)

            ## compressing all_video_emb_list to long_memory_num video_emb_list
            long_video_emb_list = []
            if len(all_video_emb_list) > long_memory_num:
                # calculate similarity
                similar_list = []
                for idx in range(len(all_video_emb_list) -1):
                    scores = all_video_emb_list[idx] @ all_video_emb_list[idx+1].transpose(-1, -2) 
                    frame_silimar = torch.mean(scores)
                    similar_list.append(frame_silimar)

                # iteratively compress the queries
                while len(all_video_emb_list) > long_memory_num:
                    max_value = max(similar_list)
                    max_index = similar_list.index(max_value)
                    new_frame_feature = (all_video_emb_list[max_index].cpu() + all_video_emb_list[max_index+1].cpu()) / 2  
                    all_video_emb_list[max_index] = new_frame_feature.cuda()
                    del(all_video_emb_list[max_index+1])
                    del(similar_list[max_index])
                    if max_index > 0:
                        similar_list[max_index - 1] = torch.mean(all_video_emb_list[max_index - 1] @ all_video_emb_list[max_index].transpose(-1, -2))
                    if max_index < len(all_video_emb_list) - 1:
                        similar_list[max_index] = torch.mean(all_video_emb_list[max_index] @ all_video_emb_list[max_index + 1].transpose(-1, -2))

            # long_video_emb_list.append(torch.cat(all_video_emb_list, dim=0).view(1, -1, D))
            long_video_emb_list.append(torch.cat(all_video_emb_list, dim=0))
            
            long_emb, short_emb = long_video_emb_list[0], short_video_emb_list[0]
            video_emb = torch.cat((long_emb.reshape(-1, long_emb.shape[-1]),short_emb.reshape(-1, long_emb.shape[-1])),dim=0)
            print(long_emb.shape, short_emb.shape, video_emb.shape)  #torch.Size([16, 144, 4096]) torch.Size([16, 256, 4096])

            anno_answer = data_anno["correct_answer"]
            anno_question = data_anno["question"]
            anno_choices = data_anno["choices"]            
            long_time_prompt = long_time_prompt.format(current_time) + ''.join([f'Clip{i+1}: <image>\n' for i in range(long_memory_num)])
            # long_time_prompt = long_time_prompt.format(current_time) + ''.join([f'Clip{i+1}: <image>\n' for i in range(len(long_video_emb_list))])#######
            short_time_prompt = short_time_prompt.format(current_time, round(q_time/fps, 2)) + ''.join([f'Frame{i+1}: <image>\n' for i in range(window_size)])
            
            long_time_prompt = ''.join([f'Clip{i+1}: <image>\n' for i in range(long_memory_num)])
            short_time_prompt = ''.join([f'Frame{i+1}: <image>\n' for i in range(window_size)])
            
            question = f"Question: {anno_question}\n" + "Options:\n" + str(anno_choices)
            question = question.rstrip()
            prompt = system + question + question_prompt
            # prompt = system + get_asr(video_path.split('/')[-1].split('.')[0],data_anno["question_time"])+question + question_prompt
            # prompt = system + question
            prompt_list = [long_time_prompt, short_time_prompt, prompt]
            response = model.longshort_chat(tokenizer, video_emb, question = prompt_list, generation_config=generation_config, 
                                                num_patches_list=[1 for i in range(window_size)],
                                                num_clips_list=[1 for i in range(len(all_video_emb_list))])
            print(response)
            
            open_question = f"Question: {anno_question}\n" 
            open_question = open_question.rstrip()
            open_prompt = open_system + open_question + open_answer_prompt
            open_prompt_list = [long_time_prompt, short_time_prompt, open_prompt]
            open_response = model.longshort_chat(tokenizer, video_emb, question = open_prompt_list, generation_config=generation_config, 
                                                num_patches_list=[1 for i in range(window_size)],
                                                num_clips_list=[1 for i in range(len(all_video_emb_list))])
            print(open_response)

            print(f"pred_id: {response}", flush=True)
            print(f"pred: {open_response}", flush=True)
            print()
            gt_id = anno_answer
            gt = data_anno["choices"][ord(gt_id)-ord("A")].split(") ")[-1]
            new_item={"q_num":data_anno["q_num"],"gt_answer_id":gt_id,"pred_id":response,"gt_answer":gt,"pred":open_response,"question":anno_question}
            json.dump(new_item, f)
            f.write('\n')
            f.flush()
