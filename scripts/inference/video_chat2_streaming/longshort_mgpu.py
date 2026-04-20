import io
import os
from models import VideoChat2_it_hd_mistral
from utils.easydict import EasyDict
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython.display import Video, HTML
from IPython import embed
from peft import get_peft_model, LoraConfig, TaskType
import copy
import json
from collections import OrderedDict
from tqdm import tqdm
import decord
import time
decord.bridge.set_bridge("torch")
import math
import random
from scipy.stats import entropy
len9_max_ent = entropy([1/9 for i in range(9)])
from sklearn.preprocessing import normalize
import pysubs2
import re
from utils.config import Config
config_file = "configs/config_mistral_hd.json"
cfg = Config.from_file(config_file)
from petrel_client.client import Client
from decord import VideoReader, cpu
client = Client('~/petreloss.conf', enable_mc=False)
from dataset.hd_utils import HD_transform_padding, HD_transform_no_padding
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
import time
import sys
debug = sys.argv[1] == 'debug'
gpu_id = sys.argv[2]
num_tasks = int(sys.argv[3])

## build model and load ckpt
if not debug:
    # load stage2 model
    cfg.model.vision_encoder.num_frames = 4
    model = VideoChat2_it_hd_mistral(config=cfg.model)

    # add lora to run stage3 model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
    )
    model.mistral_model = get_peft_model(model.mistral_model, peft_config)

    sd_path = {"PATH_TO/Ask-Anything/video_chat2/videochat2_hd_mistral_7b_stage4.pth":1.0}


    new_sd = OrderedDict()

    for idx, (path, weight) in enumerate(sd_path.items()):
        sd = torch.load(path, 'cpu')
        if 'model' in sd.keys():
            sd = sd['model']
        if idx == 0:
            for k, v in sd.items():
                new_sd[k] = v * weight
        else:
            for k, v in sd.items():
                new_sd[k] += v * weight


    msg = model.load_state_dict(new_sd, strict=False)
    print(msg)

    model = model.to(torch.device(cfg.device))
    model = model.eval()

def convert_mm_ss_to_seconds(time_str):
    # 将时间字符串分割成分钟和秒
    start_str, end_str = time_str.split('-')
    minutes, seconds = map(int, end_str.split(':'))
    end_seconds = minutes * 60 + seconds
    return end_seconds

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=True):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')

    # embed()
    # exit()
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."

    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]  # 3 * [1, n, 4096]
    # seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]  # 3 * [1, n, 4096]
    mixed_embs = torch.cat(mixed_embs, dim=1)  # [1, 965, 4096]
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
    repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([29871, 2]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)

    # embed()
    # exit()
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
    # output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

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


def longshort_memory_streaming_breakpoint_infer(qa_mode = "breakpoint"):
    assert qa_mode == "breakpoint", f"Not implement \"global\" qa!!!"
    
    # window_size = 16
    window_size = int(sys.argv[4])
    resolution = 224
    hd_num = 1#6
    stride = 1  # 2,5,10
    sliding_fps = 1.0
    # long_memory_num = window_size
    long_memory_num = int(sys.argv[5])
    model.add_global=False

    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*window_size, cur_frame=window_size)

    if not debug:
        # print(f"Eval position embedding lens: {new_pos_emb.shape}, model training pos_embed lens: {model.vision_encoder.encoder.pos_embed.shape}")
        model.vision_encoder.encoder.pos_embed = new_pos_emb
        model.local_size = resolution

    local_path_map={
        'Vript-RR':      'YOUR_PATH/datasets/Vript-RR/RR_videos/',
        'LVBench':       'YOUR_PATH/datasets/LVBench/videos/',
        'LongVideoBench':'YOUR_PATH/datasets/LongVideoBench/videos/',
    }
    anno_path = "PATH_TO/video_chat2_streaming/data/online_recalling_"+sys.argv[1]+".json"
    save_path = f"./output/longshort/"+ sys.argv[1] +"/"  
    print('save-path',save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    json_file = json.load(open(anno_path,"r"))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])
    hd_transform = HD_transform_no_padding
    system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
    question_prompt="\nOnly give the best option."
    answer_prompt="Best option:("
    long_term_prompt = "This contains a long memory of 0.0 to {:} seconds. "
    short_time_prompt = "This contains a short clip sampled in {:} to {:} seconds. "

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
                # video_name = "{:08d}".format(idx)
                data_prefix = local_path_map[data_anno["video_source"]]
                video_path = os.path.join(data_prefix, video_name)+".mp4"

                anno_fps = data_anno['fps']
                # anno_num_frame = data_anno['info']['num_frame']
                # qalist = data_anno[qa_mode]
                # qalist = data_anno["qa"]

                try:
                    if "s3://" in video_path:
                        video_bytes = client.get(video_path)
                        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
                    else:
                        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                except Exception as e:
                    print(video_path, e)
                    new_item={"question_id":data_anno["question_id"],"answer":data_anno["correct_answer"],"pred":"ERROR"}
                    json.dump(new_item, f)
                    f.write('\n')
                    f.flush()
                    continue

                real_num_frames = len(vr)
                fps = vr.get_avg_fps() 

                # if real_num_frames <= (anno_num_frame+1):
                #     for i in range(len(qalist)):
                #         qalist[i]["time"]=int(qalist[i]["time"]/anno_num_frame*real_num_frames-1) 
                #         # make sure get frame_indices success
                #         if qalist[i]["time"] <= (window_size*stride + 1):
                #             qalist[i]["time"] = (window_size*stride + window_size//2)
                
                # for idx in range(len(qalist)):
                start_time=time.time()
                print(f"-------------------",data_anno["question_id"],flush=True)
                chat = EasyDict({
                    "system": system,
                    "roles": ("Human", "Assistant"),
                    "messages": [],
                    "sep": "###"
                })

                # q_time = qalist[idx]["time"]
                if "question_time" in data_anno:
                    q_time = int(data_anno["question_time"]*fps)
                else:
                    # q_time = int((convert_mm_ss_to_seconds(data_anno["time_reference"])+int(sys.argv[1]))*fps)
                    print('!!!!!!!!!!!!!!!!! line 374')

                if q_time>real_num_frames-1:
                    q_time=real_num_frames-1
                    print("q_time>real_num_frames",q_time)
                else:
                    print("q_time<real_num_frames",q_time)

                anno_answer = data_anno["correct_answer"]
                anno_question = data_anno["question"]
                anno_choices = data_anno["choices"]

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
                chat.messages.append([chat.roles[0], long_term_prompt.format(current_time)+f"<Video><VideoHere></Video>\n"])
                chat.messages.append([chat.roles[0], short_time_prompt.format(current_time, round(q_time/fps, 2))+f"<Video><VideoHere></Video>\n"])  # ugly TODO
                # chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
                # chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])  # ugly TODO

                # TODO  here ~~
                try:
                    frames = vr.get_batch(frame_indices)
                except Exception as ee:
                    print(video_path, e)
                    new_item={"question_id":data_anno["question_id"],"answer":data_anno["correct_answer"],"pred":"ERROR"}
                    json.dump(new_item, f)
                    f.write('\n')
                    f.flush()
                    continue
                frames = frames.permute(0, 3, 1, 2)
                # print(type(frames), frames.shape, frames.dtype)
                # breakpoint()
                frames = hd_transform(frames.float(), image_size=resolution, hd_num=hd_num)
                video = transform(frames) # [32, 3, 224, 448]
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
                for i in range(num_batches):  # [T_ // window_size, 16, 3, 224, 448]
                    # if long_memory_num==0 and i<num_batches-1:
                    #     continue
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, T_ // window_size)
                    batch_video = video[start_idx:end_idx]  # [1, 16, 3, 224, 448]
                    try:
                        with torch.no_grad():
                            # video_emb, _, _ = model.encode_img(batch_video, [system], only_global=True)#!
                            video_emb, _, _ = model.encode_img(batch_video, [system])
                    except Exception as eee:
                        print(video_path, eee)
                        continue
                    short_video_emb = video_emb[0] # [1, 96*hd-num, 4096]
                    _, L, D = short_video_emb.shape
                    all_video_emb_list.extend(short_video_emb) # 30 * [96, 4096] windows*[q-num*hd-num, token-dim]
                    torch.cuda.empty_cache()
                short_video_emb_list.append(short_video_emb) #只保留最后一次的short-embed

                ## compressing all_video_emb_list to long_memory_num video_emb_list
                long_video_emb_list = []
                if len(all_video_emb_list) > long_memory_num:
                    # calculate similarity
                    similar_list = []
                    for idx in range(len(all_video_emb_list) -1):
                        scores = all_video_emb_list[idx] @ all_video_emb_list[idx+1].transpose(-1, -2) # [96, 4096] @ [4096, 96]
                        frame_silimar = torch.mean(scores)
                        similar_list.append(frame_silimar)

                    # iteratively compress the queries
                    while len(all_video_emb_list) > long_memory_num: #把all-embed删减到设定mem-num，即n个qformer的结果96*4096
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

                if all_video_emb_list:
                    long_video_emb_list.append(torch.cat(all_video_emb_list, dim=0).view(1, -1, D))
                video_list = long_video_emb_list + short_video_emb_list
                
                question = f"Question: {anno_question}\n" + "Options:\n" + str(anno_choices)
                question = question.rstrip()
                prompt = system + question + question_prompt
                # prompt = system + get_asr(video_path.split('/')[-1].split('.')[0],data_anno["question_time"])+question + question_prompt

                ask(prompt, chat)

                llm_message = answer(
                    conv=chat, model=model, do_sample=False, 
                    img_list=video_list, max_new_tokens=100, 
                    answer_prompt=answer_prompt, print_res=True,
                )[0]
                # embed()
                # exit()

                # remove potential explanation
                llm_message = llm_message.strip().split('\n')[0]
                print(f"Pred: {llm_message}", flush=True)
                print(f"Answer: {anno_answer}", flush=True)

                end_time=time.time()
                print("time:",end_time-start_time)
                print()
                
                new_item={"question_id":data_anno["question_id"],"answer":data_anno["correct_answer"],"pred":llm_message,"q_num":data_anno["q_num"]}
                json.dump(new_item, f)
                f.write('\n')
                f.flush()
    

if __name__ == "__main__":
    qa_mode = "breakpoint"  # "global"
    longshort_memory_streaming_breakpoint_infer(qa_mode)
    exit()