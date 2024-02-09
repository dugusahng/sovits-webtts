import os,re,logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb

gpt_path = os.environ.get(
    "gpt_path", "models/Nana7mi/Nana7mi-e21.ckpt"
)
sovits_path = os.environ.get("sovits_path", "models/Nana7mi/Nana7mi_e47_s2444.pth")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True"))
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa,torch
from feature_extractor import cnhubert
cnhubert.cnhubert_base_path=cnhubert_base_path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('cmudict')

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio

device = "cuda" if torch.cuda.is_available() else "cpu"

is_half = eval(
    os.environ.get("is_half", "True" if torch.cuda.is_available() else "False")
)

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

def change_sovits_weights(sovits_path):
    global vq_model,hps
    dict_s2=torch.load(sovits_path,map_location="cpu")
    hps=dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if("pretrained"not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
change_sovits_weights(sovits_path)

def change_gpt_weights(gpt_path):
    global hz,max_sec,t2s_model,config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
change_gpt_weights(gpt_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language={
    ("中文"):"zh",
    ("英文"):"en",
    ("日文"):"ja"
}


def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z. ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)

    return textlist, langlist


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)

    return phones, word2ph, norm_text
def get_bert_inf(phones, word2ph, norm_text, language):
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def nonen_clean_text_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "en" or "ja":
            pass
        else:
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def nonen_get_bert_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    print(textlist)
    print(langlist)
    bert_list = []
    for i in range(len(textlist)):
        text = textlist[i]
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(text, lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert

def get_tts_wav(selected_text, prompt_text, prompt_language, text, text_language,how_to_cut=("不切")):
    ref_wav_path = text_to_audio_mappings.get(selected_text, "")
    if not ref_wav_path:
        print("Audio file not found for the selected text.")
        return
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k=torch.cat([wav16k,zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    
    if prompt_language == "en":
        phones1, word2ph1, norm_text1 = clean_text_inf(prompt_text, prompt_language)
    else:
        phones1, word2ph1, norm_text1 = nonen_clean_text_inf(prompt_text, prompt_language)
    if(how_to_cut==("凑五句一切")):text=cut1(text)
    elif(how_to_cut==("凑50字一切")):text=cut2(text)
    elif(how_to_cut==("按中文句号。切")):text=cut3(text)
    elif(how_to_cut==("按英文句号.切")):text=cut4(text)
    text = text.replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n")
    if(text[-1]not in splits):text+="。"if text_language!="en"else "."
    texts=text.split("\n")
    audio_opt = []
    if prompt_language == "en":
        bert1 = get_bert_inf(phones1, word2ph1, norm_text1, prompt_language)
    else:
        bert1 = nonen_get_bert_inf(prompt_text, prompt_language)

    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if text_language == "en":
            phones2, word2ph2, norm_text2 = clean_text_inf(text, text_language)
        else:
            phones2, word2ph2, norm_text2 = nonen_clean_text_inf(text, text_language)

        if text_language == "en":
            bert2 = get_bert_inf(phones2, word2ph2, norm_text2, text_language)
        else:
            bert2 = nonen_get_bert_inf(text, text_language)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config["inference"]["top_k"],
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}  # 不考虑省略号


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 5))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return [inp]
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s。" % item for item in inp.strip("。").split("。")])
def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s." % item for item in inp.strip(".").split(".")])

def scan_audio_files(folder_path):
    """ 扫描指定文件夹获取音频文件列表 """
    return [f for f in os.listdir(folder_path) if f.endswith('.wav')]

def load_audio_text_mappings(folder_path, list_file_name):
    text_to_audio_mappings = {}
    audio_to_text_mappings = {}
    with open(os.path.join(folder_path, list_file_name), 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                audio_file_name = parts[0]
                text = parts[3]
                audio_file_path = os.path.join(folder_path, audio_file_name)
                text_to_audio_mappings[text] = audio_file_path
                audio_to_text_mappings[audio_file_path] = text
    return text_to_audio_mappings, audio_to_text_mappings

audio_folder_path = 'audio/Nana7mi'
text_to_audio_mappings, audio_to_text_mappings = load_audio_text_mappings(audio_folder_path, 'Nana7mi.list')

with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    gr.Markdown(value="""
    # <center>【AI七海】在线语音生成（GPT-SoVITS）\n
    
    ### <center>模型作者：Xz乔希 https://space.bilibili.com/5859321\n
    ### <center>【GPT-SoVITS】在线合集：https://www.modelscope.cn/studios/xzjosh/GPT-SoVITS\n
    ### <center>数据集下载：https://huggingface.co/datasets/XzJosh/audiodataset\n
    ### <center>声音归属：七海Nana7mi https://space.bilibili.com/434334701\n
    ### <center>GPT-SoVITS项目：https://github.com/RVC-Boss/GPT-SoVITS\n
    ### <center>使用本模型请严格遵守法律法规！发布二创作品请标注本项目作者及链接、作品使用GPT-SoVITS AI生成！\n
    ### <center>⚠️在线端不稳定且生成速度较慢，强烈建议下载模型本地推理！\n
                """)
    # with gr.Tabs():

    with gr.Group():
        gr.Markdown(value="*参考音频选择（不建议选较长的）")
        with gr.Row():
            audio_select = gr.Dropdown(label="选择参考音频（必选）", choices=list(text_to_audio_mappings.keys()))
            ref_audio = gr.Audio(label="参考音频试听")
            ref_text = gr.Textbox(label="参考音频文本")
            
    # 定义更新参考文本的函数
        def update_ref_text_and_audio(selected_text):
            audio_path = text_to_audio_mappings.get(selected_text, "")
            return selected_text, audio_path

    # 绑定下拉菜单的变化到更新函数
        audio_select.change(update_ref_text_and_audio, [audio_select], [ref_text, ref_audio])

    # 其他 Gradio 组件和功能
        prompt_language = gr.Dropdown(
            label="参考音频语种", choices=["中文", "英文", "日文"], value="中文"
        )
        gr.Markdown(value="*请填写需要合成的目标文本，中英混合选中文，日英混合选日文，暂不支持中日混合。")
        with gr.Row():
            text = gr.Textbox(label="需要合成的文本", value="")
            text_language = gr.Dropdown(
                label="需要合成的语种", choices=["中文", "英文", "日文"], value="中文"
            )
            how_to_cut = gr.Radio(
                label=("自动切分（长文本建议切分）"),
                choices=[("不切"),("凑五句一切"),("凑50字一切"),("按中文句号。切"),("按英文句号.切"),],
                value=("不切"),
                interactive=True,
            )
            inference_button = gr.Button("合成语音", variant="primary")
            output = gr.Audio(label="输出的语音")
        inference_button.click(
            get_tts_wav,
            [audio_select, ref_text, prompt_language, text, text_language,how_to_cut],
            [output],
        )


    gr.Markdown(value="文本切分工具，需要复制。")
    with gr.Row():
        text_inp = gr.Textbox(label="需要合成的切分前文本", value="")
        button1 = gr.Button("凑五句一切", variant="primary")
        button2 = gr.Button("凑50字一切", variant="primary")
        button3 = gr.Button("按中文句号。切", variant="primary")
        button4 = gr.Button("按英文句号.切", variant="primary")
        text_opt = gr.Textbox(label="切分后文本", value="")
        button1.click(cut1, [text_inp], [text_opt])
        button2.click(cut2, [text_inp], [text_opt])
        button3.click(cut3, [text_inp], [text_opt])
        button4.click(cut4, [text_inp], [text_opt])

app.queue(max_size=10)
app.launch(inbrowser=True)
