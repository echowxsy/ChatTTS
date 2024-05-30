import gradio as gr
import numpy as np
import wave
import time
import ChatTTS
import torch

chat = ChatTTS.Chat()
chat.load_models()
dim = chat.pretrain_models['gpt'].gpt.layers[0].mlp.gate_proj.in_features
std, mean = chat.pretrain_models['spk_stat'].chunk(2)

def infer_tts(text,seed,temperature,oral_level,laugh_level,break_level):
    params_refine_text = {
      'prompt': '[oral_'+str(oral_level)+'][laugh_'+str(laugh_level)+'][break_'+str(break_level)+']'
    }
    torch.manual_seed(seed)
    rand_spk = torch.randn(dim,device=std.device) * std + mean
    params_infer_code = {
    'spk_emb': rand_spk, # add sampled speaker
    'temperature': temperature, # using custom temperature
    'top_P': 0.7, # top P decode
    'top_K': 20, # top K decode
    }
    wavs = chat.infer(text, params_refine_text=params_refine_text,params_infer_code=params_infer_code, use_decoder=True)
    audio_data = np.array(wavs[0], dtype=np.float32)
    audio_data = (audio_data * 32767).astype(np.int16)
    sample_rate = 24000
    # timestamp
    out_file = "output"+str(time.time())+ ".wav"
    with wave.open(out_file, "w") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return out_file


# 定义Gradio界面
interface_inputs = [
    gr.Textbox(lines=5, label="Input Text",value="chat T T S 是一款强大的对话式文本转语音模型。它有中英混读和多说话人的能力。chat T T S 不仅能够生成自然流畅的语音，还能控制[laugh]笑声啊[laugh]，停顿啊[uv_break]语气词啊等副语言现象[uv_break]。这个韵律超越了许多开源模型[uv_break]。请注意，chat T T S 的使用应遵守法律和伦理准则，避免滥用的安全风险。[uv_break]"),
    gr.Number(minimum=0,step=1, value=6618, label="Seed"),
    gr.Slider(minimum=0.1, maximum=3, step=0.1, value=0.3, label="Temperature"),
    gr.Slider(minimum=0, maximum=9, step=1, value=2, label="Oral Level"),
    gr.Slider(minimum=0, maximum=2, step=1, value=0, label="Laugh Level"),
    gr.Slider(minimum=0, maximum=7, step=1, value=6, label="Break Level"),
]

interface_output = gr.Audio(type="filepath", label="Output Audio")

# 创建Gradio界面
iface = gr.Interface(
    fn=infer_tts,
    inputs=interface_inputs,
    outputs=interface_output,
    title="ChatTTS Demo",
    description="Enter text and adjust parameters for TTS synthesis.",
    allow_flagging="never"
)

# 启动Gradio服务器
iface.launch(server_port=8000, server_name="0.0.0.0",share=True)