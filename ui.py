import gradio as gr
import numpy as np
import wave
import time
import ChatTTS
chat = ChatTTS.Chat()
chat.load_models()


def infer_tts(text,oral_level,laugh_level,break_level):
    params_refine_text = {
      'prompt': '[oral_'+str(oral_level)+'][laugh_'+str(laugh_level)+'][break_'+str(break_level)+']'
    }
    wavs = chat.infer(text, params_refine_text=params_refine_text, use_decoder=True)
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