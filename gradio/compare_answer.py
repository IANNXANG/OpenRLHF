import gradio as gr
import pandas as pd

# 读取JSONL文件并将其转换为DataFrame
def read_jsonl(file_path):
    try:
        return pd.read_json(file_path, lines=True), None
    except Exception as e:
        return None, str(e)
    return None, "File not found"

# 初始化数据和索引
data1 = None
data2 = None
current_index = 0

# 更新当前显示的question和answer
def update_display(index):
    global data1, data2
    if data1 is None or data2 is None or index < 0 or index >= len(data1) or index >= len(data2):
        return "No data", "No data", "No data"
    question = data1.iloc[index]['question']
    gt = data1.iloc[index]['gt']
    answer1 = data1.iloc[index]['code'][0]
    answer2 = data2.iloc[index]['code'][0]
    return question, gt, answer1, answer2

# 加载文件时调用的函数
def load_files(file_path1, file_path2):
    global data1, data2, current_index
    data1, error1 = read_jsonl(file_path1)
    data2, error2 = read_jsonl(file_path2)
    
    if error1:
        return f"Error reading file 1: {error1}", "No data", "No data"
    if error2:
        return f"Error reading file 2: {error2}", "No data", "No data"
    
    current_index = 0
    return update_display(current_index)

# 切换到上一条数据
def prev_question():
    global current_index
    if data1 is not None and data2 is not None and current_index > 0:
        current_index -= 1
    return update_display(current_index)

# 切换到下一条数据
def next_question():
    global current_index
    if data1 is not None and data2 is not None and current_index < len(data1) - 1 and current_index < len(data2) - 1:
        current_index += 1
    return update_display(current_index)

# 创建Gradio界面
with gr.Blocks() as demo:
    with gr.Row():
        file_path_input1 = gr.Textbox(label="Enter path to JSONL File 1")
        file_path_input2 = gr.Textbox(label="Enter path to JSONL File 2")
        load_button = gr.Button("Load Files")
    
    question_output = gr.Textbox(label="Question")
    gt_output = gr.Textbox(label="Ground Truth")
    with gr.Row():
        answer1_output = gr.Textbox(label="Answer 1")
        answer2_output = gr.Textbox(label="Answer 2")
    
    prev_button = gr.Button("Previous Question")
    next_button = gr.Button("Next Question")

    load_button.click(fn=load_files, inputs=[file_path_input1, file_path_input2], outputs=[question_output, gt_output, answer1_output, answer2_output])
    prev_button.click(fn=prev_question, inputs=None, outputs=[question_output, gt_output, answer1_output, answer2_output])
    next_button.click(fn=next_question, inputs=None, outputs=[question_output, gt_output, answer1_output, answer2_output])

demo.launch()