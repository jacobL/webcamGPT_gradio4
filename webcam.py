import os
import cv2
import uuid

import gradio as gr
import numpy as np

import webcamgpt    

# css 設定
CSS =""" 
.gradio-container {background-color: black}
footer {visibility: hidden}
#switch_btn {background: #000000; margin-top:210px;font-size: 20px !important;}
#switch_btn  {color: white}
#switch_btn {border:0px}
#left_image_output_stream {background-color: #bdc3c7;margin-top:-40px;bottom:-10px;margin-left:300px; border-radius: 5px;width:10px !important;}
#right_image_output_stream {background-color: #bdc3c7;margin-top:-40px;bottom:-10px;margin-left:300px; border-radius: 5px;width:10px !important;}

.audio-waveform {display: none;}
//#component-20 {display: none;}
.waveform-container {display: none;}
.timestamps  {display: none;}
.control-wrapper {display: none;}
.play-pause-wrapper {display: none;}
.settings-wrapper {display: none;}
//.controls {display: none;} 
""" 

# javascript 目前沒用到
JS = """
function auto_run(){}
"""
 
start_jscode = """  
function (){  
}
"""
# 常數變數
ratio = 0.05 # zoom in/out的單位
image_height = webcamgpt.Image_Height # 影像框高度
chat_height = webcamgpt.Chat_Height # 對話框高度
Placeholder = "請先選擇圖片，再輸入問題，按下Enter或箭頭按鈕來詢問"

# 建立模型物件
opanAIConnector = webcamgpt.OpanAIConnector() 
googleConnector = webcamgpt.GoogleConnector()

# 全域變數
switch_tag = 0 # 紀錄左右屏幕切換
zoom_ratio = 1 # 紀錄zoom in/out 的狀態
left_model_select = "gemini-1.5-pro" 
right_model_select = "gemini-1.5-pro"
  
with gr.Blocks(fill_width=True,  css=CSS ) as demo: 
    with gr.Row(): 
        # 照片儲存到本地    
        def save_image_to_drive(image: np.ndarray) -> str:
            image_filename = f"{uuid.uuid4()}.jpeg"
            image_directory = "data" #照片資料夾
            os.makedirs(image_directory, exist_ok=True)
            image_path = os.path.join(image_directory, image_filename)
            cv2.imwrite(image_path, image)
            return image_path
                    
        # 右推論(左圖片+右文字)
        def right_respond(image: np.ndarray, prompt: str, chat_history):
            if image is None:  
                return None, None, None 
            
            global left_model_select
            image_tmp = image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_path = save_image_to_drive(image) 
            # "openai-o4-mini","openai-o4","gemini-1.5-pro"
            if left_model_select in ("gpt-4o-mini","gpt-4o") :
                response = opanAIConnector.prompt(image=image, prompt=prompt, model=left_model_select)
            elif left_model_select == "gemini-1.5-pro" :  
                response = googleConnector.prompt(image=image, prompt=prompt)
            chat_history.append(((image_path,), None))
            chat_history.append((prompt, response)) 
            return image_tmp,"", chat_history

        # 左推論(左文字+右圖片)    
        def left_respond(image: np.ndarray, prompt: str, chat_history):
            if image is None:  
                return None,None,None 
                
            global right_model_select
            print("left_respond:",right_model_select)
            image_tmp = image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_path = save_image_to_drive(image) 
            if right_model_select in ("gpt-4o-mini","gpt-4o") :
                response = opanAIConnector.prompt(image=image, prompt=prompt, model=right_model_select)
            elif right_model_select == "gemini-1.5-pro" :  
                response = googleConnector.prompt(image=image, prompt=prompt)
            chat_history.append(((image_path,), None))
            chat_history.append((prompt, response))
            return image_tmp,"", chat_history
        
        # 圖片放大
        def zoom_in(image: np.ndarray):
            if image is None: return None 
            global zoom_ratio
            zoom_ratio = zoom_ratio - ratio            
            if zoom_ratio <= 0.5 : # 放大的極限
                zoom_ratio = zoom_ratio + ratio    
            height, width = image.shape[:2]     
            resize_h_1 = int(height*(1-zoom_ratio))
            resize_h_2 = int(height*zoom_ratio)            
            resize_w_1 = int(width*(1-zoom_ratio))
            resize_w_2 = int(width*zoom_ratio)

            #height:1280 , width:720 
            image = image[resize_h_1:resize_h_2, resize_w_1:resize_w_2] 
            zoom_factor = 3
            return cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor) 
        
        # 圖片縮小
        def zoom_out(image: np.ndarray):
            if image is None: return None 
            global zoom_ratio 
            if zoom_ratio == 1: #沒有放大就不需要縮小
                return image           
            zoom_ratio = zoom_ratio + ratio
            height, width = image.shape[:2]
            resize_h_1 = int(height*(1-zoom_ratio))
            resize_h_2 = int(height*zoom_ratio)
            resize_w_1 = int(width*(1-zoom_ratio))
            resize_w_2 = int(width*zoom_ratio)

            #height:1280 , width:720 
            image = image[resize_h_1:resize_h_2, resize_w_1:resize_w_2] 
            zoom_factor = 3
            return cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor) 
        
        # 圖片逆時針轉90度
        def image_rotate_90_counterclockwise(image2: np.ndarray, image3: np.ndarray):
            if image2 is None: return None , None               
            return np.rot90(image2, k=1, axes=(0,1)) , np.rot90(image3, k=1, axes=(0,1)) 
        
        # 圖片左右鏡射
        def image_rotate_fliplr(image2: np.ndarray, image3: np.ndarray):
            if image2 is None: return None , None                  
            return np.fliplr(image2) , np.fliplr(image3)    
        
        # 圖片順逆時針轉90度
        def image_rotate_90_clockwise(image2: np.ndarray, image3: np.ndarray):   
            if image2 is None: return None , None             
            return np.rot90(image2, k=1, axes=(1,0)) , np.rot90(image3, k=1, axes=(1,0))
        
        # 當檔案上傳或貼圖時，圖片不做旋轉或鏡射處理
        def image_select(image: np.ndarray):
            return image, image
        
        # 當透過WebCam拍攝影像時，左圖框的影像需要逆時針旋轉並且左右鏡射
        def image_rotate_left(image: np.ndarray): 
            global zoom_ratio 
            zoom_ratio = 1 # 重置zoom值

            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) #逆時針
            image = np.fliplr(image) # 左右反轉 
 
            return image, image

        # 當透過WebCam拍攝影像時，右圖框的影像需要順時針旋轉並且左右鏡射    
        def image_rotate_right(image: np.ndarray):             
            global zoom_ratio 
            zoom_ratio = 1 # 重置zoom值
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) #順時針
            image = np.fliplr(image) # 左右反轉 
            return image, image

        # 左圖文推論模型切換
        def left_model_change(left_model):
            global left_model_select
            left_model_select = left_model
            print('left_model_change:',left_model)
        
        # 右圖文推論模型切換
        def right_model_change(right_model):
            global right_model_select
            right_model_select = right_model
            print('right_model_change:',right_model)

        # 左右框切換    
        def switch():
            global switch_tag
            if switch_tag ==0 : # switch to mode 1: left chat , right webcam
                switch_tag = 1
                return {
                    # mode 0
                    left_img_col: gr.Column(visible=False),
                    right_chat_col: gr.Column(visible=False),
                    # mode 1
                    left_chat_col: gr.Column(visible=True), 
                    right_img_col: gr.Column(visible=True)                
                }
            else : # switch to mode 0: left webcam , right chat
                switch_tag = 0
                return {
                    # mode 0
                    left_img_col: gr.Column(visible=True),
                    right_chat_col: gr.Column(visible=True),
                    # mode 1
                    left_chat_col: gr.Column(visible=False),
                    right_img_col: gr.Column(visible=False)
                }
                
        # left webcam
        with gr.Column(min_width=50,scale=1) as left_img_col:
            with gr.Group():
                # 左圖備份(隱藏，給zoom in/out使用)
                left_image_output_backup = gr.Image(height=1, sources= ['upload' , 'clipboard'], interactive=True, container=False, visible=False )               
                
                # 左圖主畫面
                left_image_output_main = gr.Image(height=image_height, sources= ['upload' , 'clipboard'], interactive=True, container=False  )                             
                
                # 左cam影像，縮小只呈現紅色按鈕
                left_image_output_stream = gr.Image(height=33, min_width=48, sources= ['webcam'], interactive=True, container=False, elem_id="left_image_output_stream")
                
                with gr.Row():                    
                    # 左圖片左右翻轉 ▮↔▯
                    left_fliplr_btn = gr.Button("▮↔▯", min_width=10, elem_id="left_fliplr_btn" )
                    
                    # 左圖片逆時針轉90度
                    left_rotate_90_counterclockwise_btn = gr.Button("↶", min_width=5, elem_id="left_rotate_90_counterclockwise_btn" )                    
                    
                    # 左圖片順時針轉90度
                    left_rotate_90_clockwise_btn = gr.Button("↷", min_width=5, elem_id="left_rotate_90_clockwise_btn" )
                    
                    # 左圖片放大
                    left_zoom_in_btn = gr.Button("㊉", min_width=5)
                    
                    # 左圖片縮小
                    left_zoom_out_btn = gr.Button("㊀", min_width=5) 
                    
                    # 左圖文推論模型選單                    
                    left_model = gr.Dropdown( min_width=150, choices=["gpt-4o-mini", "gpt-4o", "gemini-1.5-pro"], interactive=True, value="gemini-1.5-pro", container=False, elem_id="left_model" )
        # left chat 
        with gr.Column(min_width=50, scale=1, visible=False) as left_chat_col:
            with gr.Group():
                # 左對話框
                left_chatbot = gr.Chatbot(height=chat_height, container=False)
                with gr.Row(): 
                    # 左文字輸入框
                    left_message = gr.Text(placeholder=Placeholder, label="", container=False, min_width=430 )
                    
                    # 左文字輸入框 submit
                    left_send_btn = gr.Button("➤" , min_width=20)
                with gr.Row():
                    # 左語音轉文字
                    left_audio = gr.Audio(sources=["microphone"], type="filepath", streaming=False,container=False , waveform_options=gr.WaveformOptions(show_recording_waveform=False), min_width=250)
         
        # switch button 中間切換icon
        with gr.Column(min_width=50,scale=0) as b1: 
            # 左右切換按鈕           
            switch_btn = gr.Button("↹", scale=0, min_width=50, elem_id="switch_btn", elem_classes="feedback")
            
            # audio語音模型選單，預設openai whisper-1(隱藏)
            model = gr.Dropdown(choices=["whisper-1"], label="Model", value="whisper-1", visible=False, min_width=50)
            
            # audio 語音轉文字輸出類型選單，預設text(隱藏)
            response_type = gr.Dropdown(choices=["json", "text", "srt", "verbose_json", "vtt"], label="Response Type",
                                    value="text",visible=False)
            
        # right chat
        with gr.Column(min_width=50, scale=1) as right_chat_col:
            with gr.Group():
                # 右對話框
                right_chatbot = gr.Chatbot(height=chat_height, container=False,    elem_id="right_chatbot")
                with gr.Row():
                    # 右文字輸入框  
                    right_message = gr.Text(placeholder=Placeholder, label="", container=False, min_width=430 )
                    
                    # 右文字輸入框 submit
                    right_send_btn = gr.Button("➤" , min_width=20)
                with gr.Row():
                    # 右語音轉文字
                    right_audio = gr.Audio(sources=["microphone"], type="filepath", streaming=False, container=False , waveform_options=gr.WaveformOptions(show_recording_waveform=False), min_width=250) 
                    
        # right webcam            
        with gr.Column(min_width=50, scale=1, visible=False) as right_img_col:
            with gr.Group():
                # 右圖備份(隱藏，給zoom in/out使用)
                right_image_output_backup = gr.Image(height=1, sources= ['upload' , 'clipboard'], interactive=True, container=False, visible=False )               
                
                # 右圖主畫面
                right_image_output_main = gr.Image(height=image_height, sources= ['upload' , 'clipboard'], interactive=True, container=False )                             
                
                # 右cam影像，縮小只呈現紅色按鈕
                right_image_output_stream = gr.Image(height=33, min_width=48,sources= ['webcam'], interactive=True, container=False, elem_id="right_image_output_stream")
               
                with gr.Row():
                    # 右圖文推論模型選單
                    right_model = gr.Dropdown( min_width=150, choices=["gpt-4o-mini", "gpt-4o", "gemini-1.5-pro"], interactive=True, value="gemini-1.5-pro", container=False, elem_id="right_model" )
                    
                    # 右圖片左右翻轉
                    right_fliplr_btn = gr.Button("▮↔▯", min_width=20)
                    
                    # 右圖片逆時針轉90度
                    right_rotate_90_counterclockwise_btn = gr.Button("↶", min_width=20)

                    # 右圖片順時針轉90度                  
                    right_rotate_90_clockwise_btn = gr.Button("↷", min_width=20)

                    # 右圖片放大
                    right_zoom_in_btn = gr.Button("㊉", min_width=20 )
                    
                    # 右圖片縮小
                    right_zoom_out_btn = gr.Button("㊀", min_width=20 )  
    # mode 0: left webcam , right  #########################################################
    # 左影像框Event:
    # Event:左相機影像輸入
    left_image_output_stream.stream(image_rotate_left, left_image_output_stream, [left_image_output_main, left_image_output_backup]  )
    
    # Event:左相機影像輸出
    left_image_output_main.upload(image_select, left_image_output_main, [left_image_output_main, left_image_output_backup])
    
    # Event:左照片逆時針90旋轉
    left_rotate_90_counterclockwise_btn.click(image_rotate_90_counterclockwise, [left_image_output_main, left_image_output_backup], [left_image_output_main, left_image_output_backup])
    
    # Event:左照片鏡射
    left_fliplr_btn.click(image_rotate_fliplr, [left_image_output_main, left_image_output_backup], [left_image_output_main, left_image_output_backup])
    
    # Event:左照片順時針90旋轉
    left_rotate_90_clockwise_btn.click(image_rotate_90_clockwise, [left_image_output_main, left_image_output_backup], [left_image_output_main, left_image_output_backup])
   
    # Event:左照片放大
    left_zoom_in_btn.click(zoom_in, left_image_output_backup, left_image_output_main)
    
    # Event:左照片縮小
    left_zoom_out_btn.click(zoom_out, left_image_output_backup, left_image_output_main)
    
    # Event:左圖文推論模型切換
    left_model.change(left_model_change, left_model, [])

    # 右對話框Event:
    # Event:右對話Enter送出
    right_message.submit(right_respond, [left_image_output_main, right_message, right_chatbot], [left_image_output_main,right_message, right_chatbot] )
    
    # Event:右對話按鈕送出
    right_send_btn.click(right_respond, [left_image_output_main, right_message, right_chatbot], [left_image_output_main,right_message, right_chatbot] ) 
    
    # Event:右語音輸入(語音轉文字)
    right_audio.stop_recording(fn=opanAIConnector.transcript, inputs=[right_audio, model, response_type], outputs=right_message, api_name=False)
     
    # mode 1: left chat , right webcam #########################################################
    # 左對話框Event:
    # Event:左對話Enter送出
    left_message.submit(left_respond, [right_image_output_main, left_message, left_chatbot], [right_image_output_main,left_message, left_chatbot])
    
    # Event:左對話按鈕送出
    left_send_btn.click(left_respond, [right_image_output_main, left_message, left_chatbot], [right_image_output_main,left_message, left_chatbot])
    
    # Event:左語音輸入(語音轉文字)
    left_audio.stop_recording(fn=opanAIConnector.transcript, inputs=[left_audio, model, response_type], outputs=left_message, api_name=False) 

    # 右影像框
    # Event:右相機影像輸入
    right_image_output_stream.stream(image_rotate_right,right_image_output_stream,[right_image_output_main,right_image_output_backup])
    
    # Event:右相機影像輸出
    right_image_output_main.upload(image_select,right_image_output_main,[right_image_output_main,right_image_output_backup])

    # Event:右照片逆時針90旋轉
    right_rotate_90_counterclockwise_btn.click(image_rotate_90_counterclockwise,[right_image_output_main, right_image_output_backup], [right_image_output_main, right_image_output_backup])
    
    # Event:右照片鏡射
    right_fliplr_btn.click(image_rotate_fliplr,[right_image_output_main, right_image_output_backup], [right_image_output_main, right_image_output_backup])
    
    # Event:右照片順時針90旋轉
    right_rotate_90_clockwise_btn.click(image_rotate_90_clockwise, [right_image_output_main, right_image_output_backup], [right_image_output_main, right_image_output_backup])
    
    # Event:右照片放大
    right_zoom_in_btn.click(zoom_in, right_image_output_backup, right_image_output_main)
    
    # Event:右照片縮小
    right_zoom_out_btn.click(zoom_out, right_image_output_backup, right_image_output_main)    
    
    # Event:右圖文推論模型切換
    right_model.change(right_model_change, right_model, [])

    # Event:左右屏幕切換
    switch_btn.click(switch, [], [left_img_col, right_chat_col, right_img_col, left_chat_col])

# 啟動
demo.launch(debug=False, show_error=True)