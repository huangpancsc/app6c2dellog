import google.generativeai as genai
from openai import OpenAI
import os
import base64
import logging
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import time
from datetime import datetime

# 修改日誌格式，增加更多詳細信息
logging.basicConfig(
    filename='image_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ImageRecognition')
openaikey = os.environ.get('GPT4OMINIEKY')
googlegeminikey = os.environ.get('GEMINI15KEY')
twsllamakey = os.environ.get('LLAMA32VKEY')
# Initialize API settings
client_openai = OpenAI(api_key=openaikey)
client_llama = OpenAI(
   ## api_key="5cbfa72c-93eb-4977-a07e-a31acaaf3ac8",
    api_key=twsllamakey,
    base_url="https://api-ams.twcc.ai/api/models"
)
## genai.configure(api_key="AIzaSyC6iRgQarH9abLpnjU8O71L6Lb0K5pSkD4")
genai.configure(api_key=googlegeminikey)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Default prompt
DEFAULT_PROMPT = "請辨識圖片中的文字，並以中文回答，如果沒有文字請回答'沒有文字'。"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"檔案刪除成功 - 檔名: {filename}")
            return True
        return False
    except Exception as e:
        logger.error(f"檔案刪除失敗 - 檔名: {filename}, 錯誤: {str(e)}")
        return False

def log_recognition_result(model_name, filename, prompt, result_text, processing_time):
    """記錄辨識結果的詳細信息"""
    log_message = (
        f"\n{'='*50}\n"
        f"辨識完成\n"
        f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"模型: {model_name}\n"
        f"檔案: {filename}\n"
        f"提示詞: {prompt}\n"
        f"處理時間: {processing_time:.2f} 秒\n"
        f"辨識結果:\n{result_text}\n"
        f"{'='*50}"
    )
    logger.info(log_message)

def detect_text_gemini(image_path, prompt=DEFAULT_PROMPT):
    start_time = time.time()
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        image_part = {"mime_type": "image/png", "data": image_data}
        response = model.generate_content([prompt, image_part])
        
        processing_time = time.time() - start_time
        result_text = response.text
        
        log_recognition_result(
            "Gemini-1.5-Flash", 
            os.path.basename(image_path),
            prompt,
            result_text,
            processing_time
        )
        
        return {
            'text': result_text,
            'model': 'Gemini-1.5-Flash',
            'prompt': prompt,
            'processing_time': processing_time
        }
    except Exception as e:
       logger.error(f"Gemini處理錯誤 - 檔案: {image_path}, 錯誤: {str(e)}")
       return {
            'text': f"發生錯誤：{e}",
            'model': 'Gemini-1.5-Flash',
            'prompt': prompt,
            'processing_time': time.time() - start_time
        }

def detect_text_openai(image_path, prompt=DEFAULT_PROMPT):
    start_time = time.time()
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        processing_time = time.time() - start_time
        result_text = response.choices[0].message.content
        
        log_recognition_result(
            "GPT-4o-mini",
            os.path.basename(image_path),
            prompt,
            result_text,
            processing_time
        )
        
        return {
            'text': result_text,
            'model': 'GPT-4o-mini',
            'prompt': prompt,
            'processing_time': processing_time
        }
    except Exception as e:
        logger.error(f"OpenAI處理錯誤 - 檔案: {image_path}, 錯誤: {str(e)}")
        return {
            'text': f"發生錯誤：{e}",
            'model': 'GPT-4o-mini',
            'prompt': prompt,
            'processing_time': time.time() - start_time
        }

def detect_text_llama(image_path, prompt=DEFAULT_PROMPT):
    start_time = time.time()
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client_llama.chat.completions.create(
            model="llama3.2-ffm-11b-v-32k-chat",
            temperature=0.1,
            max_tokens=3600,
            top_p=0.5,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        processing_time = time.time() - start_time
        result_text = response.choices[0].message.content
        
        log_recognition_result(
            "Llama3.2-V",
            os.path.basename(image_path),
            prompt,
            result_text,
            processing_time
        )
        
        return {
            'text': result_text,
            'model': 'Llama3.2-V',
            'prompt': prompt,
            'processing_time': processing_time
        }
    except Exception as e:
        logger.error(f"Llama32-v處理錯誤 - 檔案: {image_path}, 錯誤: {str(e)}")
        return {
            'text': f"發生錯誤：{e}",
            'model': 'Llama3.2-V',
            'prompt': prompt,
            'processing_time': time.time() - start_time
        }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    text = None
    model_name = None
    image_url = None
    uploaded_filename = None
    used_prompt = None
    timestamp = str(int(time.time()))
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if request.method == 'POST':
        model_type = request.form.get('model_type')
        custom_prompt = request.form.get('prompt', '').strip()
        
        prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT

        if 'file' not in request.files:
            logger.warning("未選擇檔案")
            return render_template('index6c2dellog.html', 
                                error="沒有選擇檔案",
                                default_prompt=DEFAULT_PROMPT,
                                last_prompt=prompt)

        file = request.files['file']
        if file.filename == '':
            logger.warning("未選擇檔案")
            return render_template('index6c2dellog.html', 
                                error="沒有選擇檔案",
                                default_prompt=DEFAULT_PROMPT,
                                last_prompt=prompt)

        if file and allowed_file(file.filename):
            filename = secure_filename(f"{timestamp}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"檔案上傳成功 - 檔名: {filename}")

            if model_type == 'gemini':
                result = detect_text_gemini(file_path, prompt)
            elif model_type == 'llama':
                result = detect_text_llama(file_path, prompt)
            else:
                result = detect_text_openai(file_path, prompt)

            text = result['text']
            model_name = result['model']
            used_prompt = result['prompt']
            image_url = os.path.join('/uploads', filename)
            uploaded_filename = filename
            processing_time = result.get('processing_time', 0)

            return render_template('index6c2dellog.html', 
                                text=text, 
                                model_name=model_name,
                                image_url=image_url + f"?t={timestamp}",
                                uploaded_filename=uploaded_filename,
                                default_prompt=DEFAULT_PROMPT,
                                last_prompt=used_prompt,
                                current_time=current_time,
                                processing_time=processing_time)

    return render_template('index6c2dellog.html', 
                         default_prompt=DEFAULT_PROMPT,
                         current_time=current_time)

@app.route('/delete_file/<filename>', methods=['POST'])
def delete_uploaded_file(filename):
    if delete_file(filename):
        return jsonify({'success': True, 'message': '檔案已成功刪除'})
    return jsonify({'success': False, 'message': '刪除檔案失敗'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(file_path)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info("應用程式啟動")
    app.run(debug=True)
