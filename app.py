import streamlit as st
import boto3
from deep_translator import MyMemoryTranslator

# Khởi tạo các dịch vụ AWS (Chỉ dùng Rekognition và Polly)
rekognition = boto3.client('rekognition')
polly = boto3.client('polly')

# Giao diện Web
st.title("👁️ Ứng dụng AI Thuyết Minh Ảnh")
st.write("Tải một bức ảnh lên, AI sẽ phân tích, dịch và đọc mô tả cho bạn nghe!")

# Hàm 1: Nhận diện ảnh bằng AWS Rekognition
def nhan_dien_anh(image_bytes):
    response = rekognition.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=5,
        MinConfidence=70
    )
    labels = [label['Name'] for label in response['Labels']]
    return ", ".join(labels)

# Hàm 2: Dịch sang Tiếng Việt (Dùng MyMemory để không bị lỗi giới hạn)
def dich_sang_tieng_viet(text_tieng_anh):
    return MyMemoryTranslator(source='en-US', target='vi-VN').translate(text_tieng_anh)

# Hàm 3: Chuyển văn bản thành giọng nói bằng AWS Polly
def doc_tieng_viet(text):
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna' 
    )
    return response['AudioStream'].read()

# Xử lý tải ảnh
uploaded_file = st.file_uploader("Chọn một bức ảnh (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption='Ảnh đã tải lên', use_column_width=True)
    
    st.write("⏳ AI đang xử lý, vui lòng đợi...")
    
    try:
        # Bước 1: Nhận diện
        text_tieng_anh = nhan_dien_anh(image_bytes)
        st.success(f"**AI Nhận diện (Tiếng Anh):** {text_tieng_anh}")
        
        # Bước 2: Dịch
        text_tieng_viet = dich_sang_tieng_viet(text_tieng_anh)
        st.info(f"**Bản dịch (Tiếng Việt):** {text_tieng_viet}")
        
        # Bước 3: Đọc thành tiếng
        audio_data = doc_tieng_viet(text_tieng_viet)
        st.audio(audio_data, format='audio/mp3')
        
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")