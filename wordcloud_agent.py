import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from langdetect import detect
import re
import numpy as np
from PIL import Image
import io
import os
import urllib.request

# Download Urdu font if not present
def generate_wordcloud(text, language, mask=None):
    """Generate word cloud with language-specific settings"""
    font_path = LANGUAGE_CONFIG[language]['font']  # Correct way to access the font
    
    wordcloud = WordCloud(
        font_path=font_path,
        width=1200,
        height=800,
        background_color='white',
        max_words=200,
        mask=mask,
        collocations=False
    ).generate(text)
    
    return wordcloud
    import os
import urllib.request

# Download Urdu font if not present
if not os.path.exists("NotoNastaliqUrdu-Regular.ttf"):
    urllib.request.urlretrieve(
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoNastaliqUrdu/NotoNastaliqUrdu-Regular.ttf",
        "NotoNastaliqUrdu-Regular.ttf"
    )
    
# Set page config
st.set_page_config(page_title="Multilingual Word Cloud Agent", layout="wide")

# Language-specific settings
LANGUAGE_CONFIG = {
    'en': {'font': 'arial', 'reshaper': False},
    'ur': {'font': 'NotoNastaliqUrdu-Regular.ttf', 'reshaper': True},
    'ar': {'font': 'Arial Unicode MS', 'reshaper': True},
    'ps': {'font': 'NotoNastaliqUrdu-Regular.ttf', 'reshaper': True},
    'fa': {'font': 'Arial Unicode MS', 'reshaper': True}
}

# Text cleaning functions
def clean_text(text, language):
    """Clean text based on language"""
    if pd.isna(text):
        return ''
    
    text = str(text)
    
    # Common cleaning for all languages
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    
    # Language-specific cleaning
    if language in ['ur', 'ar', 'ps', 'fa']:
        # Remove English characters from RTL languages if needed
        text = re.sub(r'[a-zA-Z]', ' ', text)
    
    return text.strip()

def detect_language(text_sample):
    """Detect language from text sample"""
    try:
        lang = detect(text_sample[:500])  # Use first 500 chars for detection
        return lang
    except:
        return 'en'  # Default to English if detection fails

def process_text(text, language):
    """Process text based on language requirements"""
    if LANGUAGE_CONFIG[language]['reshaper']:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text

def generate_wordcloud(text, language, mask=None):
    """Generate word cloud with language-specific settings"""
    font_path = LANGUAGE_CONFIG[language]['NotoNastaliqUrdu-Regular.ttf']
    
    wordcloud = WordCloud(
        font_path=font_path,
        width=1200,
        height=800,
        background_color='white',
        max_words=200,
        mask=mask,
        collocations=False
    ).generate(text)
    
    return wordcloud

# Streamlit UI
st.title("🌍 Multilingual Word Cloud Agent")
st.write("Upload your dataset (CSV/Excel) and we'll generate a word cloud in English, Urdu, Arabic, Pashto, or Persian")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
text_column = None
language = None

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Column selection
    text_column = st.selectbox("Select the text column", df.columns)
    
    # Sample text for language detection
    sample_text = ' '.join(df[text_column].dropna().astype(str).sample(min(5, len(df))).tolist())
    detected_lang = detect_language(sample_text)
    
    # Language selection
    language = st.selectbox(
        "Select language", 
        options=['en', 'ur', 'ar', 'ps', 'fa'],
        index=['en', 'ur', 'ar', 'ps', 'fa'].index(detected_lang) if detected_lang in ['en', 'ur', 'ar', 'ps', 'fa'] else 0,
        format_func=lambda x: {
            'en': 'English',
            'ur': 'Urdu',
            'ar': 'Arabic',
            'ps': 'Pashto',
            'fa': 'Persian'
        }[x]
    )
    
    # Mask image upload
    mask_image = st.file_uploader("Optional: Upload mask image (PNG)", type=['png'])
    mask = None
    if mask_image:
        mask = np.array(Image.open(mask_image))
    
    # Generate button
    if st.button("Generate Word Cloud"):
        with st.spinner('Processing data and generating word cloud...'):
            # Clean and process text
            df['cleaned_text'] = df[text_column].apply(lambda x: clean_text(x, language))
            combined_text = ' '.join(df['cleaned_text'].dropna().astype(str).tolist())
            processed_text = process_text(combined_text, language)
            
            # Generate word cloud
            wordcloud = generate_wordcloud(processed_text, language, mask)
            
            # Display
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Download option
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0)
            img_bytes.seek(0)
            st.download_button(
                label="Download Word Cloud",
                data=img_bytes,
                file_name=f"wordcloud_{language}.png",
                mime="image/png"
            )

# Instructions
st.sidebar.markdown("""
### Instructions:
1. Upload your dataset (CSV/Excel)
2. Select the column containing text
3. The app will auto-detect language (you can change it)
4. Optionally upload a mask image
5. Click "Generate Word Cloud"
6. Download your word cloud image

### Supported Languages:
- English (en)
- Urdu (ur)
- Arabic (ar)
- Pashto (ps)
- Persian (fa)
""")
