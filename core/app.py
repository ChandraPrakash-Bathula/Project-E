from flask import Flask, request, render_template
from summarization import summarize_text
from translation import translate_text
from transcribe import convert_video_to_audio, transcribe_audio
from keyword_extraction import extract_keywords
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/another_page')
def another_page():
    return render_template('another_page.html')

@app.route('/home')
def home():
    return render_template('home.html')

# Route for summarization
@app.route('/summarize_text', methods=['POST'])
def summarize():
    text_to_summarize = request.form['text_to_summarize']
    summary = summarize_text(text_to_summarize)
    return render_template('summarization_result.html', result=summary)

# Route for translation
@app.route('/translate_text', methods=['POST'])
def translate():
    text_to_translate = request.form['text_to_translate']
    target_language = request.form['target_language']
    translated_text = translate_text(text_to_translate, target_language)
    return render_template('translation_result.html', result=translated_text)

# Route for keywords
@app.route('/extract_keywords', methods=['POST'])
def extract_keywords_route():
    text_to_process = request.form['text_to_process']
    keywords = extract_keywords(text_to_process)
    return render_template('keywords_result.html', keywords=keywords)

# Route for transcription
@app.route('/transcribe', methods=['POST'])
def transcribe():
    uploaded_file = request.files['mediaFile']
    is_video = uploaded_file.filename.endswith(('.mp4', '.mkv', '.avi'))
    if is_video:
        audio_path = convert_video_to_audio(uploaded_file)
        transcript = transcribe_audio(audio_path)
        return render_template('transcription_result.html', result=transcript)
    else:
        return "Please upload a video file."

if __name__ == '__main__':
    app.run(debug=True)
