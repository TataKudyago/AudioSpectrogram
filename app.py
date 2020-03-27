import os
import numpy as np
from pydub import AudioSegment
from flask import render_template
from librosa.core import load
from librosa import amplitude_to_db, stft, display
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['mp3'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def convert(filename):
    sound = AudioSegment.from_mp3(UPLOAD_FOLDER+'/'+filename)
    sound.export(UPLOAD_FOLDER+'/'+filename.split('.')[0], format="wav")
    delete_file(UPLOAD_FOLDER+'/'+filename)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def delete_file(path):
    if os.path.exists(path):
        os.remove(path)


delete_file('static/images/spectrogram.png')


def create_spectrogram(filename):
    convert(filename)
    y, sr = load(UPLOAD_FOLDER + '/' + filename.split('.')[0])
    delete_file(UPLOAD_FOLDER + '/' + filename.split('.')[0])
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    d = amplitude_to_db(np.abs(stft(y)), ref=np.max)
    display.specshow(d, sr=sr, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    delete_file('static/images/spectrogram.png')
    plt.savefig('static/images/spectrogram.png')


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            create_spectrogram(filename)
            return redirect(url_for('upload_file',
                                    filename=filename))
    return render_template('upload_file.html')


if __name__ == '__main__':
    app.run()
