from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
import os, sys
from demo import Args, demo, Midi2Octuple, Octuple2Midi
from midi2audio import FluidSynth

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})


# 存储上传文件的目录
current_file_path = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(current_file_path, 'upload')
OUTPUT_FOLDER = os.path.join(current_file_path, 'output')
sf2 = os.path.join(current_file_path, 'default.sf2')
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        midi_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(midi_filename)

        # Convert MIDI to MP3
        wav_filename = midi_filename.rsplit('.', 1)[0] + '.wav'
        try:
            FluidSynth(sound_font=sf2).midi_to_audio(midi_filename, wav_filename)
            print(f"Transfered {midi_filename} to wav.")
        except Exception as e:
            print(e)
            return jsonify({'error': str(e)})
        return jsonify({'filename': file.filename.rsplit('.', 1)[0]})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['midi', 'mid']

@app.route('/api/<folder>/<path:filename>', methods=['GET'])
def download_file(folder, filename):
    if folder == 'upload':
        directory = app.config['UPLOAD_FOLDER']
    elif folder == 'output':
        directory = app.config['OUTPUT_FOLDER']
    else:
        return "Invalid folder", 404

    return send_from_directory(directory, filename)





@app.route('/api/generate/<path:model>/<path:filename>')
def generate(model, filename):
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}.mid")
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}.mid")
    
    input_oct = Midi2Octuple(input_path)
    
    
    if model == 'pianobart':
        args = Args(ckpt='PianoBART_Giant.ckpt', input=input_path, output=output_path)
    elif model == 'pianobart-simple':
        args = Args(ckpt='PianoBART_Giant.ckpt', input=input_path, output=output_path)
    else:
        return "ERROR: No Such Model", 400
    try:
        input_oct, output_oct = demo(args)
    except Exception as e:
        print(e)
        return f"PianoBart Generating ERROR: {e}", 400
    intro_midi = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_intro.mid")
    # output_midi = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}_output.mid")
    intro_wav = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_intro.wav")
    # output_wav = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}_output.wav")
    try:
        Octuple2Midi(input_oct, intro_midi)
        FluidSynth(sound_font=sf2).midi_to_audio(intro_midi, intro_wav)
        print(f"Transfered {intro_midi} to wav.")
        # Octuple2Midi(output_oct, output_midi)
        # FluidSynth(sound_font=sf2).midi_to_audio(output_midi, output_wav)
        # print(f"Transfered {output_midi} to wav.")
        wav_filename = args.output.rsplit('.', 1)[0] + '.wav'
        FluidSynth(sound_font=sf2).midi_to_audio(args.output, wav_filename)
        print(f"Transfered {args.output} to wav.")
    except Exception as e:
        print(e)
        return f"ERROR: {e}", 400
    return jsonify({'result': filename.rsplit('.', 1)[0]})
    



@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:fallback>')
def fallback(fallback):       # Vue Router 的 mode 为 'hash' 时可移除该方法
    if fallback.startswith('css/') or fallback.startswith('js/')\
            or fallback.startswith('img/') or fallback == 'favicon.ico':
        return app.send_static_file(fallback)
    else:
        return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=DEBUG)
