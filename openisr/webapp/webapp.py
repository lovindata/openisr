from flask import Flask, render_template, request
import os
#from edsr.edsr import Edsr
#from nesrganp.nesrganp import NErganp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def merger(out_edsr, out_nerganp):
    output = (out_edsr + out_nerganp) / 2
    output = None
    None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = 'tmp/' + f.filename
        f.save(saveLocation)
        inference, confidence = None, None
        os.remove(saveLocation)
        # respond with the inference
        return render_template('inference.html', name=inference, confidence=confidence)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)
