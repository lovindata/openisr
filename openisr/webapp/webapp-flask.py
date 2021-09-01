import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import cv2

from edsr.edsr import Edsr
from nesrganp.nesrganp import NErganp


UPLOAD_FOLDER = os.path.join('webapp', 'tmp')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

edsr = Edsr(os.path.join('edsr', 'ressources', 'EDSR_x4.pb'))
nesrganp = NErganp(os.path.join('nesrganp', 'ressources', 'nESRGANplus.pth'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def merge_save(out_edsr, out_nerganp, out_filename):
    output = (out_edsr + out_nerganp) / 2
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER,out_filename), output)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html')
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            savePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(savePath)
            out_filename = f"openisr-{filename}"
            merge_save(edsr.predict(savePath), nesrganp.predict(savePath), out_filename)
            return render_template('inference.html', out_filename=out_filename)

@app.route('/download_my_image/<filename>', methods=['GET'])
def download(filename):
    uploads = os.path.join('D:\prog\proj\openisr\openisr\openisr', app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, filename=filename, as_attachment=True)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)
