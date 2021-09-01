import os

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.responses import HTMLResponse

import cv2
from edsr.edsr import Edsr
from nesrganp.nesrganp import NErganp


PROJECT_PATH = os.getcwd()
STATIC_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'static')
TEMPLATES_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'templates')
UPLOAD_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)

edsr = Edsr(os.path.join('edsr', 'ressources', 'EDSR_x4.pb'))
nesrganp = NErganp(os.path.join('nesrganp', 'ressources', 'nESRGANplus.pth'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_save(in_path, out_path):
    out_edsr, out_nerganp = edsr.predict(in_path), nesrganp.predict(in_path)
    output = (out_edsr + out_nerganp) / 2
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_path), output)


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})

@app.get('/about', response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {'request': request})

@app.post("/infer")
def infer(request: Request, image: UploadFile = File(...)):
    if image.filename == '':
        return templates.TemplateResponse("index.html")
    if image.filename and allowed_file(image.filename):
        in_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(in_path, 'wb') as file:
            contents = image.read()
            file.write(contents)
        
        out_path = os.path.join(UPLOAD_FOLDER, f'openisr-{image.filename}')
        process_save(in_path, out_path)
        return templates.TemplateResponse("inference.html", {'request': request, 'dl_path': out_path})

@app.get("/download_my_image/{dl_path}")
def download(dl_path):
    _, imagename = os.path.split(dl_path)
    ext = dl_path.rsplit('.', 1)[1].lower() if '.' in dl_path else 'png'
    return FileResponse(dl_path, media_type=f'image/{ext}', filename=imagename)