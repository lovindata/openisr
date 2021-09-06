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
STATIC_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'resources', 'static')
TEMPLATES_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'resources', 'templates')
UPLOAD_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'resources', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)

edsr = Edsr(os.path.join('edsr', 'resources', 'EDSR_x4.pb'))
nesrganp = NErganp(os.path.join('nesrganp', 'resources', 'nESRGANplus.pth'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_save(in_path, out_path):
    in_img = cv2.imread(in_path, cv2.IMREAD_COLOR) # Read as BGR
    out_edsr, out_nerganp = edsr.predict(in_img), nesrganp.predict(in_img)
    output = (out_edsr + out_nerganp) / 2
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_path), output)


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})

@app.get('/about', response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {'request': request})

@app.post("/infer", response_class=HTMLResponse)
async def infer(request: Request, image: UploadFile = File(...)):
    if image.filename and allowed_file(image.filename):
        in_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(in_path, 'wb') as file:
            contents = await image.read()
            file.write(contents)
        
        out_image_name = f'openisr-{image.filename}'
        out_path = os.path.join(UPLOAD_FOLDER, f'openisr-{image.filename}')
        process_save(in_path, out_path)
        return templates.TemplateResponse("inference.html", {'request': request, 'out_image_name': out_image_name})
    else:
        return templates.TemplateResponse("index.html", {'request': request})

@app.get("/download/{out_image_name}", response_class=FileResponse)
def download(out_image_name: str):
    out_path = os.path.join(UPLOAD_FOLDER, out_image_name)
    _, ext = os.path.splitext(out_image_name)
    return FileResponse(out_path, media_type=f'image/{ext}', filename=out_image_name)