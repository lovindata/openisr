import os

from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import HTMLResponse

from celery.result import AsyncResult
from tasksapp.app import process_save


# Project global paths
PROJECT_PATH = os.getcwd()
STATIC_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'resources', 'static')
TEMPLATES_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'resources', 'templates')
UPLOAD_FOLDER = os.path.join(PROJECT_PATH, 'webapp', 'resources', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize FastAPI main variable
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)

# File extension handler
def allowed_file(filename: str) -> bool:
    """File extension handler.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: `true` if allowed file extension `false` otherwise.
    """
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# FastAPI routes
@app.get('/', response_class=HTMLResponse)
def index(request: Request) -> templates.TemplateResponse:
    """Route to render `index.html` when GET at `/`.

    Args:
        request (Request): An empty body.

    Returns:
        TemplateResponse: Template `index.html` rendered with an HTTP response.
    """
    
    return templates.TemplateResponse("index.html", {'request': request})

@app.get('/about', response_class=HTMLResponse)
def about(request: Request) -> templates.TemplateResponse:
    """Route to render `about.html` when GET at `/about`.

    Args:
        request (Request): An empty body.

    Returns:
        TemplateResponse: Template `about.html` rendered with an HTTP response.
    """
    
    return templates.TemplateResponse("about.html", {'request': request})

@app.post("/infer", response_class=HTMLResponse)
async def infer(request: Request, image: UploadFile = File(...)) -> templates.TemplateResponse:
    """Route to render `inference.html` when POST at `/infer` and run a Celery task for prediction.

    Args:
        request (Request): An empty body.

    Returns:
        TemplateResponse: Template `inference.html` rendered with an HTTP response and {'task_id' (str), 'out_image_name' (str)}.
    """
    
    # Check if allowed extension
    if image.filename and allowed_file(image.filename):
        # Save the received image to the FileSystem
        in_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(in_path, 'wb') as file:
            contents = await image.read()
            file.write(contents)
        
        # Async predict on the image saved
        out_image_name = f'openisr-{image.filename}'
        out_path = os.path.join(UPLOAD_FOLDER, f'openisr-{image.filename}')
        task_var = process_save.delay(in_path, out_path)
        
        # Go to "inference.html"
        return templates.TemplateResponse("inference.html", {'request': request, 'task_id': task_var.id, 'out_image_name': out_image_name})
    else:
        return templates.TemplateResponse("index.html", {'request': request})

@app.get("/tasks/{task_id}")
def get_status(task_id: str) -> JSONResponse:
    """Route to get the Celery task status with ID = `task_id`.

    Args:
        task_id (str): The Celery task ID.

    Returns:
        JSONResponse: Containing {"task_id" (str), "task_status" (str), "task_result" (str)}
    """
    
    # Get the result from celery in an async mode
    task_result = AsyncResult(task_id) 
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JSONResponse(result)

@app.get("/download/{out_image_name}", response_class=FileResponse)
def download(out_image_name: str) -> FileResponse:
    """Route to download the processed image at `out_image_name`.

    Args:
        out_image_name (str): The processed image path.

    Returns:
        FileResponse: Launch the download.
    """
    
    # Return a FileResponse to automatically launch the download for the user
    out_path = os.path.join(UPLOAD_FOLDER, out_image_name)
    _, ext = os.path.splitext(out_image_name)
    return FileResponse(out_path, media_type=f'image/{ext}', filename=out_image_name)