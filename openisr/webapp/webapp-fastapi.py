from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/image")
async def image(image: UploadFile = File(...)):
    return {"filename": image.filename}