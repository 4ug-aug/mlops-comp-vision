from fastapi import FastAPI
from fastapi import UploadFile, File
from typing import Optional
import cv2
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
import torch
from PIL import Image
import os

def predict_step(image_paths):
   # Load model checkpoint
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # Get latest model checkpoint from models/trained_models
   latest_model_checkpoint = max([os.path.join("models/trained_models", f) for f in os.listdir("models/trained_models")], key=os.path.getctime)

   model = torch.load(latest_model_checkpoint)

   # Load image
   image = Image.open(image_paths[0])
   with torch.no_grad():
      log_ps = model(image)

      ps = torch.exp(log_ps)
      top_p, top_class = ps.topk(1, dim=1)

   return top_class

app = FastAPI()

@app.get('/')
async def upload_form_file():
    return HTMLResponse(content="""
      <h1> Upload an image of a butterfly to have it classified</h1>
      <form method="post" action="/uploadfile/" enctype="multipart/form-data">
         <input name="file" type="file">
         <input type="submit">
      </form>""")

@app.post("/uploadfile/")
# Example post url = localhost:8000/uploadfile/?filename=hello.txt
async def create_upload_file(file: UploadFile = File(...), h: Optional[int] = 224, w: Optional[int] = 224):
   contents = await file.read()
   # Save file locally
   with open(file.filename, "wb") as f:
      f.write(contents)
      f.close()

   img = cv2.imread(file.filename)
   res = cv2.resize(img, (h, w))
   cv2.imwrite(file.filename, res)

   # Predict
   preds = predict_step([file.filename])

   FileResponse(file.filename)

   # return html listing all the predictions
   return HTMLResponse(content=f"""
         <h1> Predictions </h1>
         <script> 
            var preds = {preds}
            for (var i = 0; i < preds.length; i++) {{
               document.write("<p>" + preds[i] + "</p>")
            }}
         </script>
         """)




