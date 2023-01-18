from fastapi import FastAPI
from fastapi import UploadFile, File, BackgroundTasks
from typing import Optional
import cv2
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
import torch
from PIL import Image, ImageStat
import os
from transformers import CLIPProcessor, CLIPModel
import datetime

from app.app.data_drift import make_report

def add_to_database(
   now: str, features, prediction: int):
   with open('prediction_database.csv', 'a+') as file:
      # if file is empty, write header
      if os.stat('prediction_database.csv').st_size == 0:
         file.write("time, sepal_length, sepal_width, petal_length, petal_width, prediction\n")
      else:
         entry = ""
         for feature in features:
            entry += str(feature) + ", "
         entry += str(prediction) + "\n"
         file.write(entry)
   return None

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
async def create_upload_file(file: UploadFile = File(...), h: Optional[int] = 224, w: Optional[int] = 224, background_tasks: BackgroundTasks = None):
   contents = await file.read()
   # Save file locally
   with open(file.filename, "wb") as f:
      f.write(contents)
      f.close()

   img = cv2.imread(file.filename)
   res = cv2.resize(img, (h, w))
   cv2.imwrite(file.filename, res)

   # Create metadata for image
   now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
   processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

   inputs = processor(text=None, images=file.filename, return_tensors="pt", padding=True)
   img_features = model.get_image_features(inputs['pixel_values'])

   # Predict
   preds = predict_step([file.filename])
   background_tasks.add_task(add_to_database, now, img_features, preds[0])

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


# Monitoring endpoint
@app.get("/data_monitoring")
def iris_monitoring(): 
   # Make report
   make_report()
   # Return report
   return FileResponse("report.html")



