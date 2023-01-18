from fastapi import FastAPI
from fastapi import UploadFile, File, BackgroundTasks
from typing import Optional
import cv2
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime

from app.app.data_drift import make_report

def add_to_database(
   now: str, features, prediction: int):
   with open('app/prediction_database.csv', 'a+') as file:
      # Convert features to list
      features = features.tolist()[0]
      entry = f"{now},"
      for feature in features:
         entry += str(feature) + ", "
      entry += str(prediction) + "\n"

      if os.stat('app/prediction_database.csv').st_size == 0:
         # Write header
         header = "timestamp,"
         for i in range(1, len(features)):
            header += f"feature_{i}, "
         header += "prediction\n"
         file.write(header)
         file.write(entry)
      else:
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
   transform = transforms.ToTensor()
   image = transform(image).unsqueeze(dim=0)
   with torch.no_grad():
      log_ps = model(image)

      ps = torch.exp(log_ps)
      top_p, top_class = ps.topk(1, dim=1)

   return top_class.item()

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
   img_path = "app/"+file.filename
   with open(img_path, "wb") as f:
      f.write(contents)
      f.close()

   img = cv2.imread(img_path)
   res = cv2.resize(img, (h, w))
   cv2.imwrite(img_path, res)

   # Create metadata for image
   now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
   processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

   # Load image as PIL
   img = Image.open(img_path)

   inputs = processor(text=None, images=img, return_tensors="pt", padding=True)
   img_features = model.get_image_features(inputs['pixel_values'])

   # Predict
   preds = predict_step([img_path])
   background_tasks.add_task(add_to_database, now, img_features, preds)

   FileResponse(img_path)

   # return html listing all the predictions
   return HTMLResponse(content=f"""
         <h1> Predictions </h1>
         <p> The image was classified as: {preds} </p>
         """)


# Monitoring endpoint
@app.get("/data_monitoring")
def iris_monitoring(): 
   # Make report
   make_report()
   # Return report
   return FileResponse("app/report.html")



