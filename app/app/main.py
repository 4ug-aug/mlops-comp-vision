from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
import re
from fastapi import UploadFile, File
from typing import Optional
import cv2
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}
def predict_step(image_paths):
   images = []
   for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
         i_image = i_image.convert(mode="RGB")

      images.append(i_image)
   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
   pixel_values = pixel_values.to(device)
   output_ids = model.generate(pixel_values, **gen_kwargs)
   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
   preds = [pred.strip() for pred in preds]
   return preds

app = FastAPI()

@app.get('/')
async def upload_form_file():
    return HTMLResponse(content="""
      <h1> Upload an image for caption generation</h1>
      <form method="post" action="/uploadfile/" enctype="multipart/form-data">
         <input name="file" type="file">
         <input type="submit">
      </form>""")

@app.post("/uploadfile/")
# Example post url = localhost:8000/uploadfile/?filename=hello.txt
async def create_upload_file(file: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
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




