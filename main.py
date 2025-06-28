from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess_text

app = FastAPI()

model = load_model("model/model_klasifikasi_aduan.h5")

DEPARTEMENTS = ["DAAK", "KEAMANAN", "KEMAHASISWAAN", "PENGAJARAN", "PERPUS", "SARPRAS", "UPT_LAB"]

class Aduan(BaseModel):
    judul: str
    deskripsi: str

@app.post("/klasifikasi")
def klasifikasi_aduan(aduan: Aduan):
    try:
        input_text = aduan.judul + " " + aduan.deskripsi
        processed = preprocess_text(input_text)
        prediction = model.predict(processed)
        label_index = np.argmax(prediction)
        departemen = DEPARTEMENTS[label_index]
        return {
            "departemen": departemen,
            "confidence": float(prediction[0][label_index])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))