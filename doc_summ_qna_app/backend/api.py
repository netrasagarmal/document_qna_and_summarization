"""
Total APIS
1. create session
2. upload file
3. upload text
4. delete file
5. generate_summary
6. qna
7. load data


"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from io import BytesIO
import uvicorn
import requests, ast, json
import os
import requests, ast, json
from pydantic import BaseModel
import uuid
from summ_and_qna import DocSummAndQnA
import tempfile


current_session = {}

# Initialize FastAPI app
app = FastAPI()

# class CreateSession(BaseModel):
#     file 

ans = """The **LayoutLMv3** paper titled **"LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking"** presents the third version of the LayoutLM series, designed for Document AI tasks by Microsoft Research Asia.

Here are the **key details and contributions** from the paper:

---

### 📄 **Paper Details**
- **Title:** LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
- **Authors:** Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei
- **Published by:** Microsoft Research Asia
- **arXiv Link:** [https://arxiv.org/abs/2204.08387](https://arxiv.org/abs/2204.08387)

---

### 🚀 Motivation
The LayoutLM series aims to better understand scanned documents or digital-born PDFs by combining **text, layout, and image** information. LayoutLMv3 improves over previous versions with better pre-training and vision-language alignment.

---

### 🔑 Core Contributions

1. **Unified Multi-modal Masked Modeling**
   - Introduces a **unified pre-training objective**: both **text tokens and image patches** are randomly masked and predicted, helping the model learn aligned semantic representations.
   - Combines **masked language modeling (MLM)** and **image masking** like in MAE (Masked Autoencoders).

2. **Multi-modal Transformer Encoder**
   - Uses a single transformer backbone to jointly process **text, layout (bounding box), and image patches**.
   - Text embeddings + 2D positional encodings (layout) + visual features from an image encoder are fused together.

3. **Efficient Pre-training**
   - Built on **BEiT** and **RoBERTa**-like pre-training techniques for both vision and language.
   - Efficient handling of document images and text in a single pipeline.

---

### 🏆 Results
LayoutLMv3 achieves **state-of-the-art performance** on multiple Document AI benchmarks:

- **FUNSD** (Form Understanding in Noisy Scanned Documents)
- **SROIE** (Scanned Receipt OCR and Information Extraction)
- **CORD** (Consolidated Receipt Dataset)
- **DocVQA** (Document Visual Question Answering)

For instance, it significantly outperforms previous LayoutLMv2 and TILT models in key metrics like F1 and exact match.

---

### 🔍 Architecture Overview

**Inputs:**
- Text tokens (from OCR)
- Bounding box coordinates (layout info)
- Image patches (from resized document image)

**Embedding Layer:**
- Text embeddings + layout embeddings
- Image patch embeddings

**Transformer Encoder:**
- Processes all modalities together
- Uses cross-modality attention to learn interactions

**Pre-training Objectives:**
- **Masked Language Modeling (MLM)** for text
- **Masked Image Modeling (MIM)** for visual patches

---

### 🧠 Technical Highlights

- **Image Encoder**: Swin Transformer-based image patch encoder.
- **Token Alignment**: Improves alignment between visual patches and text regions through unified attention.
- **2D Positional Encoding**: Adds spatial awareness using the layout (bounding boxes).

---

### 📦 Applications
- Invoice and receipt parsing
- Form understanding
- Document classification
- Visual Question Answering (VQA) on documents
- Document-level information extraction

---

Would you like a visual diagram of the architecture or a breakdown of how it compares to LayoutLMv2 or DocFormer?"""
qna_dict = {
    "layoutlm architecture":"""### 🔍 Architecture Overview

                                **Inputs:**
                                - Text tokens (from OCR)
                                - Bounding box coordinates (layout info)
                                - Image patches (from resized document image)

                                **Embedding Layer:**
                                - Text embeddings + layout embeddings
                                - Image patch embeddings

                                **Transformer Encoder:**
                                - Processes all modalities together
                                - Uses cross-modality attention to learn interactions

                                **Pre-training Objectives:**
                                - **Masked Language Modeling (MLM)** for text
                                - **Masked Image Modeling (MIM)** for visual patches
                                """,
    "layoutlm sample code":"""Here’s a **sample code** to use **LayoutLMv3** from the Hugging Face Transformers library for **Document Question Answering (DocVQA)** or other tasks like key-value extraction.

                                ---

                                ### 🔧 **Installation**
                                First, install the required libraries:

                                ```bash
                                pip install transformers torchvision pytesseract
                                ```

                                You’ll also need **Tesseract OCR** installed on your system. On Ubuntu:

                                ```bash
                                sudo apt install tesseract-ocr
                                ```

                                ---

                                ### 📄 **Sample Code: LayoutLMv3 for Document QA**

                                ```python
                                from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
                                from PIL import Image
                                import torch

                                # Load model and processor
                                model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")
                                processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

                                # Load your document image
                                image = Image.open("example_document.png").convert("RGB")

                                # Your question about the document
                                question = "What is the invoice number?"

                                # Preprocess inputs (uses Tesseract OCR under the hood)
                                inputs = processor(image, question, return_tensors="pt")

                                # Inference
                                outputs = model(**inputs)
                                start_logits = outputs.start_logits
                                end_logits = outputs.end_logits

                                # Decode answer
                                start_index = torch.argmax(start_logits)
                                end_index = torch.argmax(end_logits)
                                answer = processor.tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1])

                                print("Answer:", answer)
                                ```

                                ---

                                ### 📌 Notes
                                - Replace `"example_document.png"` with the path to your document.
                                - The model uses OCR (Tesseract) under the hood to extract text and layout info.
                                - You can adapt this for key-value extraction, classification, or VQA.

                                ---

                                Want a code sample for **key-value pair extraction** or **fine-tuning** as well?
                                """,
    
}

@app.post("/qna")
def qna(json_payload:dict):

    try:
    
        session_id = json_payload.get("session_id")
        question = json_payload.get("question")
        answer = ""
        if not session_id:
            return JSONResponse(status_code=400, content={"error": "No session id provided"})
        else:
            if question in qna_dict:
                answer = qna_dict[question]
            else:
                answer = """ Sorry but I dont have the answer"""

            return JSONResponse(status_code=200, content={"summary": answer})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "issue at backend during text upload upload"})


@app.post("/generate_summary")
def upload_file(json_payload:dict):

    try:
    
        session_id = json_payload.get("session_id")
        if not session_id:
            return JSONResponse(status_code=400, content={"error": "No session id provided"})
        else:
            summary = """Here's a **Summary** of the **LayoutLMv3** research paper:

                ---

                ### **LayoutLMv3 Summary**

                **LayoutLMv3** is a multi-modal transformer model designed for **Document AI** tasks. It processes **text, layout (bounding boxes), and image data** together using a unified architecture. The model introduces **joint masked modeling**—masking both **text tokens** and **image patches** during pre-training—to better align visual and textual representations.

                **Key features:**
                - Uses a **single transformer** for all modalities.
                - Improves **text-image alignment** with unified attention.
                - Employs **2D spatial embeddings** to capture document layout.
                - Achieves **state-of-the-art** results on datasets like FUNSD, SROIE, CORD, and DocVQA.

                **Applications**: invoice extraction, form understanding, document classification, document VQA.
            """

            return JSONResponse(status_code=200, content={"summary": summary})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "issue at backend during text upload upload"})
    
@app.post("/delete_temp_file")
async def delete_temp_file(json_payload:dict):
    
    session_id = json_payload.get("session_id")
    if not session_id:
        return JSONResponse(status_code=400, content={"error": "No session id provided"})
    else:
        file_path = current_session[session_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"file deleted from path : {file_path}")

    return JSONResponse(status_code=200, content={"message": "File deleted"})

@app.post("/upload_text")
def upload_text(json_payload:dict):
    try:

        session_id = json_payload.get("session_id")
        input_text = json_payload.get("input_text")
        current_session[str(session_id)] = {
            "obj":DocSummAndQnA(),
            "text":input_text
        }

        return JSONResponse(status_code=200, content={"message": "File deleted"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "issue at backend during text upload upload"})

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), json_data:str = Form(...)):

    print("file received")

    try:

        suffix = file.filename.split('.')[-1]
        # print(suffix, file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        data = json.loads(json_data)
        session_id = data["session_id"]

        print(f"file received, session id created, {session_id}, file path: {tmp_path}")

        # if session_id in current_session:
        current_session[str(session_id)]["obj"] = DocSummAndQnA()
        current_session[str(session_id)]["file_path"] = tmp_path
        current_session[str(session_id)]["file_type"] = ""
        # else:
        #     return JSONResponse(status_code=400, content={"error": "session id not "})


        return JSONResponse(status_code=200, content={"message": "file received at backend"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "issue at backend during file upload"})



@app.get("/create_session")
def create_session():

    try:
        session_id = uuid.uuid4().hex
        current_session[str(session_id)] = {}

        return JSONResponse(status_code=200, content={"session_id": session_id})
    except Exception as e:
            return JSONResponse(status_code=400, content={"error": "issue at backend during session creation"})


if __name__ == "__main__":
    uvicorn.run("api:app", host="localhost", port=8000, reload=True, log_level="debug")
