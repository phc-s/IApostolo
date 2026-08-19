import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import errors

AVAILABLE_MODELS = [

"gemini-2.5-flash",
"gemini-3.1-flash-lite",
"gemini-3.5-flash-lite",
"gemini-2.5-flash-lite",
"gemini-3-flash",
"gemini-3.5-flash",
"gemini-3.6-flash",
"gemini-3.7-flash",

] 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILENAME = "OpenBible.pt-BR.pdf"
PDF_PATH = os.path.join(BASE_DIR, PDF_FILENAME)

PDF_TEXT_CACHE = ""

def load_pdf_context(filepath: str) -> str:
    if not os.path.exists(filepath):
        print(f"⚠️ AVISO: {filepath} não foi encontrado no caminho especificado.")
        print(f"📁 Arquivos presentes em '{BASE_DIR}': {os.listdir(BASE_DIR)}")
        return ""
    
    try:
        try:
            import pypdf as pdf_lib
        except ImportError:
            import PyPDF2 as pdf_lib

        text = ""
        with open(filepath, 'rb') as file:
            pdf_reader = pdf_lib.PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        print(f"✅ Contexto do PDF carregado em memória ({len(text)} caracteres).")
        return text
    except Exception as e:
        print(f"❌ Erro ao processar o arquivo PDF: {e}")
        return ""

@asynccontextmanager
async def lifespan(app: FastAPI):

    global PDF_TEXT_CACHE
    PDF_TEXT_CACHE = load_pdf_context(PDF_PATH)
    yield

    PDF_TEXT_CACHE = ""

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gemini-3.5-flash-lite" 

@app.get("/")
def read_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Servidor online, mas index.html não foi encontrado."}

@app.get("/download-pdf")
def download_pdf():
    if os.path.exists(PDF_PATH):
        return FileResponse(
            path=PDF_PATH, 
            filename="Torah-OpenBible.pdf", 
            media_type="application/pdf"
        )
    return {
        "error": f"Arquivo PDF não encontrado no caminho: {PDF_PATH}",
        "files_in_dir": os.listdir(BASE_DIR)
    }

@app.get("/models")
def list_models():
    return {"available_models": AVAILABLE_MODELS}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"response": "Erro: A variável GEMINI_API_KEY não foi configurada nas variáveis de ambiente."}
    
    selected_model = request.model if request.model in AVAILABLE_MODELS else "gemini-3.5-flash-lite" 

    if not PDF_TEXT_CACHE:
        return {"response": "Erro: O conteúdo do PDF não está disponível no servidor. Verifique se o arquivo está na raiz do projeto."}

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Você é um assistente especializado sobre a Bíblia/Pentateuco. Responda à pergunta do usuário estritamente com base no contexto fornecido.
        
        INSTRUÇÕES DE FORMATAÇÃO:
        - Utilize formatação Markdown para deixar a leitura agradável.
        - Use **negrito** para destacar nomes, versículos ou pontos principais.
        - Use *itálico* para termos em hebraico, citações diretas ou ênfases sutis.
        - Utilize listas com tópicos quando necessário.

        Contexto:
        {PDF_TEXT_CACHE} 

        Pergunta: {request.message}
        """

        ai_response = client.models.generate_content(
            model=selected_model,
            contents=prompt,
        )
        return {"response": ai_response.text}

    except errors.APIError as e:
        if e.code == 429:
            return {
                "response": "⚠️ O limite de requisições da API do Gemini foi atingido. Por favor, tente novamente mais tarde."
            }
        print(f"❌ Erro na API do Gemini: {e}")
        return {"response": f"Erro na API Gemini ({e.code}): {e.message}"}

    except Exception as e:
        print(f"❌ Erro inesperado no servidor: {e}")
        return {"response": "Desculpe, ocorreu um erro interno ao processar sua mensagem."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)