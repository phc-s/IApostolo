import os
import asyncio
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

GEMINI_PDF_FILE = None
ACTIVE_PDF_PATH = None

def find_pdf_file() -> Optional[str]:
    """Busca o PDF na raiz, tratando diferenças de maiúsculas/minúsculas."""
    exact_path = os.path.join(BASE_DIR, "OpenBible.pt-BR.pdf")
    if os.path.exists(exact_path):
        return exact_path

    for file in os.listdir(BASE_DIR):
        if file.lower().endswith(".pdf"):
            return os.path.join(BASE_DIR, file)
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GEMINI_PDF_FILE, ACTIVE_PDF_PATH
    api_key = os.environ.get("GEMINI_API_KEY")
    ACTIVE_PDF_PATH = find_pdf_file()

    if not api_key:
        print("⚠️ AVISO: GEMINI_API_KEY não foi configurada nas variáveis do Render.")
    elif not ACTIVE_PDF_PATH:
        print(f"⚠️ AVISO: Nenhum arquivo .pdf foi encontrado em '{BASE_DIR}'.")
        print(f"📁 Conteúdo da pasta no Render: {os.listdir(BASE_DIR)}")
    else:
        try:
            client = genai.Client(api_key=api_key)
            print(f"⏳ Fazendo upload do PDF '{os.path.basename(ACTIVE_PDF_PATH)}' para a API do Gemini...")
            GEMINI_PDF_FILE = client.files.upload(file=ACTIVE_PDF_PATH)
            print(f"✅ PDF carregado com sucesso na API do Gemini: {GEMINI_PDF_FILE.name}")
        except Exception as e:
            print(f"❌ Erro ao fazer upload do PDF para o Gemini: {e}")

    yield
    GEMINI_PDF_FILE = None

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

@app.get("/OpenBible.pt-BR.pdf")
def serve_pdf_direct():
    pdf_path = find_pdf_file()
    if pdf_path and os.path.exists(pdf_path):
        return FileResponse(
            path=pdf_path, 
            filename="OpenBible.pt-BR.pdf", 
            media_type="application/pdf"
        )
    return {
        "error": "Arquivo PDF não encontrado na raiz do servidor.",
        "arquivos_na_raiz": os.listdir(BASE_DIR)
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

    if not GEMINI_PDF_FILE:
        return {"response": f"Erro: O arquivo {PDF_FILENAME} não está pronto ou não foi encontrado no servidor."}

    client = genai.Client(api_key=api_key)

    prompt = f"""
    Você é um assistente especializado sobre a Bíblia/Pentateuco. Responda à pergunta do usuário estritamente com base no documento fornecido.
    
    INSTRUÇÕES DE FORMATAÇÃO:
    - Utilize formatação Markdown para deixar a leitura agradável.
    - Use **negrito** para destacar nomes, versículos ou pontos principais.
    - Use *itálico* para termos em hebraico, citações diretas ou ênfases sutis.
    - Utilize listas com tópicos quando necessário.

    Pergunta: {request.message}
    """

    # Retry automático para tratar o estouro de RPM (Erro 429)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ai_response = client.models.generate_content(
                model=selected_model,
                contents=[GEMINI_PDF_FILE, prompt],
            )
            return {"response": ai_response.text}

        except errors.APIError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                print(f"⚠️ Limite de requisições (429) atingido. Aguardando {wait_time}s antes da tentativa {attempt + 2}...")
                await asyncio.sleep(wait_time)
                continue
            
            print(f"❌ Erro na API do Gemini: {e}")
            if e.code == 429:
                return {"response": "⚠️ O limite de requisições (RPM) foi atingido. Por favor, aguarde alguns segundos e tente novamente."}
            return {"response": f"Erro na API Gemini ({e.code}): {e.message}"}

        except Exception as e:
            print(f"❌ Erro inesperado no servidor: {e}")
            return {"response": "Desculpe, ocorreu um erro interno ao processar sua mensagem."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)