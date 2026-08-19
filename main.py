import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import errors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gemini-2.5-flash"

def load_pdf_context(filepath: str) -> str:
    if not os.path.exists(filepath):
        print(f"⚠️ AVISO: {filepath} não encontrado no servidor.")
        return ""
    try:
        import PyPDF2
        text = ""
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        print(f"❌ Erro ao ler PDF: {e}")
        return ""

@app.get("/")
def read_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Servidor online, mas index.html não foi encontrado."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"response": "Erro: A variável GEMINI_API_KEY não foi configurada no servidor."}
    
    selected_model = request.model if request.model in AVAILABLE_MODELS else "gemini-2.5-flash"

    pdf_context = load_pdf_context("OpenBible.pt-BR.pdf")
    if not pdf_context:
        return {"response": "Não tenho contexto do documento disponível no servidor."}

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Você é um assistente especializado sobre a Bíblia/Pentateuco. Responda à pergunta do usuário estritamente com base no contexto fornecido.
        
        INSTRUÇÕES DE FORMATAÇÃO:
        - Utilize formatação Markdown para deixar a leitura agradável.
        - Use **negrito** para destacar nomes, versículos ou pontos principais.
        - Use *itálico* para termos em hebraico, citações diretas ou ênfases sutis.
        - Utilize listas com tópicos (bullet points) quando necessário.

        Contexto:
        {pdf_context} 

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
                "response": "⚠️ O limite gratuito de requisições da API do Gemini foi atingido. Por favor, tente mais tarde."
            }
        print(f"❌ Erro na API do Gemini: {e}")
        return {"response": f"Erro na API Gemini ({e.code}): {e.message}"}

    except Exception as e:
        print(f"❌ Erro inesperado no servidor: {e}")
        return {"response": "Desculpe, tive um problema interno ao me conectar com a IA."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)