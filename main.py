import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Função de leitura segura tratada dentro das requisições
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
    # Verifica se o index.html existe para evitar crash no GET /
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Servidor online, mas index.html não foi encontrado."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"response": "Erro: A variável GEMINI_API_KEY não foi configurada no Render."}
    
    # Lê o PDF apenas quando necessário
    pdf_context = load_pdf_context("OpenBible_pt-BR_Proverbios.pdf")
    if not pdf_context:
        return {"response": "Não tenho contexto do documento disponível no servidor."}

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Você é um assistente sobre a Bíblia. Responda à pergunta do usuário estritamente com base no contexto:

        Contexto:
        {pdf_context[:20000]} 

        Pergunta: {request.message}
        """

        ai_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return {"response": ai_response.text}
    except Exception as e:
        print(f"Erro na API da Gemini: {e}")
        return {"response": "Desculpe, tive um problema ao me conectar com a IA."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
