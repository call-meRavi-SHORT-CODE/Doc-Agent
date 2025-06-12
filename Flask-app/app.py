# app.py (Flask)

from flask import Flask, render_template, request
import requests




app = Flask(__name__)

FRAMEWORKS = ['LangGraph', 'AutoGen']
MODELS     = ['gpt-4o','gpt-4o-mini',"gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo", 'llama3-8b-8192','llama3-70b-8192',"llama-3.3-70b-versatile"]  
VECTORSTORES = ['Faiss', 'Chroma', 'Annoy']

@app.route('/', methods=['GET'])
def index():
    return render_template(
        'base.html', 
        frameworks=FRAMEWORKS,
        models=MODELS,
        vectorstores=VECTORSTORES,
        selected_fw=FRAMEWORKS[0],
        selected_m=MODELS[0],
        selected_vs=VECTORSTORES[0],
        prompt_text='',
        response=None
    )

@app.route('/generate', methods=['POST'])
def generate():
    # 1) Read form selections
    selected_fw = request.form.get('framework')
    selected_m  = request.form.get('model')
    selected_vs = request.form.get('vector_store')
    prompt_text = request.form.get('prompt_text', '').strip()

    # 2) Build payload to match FastAPI's RAGRequest:
    payload = {
        "framework": selected_fw.lower(),            # e.g. "langgraph"
        "llm_model": selected_m.lower(),             # e.g. "openai"
        "vector_store": selected_vs.lower(),         # e.g. "faiss"
        "query": prompt_text                 # the user’s question
    }

    try:
        api_resp = requests.post(
            "http://localhost:8000/ask",   # <-- your FastAPI endpoint
            json=payload,
            timeout=30
        )
        api_resp.raise_for_status()
        # FastAPI returns {"answer": "..."}
        response = api_resp.json().get('answer', 'No answer field returned.')

        #print(response)


        if prompt_text and response.startswith(prompt_text):
            response = response[len(prompt_text):].lstrip()

        def clean_response(text: str) -> str:
            start_index = text.find("Here")
            if start_index == -1:
                return text  # Return original if "Here" not found
            return text[start_index:].strip()
        
        response = clean_response(response)

        
            


    except Exception as e:
        response = f"❌ Error calling RAG API: {e}"

    return render_template(
        'base.html',
        frameworks=FRAMEWORKS,
        models=MODELS,
        vectorstores=VECTORSTORES,
        selected_fw=selected_fw,
        selected_m=selected_m,
        selected_vs=selected_vs,
        prompt_text=prompt_text,
        response=response
    )

@app.route('/logs')
def logs():
    return render_template('logs.html')

@app.route('/metrics')
def metrics():
    return render_template(
        'metrics.html',
        frameworks=FRAMEWORKS,
        models=MODELS,
        vectorstores=VECTORSTORES
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)