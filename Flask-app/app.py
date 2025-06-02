from flask import Flask, render_template, request
import requests

app = Flask(__name__)

FRAMEWORKS = ['LangChain', 'CrewAI', 'LlamaIndex']
MODELS = ['OpenAI', 'Anthropic', 'Cohere']
VECTORSTORES = ['Chroma', 'Weaviate', 'Pinecone']

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
    selected_fw = request.form.get('framework')
    selected_m = request.form.get('model')
    selected_vs = request.form.get('vector_store')
    prompt_text = request.form.get('prompt_text', '').strip()

    payload = {
        "framework": selected_fw,
        "model": selected_m,
        "vector_store": selected_vs,
        "prompt": prompt_text
    }

    try:
        # ✅ Corrected URL to match FastAPI
        api_resp = requests.post(
            "http://localhost:8000/api/rag-query",
            json=payload,
            timeout=10
        )
        api_resp.raise_for_status()
        response = api_resp.json().get('answer', 'No answer field returned.')
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