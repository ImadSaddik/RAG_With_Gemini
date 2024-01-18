import os
from dotenv import load_dotenv

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import chainlit as cl
from chainlit.input_widget import Slider

import google.generativeai as genai


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]


def get_relevant_passages(query, db, n_results=10):
    passages = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    return passages


def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace(
        "'", "").replace('"', "").replace("\n", " ")

    prompt = f"""question : {query}.\n
    Informations supplémentaires:\n {escaped}\n
    Si vous trouvez que la question n'a aucun rapport avec les informations supplémentaires, vous pouvez l'ignorer et répond par 'OUT OF CONTEXT'.\n
    Votre réponse :
    """

    return prompt


def convert_pasages_to_string(passages):
    context = ""

    for passage in passages:
        context += passage + "\n"

    return context


config = {
    'max_output_tokens': 128,
    'temperature': 0.9,
    'top_p': 0.9,
    'top_k': 50,
}


@cl.on_chat_start
async def start():
    setUpGoogleAPI()
    loadVectorDataBase()

    settings = await cl.ChatSettings([
        Slider(
            id="temperature",
            label="Temperature",
            initial=config['temperature'],
            min=0,
            max=1,
            step=0.1,
        ),
        Slider(
            id="top_p",
            label="Top P",
            initial=config['top_p'],
            min=0,
            max=1,
            step=0.1,
        ),
        Slider(
            id="top_k",
            label="Top K",
            initial=config['top_k'],
            min=0,
            max=100,
            step=1,
        ),
        Slider(
            id="max_output_tokens",
            label="Max output tokens",
            initial=config['max_output_tokens'],
            min=0,
            max=1024,
            step=1,
        )
    ]).send()

    await setup_model(settings)


@cl.on_settings_update
async def setup_model(settings):
    config['temperature'] = settings['temperature']
    config['top_p'] = float(settings['top_p'])
    config['top_k'] = int(settings['top_k'])
    config['max_output_tokens'] = int(settings['max_output_tokens'])
    
    model = genai.GenerativeModel(model_name="gemini-pro", generation_config=config)
    cl.user_session.set('model', model)
    

def setUpGoogleAPI():
    load_dotenv()

    api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=api_key)


def loadVectorDataBase():
    chroma_client = chromadb.PersistentClient(path="../database/")

    db = chroma_client.get_or_create_collection(
        name="sme_db", embedding_function=GeminiEmbeddingFunction())

    cl.user_session.set('db', db)
    
    
@cl.on_message
async def main(message):
    model = cl.user_session.get('model')
    
    question = message.content
    db = cl.user_session.get('db')
    passages = get_relevant_passages(question, db, 5)
    
    prompt = make_prompt(message.content, convert_pasages_to_string(passages))
    
    answer = model.generate_content(prompt)
    await cl.Message(content=answer.text).send()
    
    
    