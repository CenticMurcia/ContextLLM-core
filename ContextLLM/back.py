from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain.vectorstores import VectorStore 
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from langchain_core.documents.base import Document
from langchain.load import dumps, loads
from langchain_core.output_parsers import JsonOutputParser
import time
import shutil
import os 

#eliminar carpeta para que no haya conflicto con los ids
if os.path.exists('./DATA/chroma_db'):
    shutil.rmtree('./DATA/chroma_db')

#export OLLAMA_HOST=0.0.0.0:11435

llm = Ollama(base_url="http://10.7.15.205:7869",
             model="mistral:7b",
             verbose=True,  
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

hugg_embeddings = HuggingFaceEmbeddings(model_name= "mxbai-embed-large-v1")

hugg_emb_bgem3 = HuggingFaceEmbeddings(model_name = "BAAI/bge-m3")

myFile = 'annualreport.pdf'
loader = PyPDFLoader("DATA/" + myFile)

data = loader.load()

vectorstore = Chroma(
        collection_name="documents",
        persist_directory="./DATA/chroma_db",
        embedding_function=hugg_embeddings
    )

store = InMemoryStore()

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    length_function = len
)

child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        length_function=len
    )

full_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

full_retriever.add_documents(data, ids = None)

vectorstore_bgem3 = Chroma(
        collection_name="documents",
        persist_directory="./DATA/chroma_db",
        embedding_function=hugg_emb_bgem3
    )

full_retriever_bge = ParentDocumentRetriever(
    vectorstore=vectorstore_bgem3,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

full_retriever_bge.add_documents(data, ids = None)

'''
Explicación del retriever, en particular la función _get_relevant_documents() para saber como usar el retriever cuando se quieran añadir más 
condiciones en el rrf: 
Esta clase está estructurada para que con las funciones rrf() y get_fused_scores() se puedan ir añadiendo tantos metodos de recuperación de 
contexto como se quiera. (Siempre que los documentos recuperados tengan un orden concreto, para los que no, ya está el método fused_scores_literal()
que asigna a todos los documentos el mismo valor). 
¿Cómo funciona y se puede automatizar la función _get_relevant_documents()?
Supongamos que tenemos n criterios de recuperación de contextos (en este ejemplo tenemos 4, literal, mxbai l2, mxbai cos, bge cos)
En primer lugar, se necestita ir calculando los scores de los documentos que vamos obteniendo para ello, se va haciendo uso de la función get_fused_scores()
scores1 =  get_fused_scores({},lista1), scores2 = get_fused_scores(scores1,lista2), ... , scoresn = get_fused_scores(scores(n-1), listan)
Esto lo que va haciendo es ir acumulando los scores de acuerdo al valor que se ha decidido dar. En este caso, 1/1+rank. 
Por ultimo una vez tengamos todos los documentos con sus scores se hace uso de la función rrf con los ultimos scores, que se encarga de dar la lista final de 
documentos ordenada sin los scores ya para que la procese la función del retriever. 
rrf(scores(n)) puesto que esta función desde dentro ya se encarga de calcular los scores para la ultima lista.
'''
class CustomRetriever_advanced(BaseRetriever):
    '''En este caso se necesitan dos db vectoriales que almacenen los embeddings que genera cada modelo diferente. Se pueden usar
    tantos modelos de generación de embeddings como se quiera simplemente habrá que pasar en esta clase del retriever personalizado 
    las bases de datos correspondientes. También se va a añadir una variable k que será el número de documentos que se quieran recuperar
    Por ejemplo si k = 10 pues el retriever generará un contexto con los 10 documentos que mayor score hayan generado. En el caso anterior 
    que k no se especificaba, en los documentos se metían todos los que se iban recuperando de cada método. '''
    vs: VectorStore
    vs_2: VectorStore
    k: int
    '''fused_scores es un diccionario que va a ir actualizando la solucion de rrf por lo que en la primera
    iteración fused_scores = {} (diccionario vacio) mientras que para las demás iteraciones será la solución
    de esta función junto con las diferentes listas que queramos ir pasandole el rrf'''
    
    '''
    Cada vez que se hace una llamada a la función rrf nos da una lista de objetos Document (sin el score) que están ya listos 
    para pasar al retrievalQA. Es necesario pasarle la lista de scores que se va obteniendo conforme se van usando técnicas en
    el reranking. 
    '''
    def rrf(self,fused_scores):
        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        #falta pasar estos resultados a una lista de documentos que es lo que devuelve el retriver (loads)
        lista_rerank = []
        print('Documentos finales (score)')
        for doc, score in reranked_results.items():
            lista_rerank.append(loads(doc))
            print('------')
            print(score)
        return lista_rerank
    
    '''
    La función get_fused_scores, nos sirve para ir actualizando los scores de una lista de documetnos dada. Por ejemplo, si estamos 
    en la primera iteración del rrf, previus_fused = {} y esta función devolverá un diccionario actualizado con los documetnos de la lista 
    y su correspondiente score. Este diccionario será necesario para pasarselo luego en la función rrf. 
    Cuando no estamos en la primera iteración y previus_fused ya no es un diccionario vacio, se le pasará un diccionario de documentos y 
    scores junto con una nueva lista que tendrá los documentos que queremos incorporar a la lista de scores. 
    '''   
    def get_fused_scores(self,previus_fused, lista):
        for rank, doc in enumerate(lista):
            doc_str = dumps(doc)
            if doc_str not in previus_fused:
                previus_fused[doc_str] = 0
            previus_score = previus_fused[doc_str]
            print(f'scores previos {previus_score}')
            previus_fused[doc_str] += 1 / (rank + 1)
        return previus_fused
    
    '''
    Este método realiza la misma función que la anterior solo que en esta se ha utilizado para arreglar el problema de tener doc_id diferente para el
    mismo chunk en dos bases de datos. Lo que hace es ir metiendo los documentos en una lista y actualizar el metadata de los nuevos que van entrando si 
    ya se encuentran en la lista. Asi solucionamos el tema de la duplicidad, el resto de funcionamiento es idéntico.
    '''
    def get_fused_scores_v2(self,previus_fused, lista):
        
        previus_list = []
        for doc, score in previus_fused.items():
            previus_list.append(loads(doc))
        
        for d in lista:
            for p_d in previus_list:
                if d.page_content == p_d.page_content: 
                    d.metadata['doc_id'] = p_d.metadata['doc_id']

        for rank, doc in enumerate(lista):
            doc_str = dumps(doc)
            if doc_str not in previus_fused:
                previus_fused[doc_str] = 0
            previus_score = previus_fused[doc_str]
            print(f'scores previos {previus_score}')
            previus_fused[doc_str] += 1 / (rank + 1)

        return previus_fused


    '''
    Nota: Esta función no haría falta si no se le diera la misma puntuación a cada documento obtenido en la búsqueda literal. Esto se debe a que 
    cuando se buscan documentos de forma literal todos han de tener el mismo valor. No se puede distinguir de esta manera si hay uno "mejor" que otro. 
    Como aqui tenemos fused_scores = {}, esto significa que si queremos usar esta estrategia siempre tiene que ser los primeros scores que se recopilan, 
    si no hay que cambiar y pasarle como argumetno unos scores previos y acumular con += 0.5
    En resumen no es necesario aqui comprobar lo de la lista auxiliar y cambiar el metadata de doc_id ya que en este caso serían los primeros documentos 
    que entrarían en la lista y en ningún caso estarían duplicados. 
    '''
    def fused_scores_literal(self, lista):
        fused_scores = {}
        for doc in lista:
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previus_score = fused_scores[doc_str]
            print(f'scores previos {previus_score}')
            fused_scores[doc_str] = 0.5
        return fused_scores
    ''' 
    Genera una consulta para que sea dinámica la busqueda literal en la base de datos, dada una lista de palabras clave. 
    El método de busqueda literal en la base de datos chroma, necesita el siguiente formato
    {
    "$and": [
        {"$contains": "key_word1"},
        {"$contains": "key_word2"}
        ]
    }
    Con lo cual, este método en función de las palabras clave que tenga nuestra query, se van añadiendo a la consulta y no se
    necesita ir creando una consulta a mano para cada búsqueda literal que tenga diferente numero de palabras clave. 
    '''
    def generaConsulta(self,key_words):
        busquedas = []
        for w in key_words:
            conta = {"$contains":str(w)}
            busquedas.append(conta)
        #Duda: Poner un or en vez de un and. En el caso de que existan muchos keywords puede ser dificil encontrar documentos que los contengan todas. 
        consulta = {"$and": busquedas}
        return consulta
    
    '''Corrective RAG:
    La función CRAG, recibe una query (pregunta) y una serie de documentos que han sido recuperados con el objetivo de responder la pregunta
    de forma correcta. En este caso, se crea un LLM que sea capaz de evaluar cada uno de estos documentos y decida si su contenido es adecuado
    para responder la pregunta. Los documentos que se consideren adecuados son aquellos que se devuelven en esta función. 
    '''
    def CRAG(self, query, docs):

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
            input_variables=["question", "document"],
        )

        retrieval_grader = prompt | llm | JsonOutputParser()

        valid_docs = []

        for d in docs:
            score = retrieval_grader.invoke({"question": query, "document": d.page_content})
            if score['score'] == "yes":
                valid_docs.append(d)
            else:
                continue

        return valid_docs
    
    '''
    En el RetrievalQA, que se pasa en la cadena, el retriever que se usa por la clase para obtener los diferentes documentos hace uso de una función
    llamada get_relevant_documents() que viene predeterminada en el retriever base. Cuando hacemos un retriever personalizado, es necesario crear una
    función _get_relevant_documents(), con _ delante del mismo nombre anterior para que a la hora de recuperar documentos la función RetrievalQA haga 
    uso de la función que hemos personalizado para recuperar los documentos. Solo tenemos que tener en cuenta que al aplicar los cambios que queremos
    al personalizar esta función, se devuelva una lista de documentos (objetos Document)
    '''
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
        #Quitar la parte de spacy pq da error de dependencias, esto es la parte literal !!
        '''
        spacyModel = spacy.load("en_core_web_sm")
        list = self.vs.get(
            where_document=self.generaConsulta(spacyModel(query).ents)
            )
        
        literal_docs = []
      
        for i in range(len(list['ids'])):
            doc = Document(page_content=list['documents'][i],metadata=list['metadatas'][i])
            literal_docs.append(doc)
        '''
        
        #documentos
        docs_l2 = self.vs.similarity_search(query)

        docs_simcos = self.vs.similarity_search_by_vector(hugg_embeddings.embed_query(query))
        
        docs_simcos_bge = self.vs_2.similarity_search_by_vector(hugg_emb_bgem3.embed_query(query))
        #Scores acumulados de cada haciendo uso de cada tecnica
        #scores_literal = self.fused_scores_literal(literal_docs)

        scores_l2 = self.get_fused_scores_v2({},docs_l2)
        
        scores_simcos = self.get_fused_scores_v2(scores_l2,docs_simcos)

        scores_bge = self.get_fused_scores_v2(scores_simcos,docs_simcos_bge)
        #Recuperación de documentos
        rrf_documents = self.rrf(scores_bge)

        #evaluación de los documentos con CRAG.
        crag_documents = self.CRAG(query,rrf_documents)

        ids = []
        for d in crag_documents: 
            if d.metadata['doc_id'] not in ids: 
                ids.append(d.metadata['doc_id'])
        print('ids: ')
        #print(ids)
        print('--------------------------------')
        print('store ids:')
        #print(list(store.yield_keys()))
        print('--------------------------------')
        #Los ids se llenan y aparecen bien, el problema es que store.mget(ids) no devuelve nada. Store vacia? 
        parent_docs = store.mget(ids)
        #Aqui parent_docs es un array de [None]
        #De momento, str_docs y docs acaban vacios. 
        str_docs = []
        docs = []
        for pd in parent_docs:
            parent_doc_str = dumps(pd)
            if parent_doc_str not in str_docs:
                if pd is not None:
                    str_docs.append(parent_doc_str)
                    docs.append(loads(parent_doc_str))

        return docs[0:self.k]
    

template = """
    You are a knowledgeable chatbot, here to help with questions of the user.
    Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:
    It is mandatorian that only if the answer is not in the context, answer "I have not enough context in order to answer this" and stop the answer.
    Try to use the memory context in the answer only if the question mentions it.
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question"
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=CustomRetriever_advanced(vs = vectorstore, vs_2=vectorstore_bgem3, k=4),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": memory,
    }
)

#Para generar respuestas
def answer(query):
    return qa_chain(query)['result']

#Tratar la salida de texto por el front: (recibe una cadena de texto la separa y va dando las palabras poco a poco)
def modify_output(input):
    for text in input.split():
        yield text + " "
        time.sleep(0.05)