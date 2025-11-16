import requests, random ,re
from utils import top_k_precision

def google_form(results):
    return [f"Title: {x['title']}\n\nSnippet: {x['snippet']}" for x in results]


def vectordb_form(results):
    # print(results[0])
    # return [f"Relevance: {x['score']}\nDocument: {re.sub(r'[\s]{2,}', ' ', x['payload']['text'])}" for x in results]
    out = []
    for x in results:
        doc = re.sub(r'[\s]{2,}', ' ', x['payload']['text'])
        out.append(f"Relevance: {x['score']}\nDocument: {doc}")
    return out

def db_retrieval(node, input, headers):
    if 'query' in node.metadata:
        # data = input.get(node.metadata['query'], None)
        # if data:
        #     data = data.value
        # data = {'query': data}
        data = node.metadata['query'] if 'query' in node.metadata else None
        response = requests.post(url=''.join(['http://', node.url, node.endpoint]), data=data, headers=headers)

        
        res = response.json()
        text = []
        output_strat = node.metadata['output_strat']
        if output_strat == 'concat':
            text = res
        elif output_strat.startswith('top'): # top k, where k is < 10
            text.extend(res[:int(output_strat[-1])]) 
        elif output_strat == 'random': # random k, where k is < 10
            text.append(random.sample(res, k=int(output_strat[-1])))
            
        eval = { "enabled?": False }
        k = int(output_strat[-1])
        if "metrics" in node.metadata and len(node.metadata["metrics"]) > 0:
            original_docs = [doc["name"].rsplit('.', 1)[0] for doc in input['supporting_docs'].value]
            topk_docs = [vec["payload"]["source"] for vec in res]
            
            eval["enabled?"] = True
            eval["predicted"] = topk_docs
            eval["actual"] = original_docs
            
            for metric in node.metadata["metrics"]:
                if metric == "precision":
                    # Vector docs in DB don't have file endings, so we drop file endings from supporting docs here
                    eval[f"precision@{k}"] = top_k_precision(topk_docs, original_docs, k)
                    
        return text, res, eval
    else:
        return []

OUTPUT_FORMAT_MAPPING = {
    'google.com': google_form, 
    'localhost:4000': vectordb_form, 
}

REQUESTS_MAPPING = {
    'localhost:4000': db_retrieval,
}

EVAL_METRICS = {
    'example.json': ['bleu']
}

PROVIDER_MAPPING = {
    'gpt-4o-mini': 'openai',
    'gpt-4o': 'openai',
    'o3': 'openai',
    'o3-deep-research': 'openai',
    "claude-3-5-sonnet-latest": 'anthropic',

}
