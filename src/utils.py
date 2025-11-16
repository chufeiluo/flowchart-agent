import os, re

from pdf.PineconePDFExtractor import PdfProcessor

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from copy import deepcopy
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


url = 'localhost:4000'
endpoint = 'vector-db/upsert'

def preproc(text):
    return text.replace('“', '"').replace('”', '"').replace('’', "'").replace(u"\uFB00", 'ff').replace(u"\uFB01", 'fi').replace('|', 'I').replace(u"\u00A0", ' ').split('\t')[-1].strip()

def upload(doc, namespace='case-law'):
    # data = deepcopy(doc)
    # data['namespace'] = namespace
    try:
        # requests.post('/'.join([url, endpoint]), data=data)
        response = requests.post('/'.join(['http:/',url, endpoint]), data={'content': doc, 'collection': namespace})
        return response
    except Exception as e:
        print(e)
        return

tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " "]
)

data = []
data_dir = '../constructive_dismissal/caselaw'
extractor = PdfProcessor(1)
lowercase_reg = r'^([a-z]|[0-9]{2,})'

def process_pdf(data_dir, fn):
    pdf_text = extractor.process_files([os.path.join(data_dir, fn)])["documents"][0]["text"]
    # print(pdf_text)
    text = []
    paragraph = []
    for line in pdf_text.split('\n'):
        if 'Copyright' in line or 'page' in line.lower() or 'CanLII' in line or fn.split('.pdf')[0] in line:
            pass
        else:
            if re.match(lowercase_reg, line.strip()):
                paragraph.append(preproc(line))
            else:
                text.append(' '.join(paragraph))
                paragraph = [preproc(line)]
            # text.append(line)

    text.append(' '.join(paragraph))

    return text

def process_image(data_dir, fn):
    
    loader = UnstructuredImageLoader(os.path.join(data_dir, fn))
    data = loader.load()

    return preproc(data[0].page_content).split('\n')

def process_docx(data_dir, fn):
    
    loader = UnstructuredWordDocumentLoader(os.path.join(data_dir, fn))
    data = loader.load()

    return preproc(data[0].page_content).split('\n')

def split_docs(data_dir):
    data = []
    gold = []
    for fn in os.listdir(data_dir):
        print(fn)
        try:
            # extractor = PyPDFLoader(1)
            if fn.endswith('.pdf'):
                text = process_pdf(data_dir, fn)
            elif fn.split('.')[-1] in ['png', 'jpg', 'jpeg']:
                text = process_image(data_dir, fn)
            elif fn.split('.')[-1] in ['doc', 'docx']:
                text = process_docx(data_dir, fn)
            else:
                temp = [x for y in split_docs(os.path.join(data_dir, fn)) for x in y]
                print(temp)
                for d in temp:
                    d['name'] += '_' + fn
                data.extend(temp)
                print(data)
                continue
            # Skip `gold` population, with cleaner logic.
                # Isolates skip to this block of the for-loop instead of having continue statements at the top
            # if fn.lower().startswith('benchmark'):
            #     gold.append({'name': '.'.join(fn.split('.')[:-1]), 'content': re.sub(r'[\s]{3,}', '\n\n', '\n'.join([x.strip() for x in text])).strip()})

            # else:
            data.append({'name': '.'.join(fn.split('.')[:-1]), 'content': re.sub(r'[\s]{3,}', '\n\n', '\n'.join([x.strip() for x in text])).strip()})

        except Exception as e:
            print(f"Warning {os.path.join(data_dir, fn)}: {e}")
            continue
        
        

    return gold, data

def split_upsert(data_dir, namespace):
    data = split_docs(data_dir)

    for x in data:
        print(f'Upserting {x["name"]}' + '\n')
        # print(x['content'])

        chunked_texts = text_splitter.split_text(x['content'])
        # print(chunked_texts)
        x['court'] = ('court_of_appeals' if "COURT OF APPEAL" in x['content'] else 'supreme_court' if "SUPREME COURT" in x['content'] else 'lower_court')

        for chunk in chunked_texts:
            upload(chunk, namespace)
            

def top_k_precision(predicted, actual, k):
    """
    Calculate precision@k for string labels.
    Primarily used for evaluating retrieved vectors.
    Returns a precision@k score.
    """
    actual_set = set(actual)
    top_k_pred = predicted[:k]
    relevant_count = sum(1 for pred in top_k_pred if pred in actual_set)
    return relevant_count / k