from typing import List, Callable, Any, Tuple, Dict
import json

import random, re

from globals import REQUESTS_MAPPING, OUTPUT_FORMAT_MAPPING, PROVIDER_MAPPING

from langchain.chat_models import init_chat_model

# from langchain_community.llms import OpenAI # For OpenAI's non-chat models
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


llm = None

def init(model_name, t=1):
    global llm

    llm = init_chat_model(
            model_name,
            model_provider=PROVIDER_MAPPING[model_name],
            temperature=t
            )



class FlowchartTask:
    def __init__(self, inp):
        self.type = inp['type']
        self.name = inp['name']
        if self.type == 'api':
            if 'url' not in inp or 'endpoint' not in inp:
                pass
            else:                
                self.url = inp['url']
                self.endpoint = inp['endpoint']
                self.metadata = inp.get('metadata', {}) # one of these options: concat, topk, random
        elif self.type == 'llm':
            self.instructions = inp['instructions']
            self.inputFormat = inp.get('input_format', None)
            self.outputFormat = inp.get('output_format', None)

    def __repr__(self):
        return f"FlowchartNode(name={self.name.__repr__()}, instructions={self.instructions.__repr__()}, inputFormat={self.inputFormat.__repr__()}, outputFormat={self.outputFormat.__repr__()})"

    def to_prompt(self, input, input_reg = '[input]', doc_reg='[docs]'):
        """
        Convert a FlowchartTask to a prompt string.
        """
        assert self.type == 'llm', 'node type mismatch, must be an llm call'

        inp_form = f'Input Format: {self.inputFormat}'
        out_form = f'Output Format: {self.outputFormat}'    

        keys = re.findall(r'(\[(.*?)\])', self.instructions)
        # print(keys)
        instructions = self.instructions
        for k in keys:
            # print(k[0], input[k[1]].value)
            # print('**************************************************************************\n\n\n\n')
            instructions = instructions.replace(k[0], input[k[1]].value)

        return f"""{instructions}
    {(inp_form if self.inputFormat is not None else '')}
    {(out_form if self.outputFormat is not None else '')}
    """

class FlowchartTaskResult:
    def __init__(self, value: Any, executionDetails: Dict):
        self.value = value
        self.executionDetails = executionDetails

    def __dict__(self):
        if 'promptMessages' in self.executionDetails:
            execution_details = {
                **self.executionDetails,
                'promptMessages': [x.content for x in self.executionDetails['promptMessages']]
            }
        else:
            execution_details = self.executionDetails

        return {'value': self.value, 'executionDetails': execution_details}
    
    def __repr__(self):
        return f"FlowchartNodeOutput(value={self.value.__repr__()}, executionDetails={self.executionDetails.__repr__()})"
    
    

class DebugMessage: # LLM func. debug class - use when testing functionality of new benchmarking features
    def __init__(self, content):
        self.content = content
        
    def __dict__(self):
        return {'content':self.content}
        
# This is a linear flowchart structure, the nodes are connected in a sequence.
class Flowchart:
    # nodes: List[FlowchartTask]

    def __init__(self, nodes: List[FlowchartTask]):
        self.nodes = nodes


    def llm_execution(self, node: FlowchartTask, input) -> FlowchartTaskResult:
        """
        Execute a FlowchartNode and return the output.
        This function simulates the execution of a flowchart node using a basic LLM call.

        currently, the audit is just the prompt and the response from the LLM.
        """
        assert isinstance(
                node, FlowchartTask), "node must be a FlowchartTask instance"
        
        if 'supporting_docs' in input:            
            input['docs'] = FlowchartTaskResult(value='\n'.join(
                ['{0}: {1}'.format(x['name'], x['content']) for x in input['supporting_docs'].value]
            ), executionDetails={})
        messages = [
            SystemMessage(content="You are a helpful assistant. Please answer the user's input as factually as possible."),
            HumanMessage(content=node.to_prompt(input)),
        ]
        count = 0
        while count < 3:
            try:
                # Comment out llm line and uncomment Debug one if testing new feaures (to avoid unnecessary LLM calls)
                response = llm.invoke(messages)
                # response = DebugMessage('{"answer": "response", "analysis": "something"}').__dict__()  

                # print(response.content, re.sub(r'\`\`\`.*', '', response.content))
                temp = json.loads(re.sub(r',(?=\n})', '', re.sub(r'\`\`\`.*', '', response.content)).strip().replace('\n', ''))

                break
            except json.decoder.JSONDecodeError as e:
                print('retrying')
                count += 1

        assert temp, "llm_execution did not get a good response within 3 attempts."
        return FlowchartTaskResult(
            value=f"{temp['answer']}\n{temp['analysis']}", 
            executionDetails={"promptMessages": messages, "original": temp, "response": response}
        )

    def api_execution(self, node: FlowchartTask, input, headers={}) -> FlowchartTaskResult:
        if node.url in REQUESTS_MAPPING:
            # print(input)
            text, res, eval = REQUESTS_MAPPING[node.url](node, input, headers)
        else:
            return "Not supported"
        

        out = OUTPUT_FORMAT_MAPPING[node.url](text)
        
        return FlowchartTaskResult(
            value='\n**********\n'.join(out), 
            executionDetails={'full_response': res, 'input': input, 'eval': eval}
        )


    def execute(self, input: Any) -> List[FlowchartTaskResult]:
        """
        Execute a Flowchart and return the output.
        This function .
        """
        assert isinstance(
            self.nodes, list), "flowchart must be a list of FlowchartTask instances"
        
        # if len(input['supporting_docs'][1]) > 1:
        #     print(f"Supporting docs: elem 0 (golden label) has {len(input['supporting_docs'][0])} elements")
        #     print(f"Supporting docs: elem 1 (supporting docs) has {len(input['supporting_docs'][1])} elements, each elem is a {type(input['supporting_docs'][1][0])}")
            
        results = {'supporting_docs': FlowchartTaskResult(value=input['supporting_docs'][1], executionDetails={}),
                    'input': FlowchartTaskResult(value=input['query'], executionDetails={}),
                    }
        for node in self.nodes:
            if node.type == 'llm':
                results[node.name] = self.llm_execution(node, results)
            elif node.type == 'api':
                # data = {'query': results[-1].value, 'collection': 'constructive_dismissal'}
                results[node.name] = self.api_execution(node, results)

        return list(results.values())  # return all outputs, except for the initial input as the first element


def load_flowchart(filename: str):
    with open(filename, 'r') as f:
        data = json.load(f)

    return Flowchart([FlowchartTask(x) for x in data])