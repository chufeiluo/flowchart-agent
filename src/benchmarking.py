from flowchart import load_flowchart, init
import json, os
import dotenv
import time
import argparse

import pickle as pkl

from tqdm import *


def format_flowchart_results(results):
    """Formats the list of FlowchartTaskResults into a human-readable version."""
    formatted = []
    for i, item in enumerate(results):
            formatted.append(f"Result {i+1}:")
            for key, value in item.items():
                if isinstance(value, list):
                    formatted.append(f"  {key}:")
                    for v in value:
                        formatted.append(f"    {v}")
                else:
                    formatted.append(f"  {key}: {value}")
            for i in item['result']: 
                if "eval" in i["executionDetails"]:
                    formatted.append("eval:")
                    formatted.append(json.dumps(i["executionDetails"]["eval"], indent=4))
            formatted.append("")  # Empty line between results
    return "\n".join(formatted)

def example_test(input_file, flowchart_file, model, n=-1):
    fc = load_flowchart(flowchart_file)
    init(model)

    if input_file.endswith('jsonl'):
        with open(input_file, 'r') as f:
            data = [json.loads(x) for x in f.readlines()] # this example assumes the input is already in jsonl format

    res = []
    if n == -1:
        n = len(data)

    
    for d in tqdm(data[:n]):
        result = fc.execute(d)
        # print(result[-1])
        # result[0]['gold_label'] = d['gold_label'][0]
        gold_label = d['gold_label'][0] if 'gold_label' in d and len(d['gold_label']) > 0 else "" 
        res.append({'result': [x.__dict__() for x in result], 'gold_label': gold_label}) 

    return res

if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help = "model name", default='gpt-4o-mini')
    parser.add_argument("-i", "--input", help = "input file name", default='src/data/cd.jsonl')
    parser.add_argument("-f", "--flow", help = "flowchart", default='src/flowcharts/rag.json')
    parser.add_argument("-k", "--topk", help = "top k", type=int, default=10)
    parser.add_argument("-o", "--out", help = "output folder", default='src/data/results')

    args = parser.parse_args()

    models = ['gpt-4o', 'o3', 'gpt-4o-mini'] 
    for m in models:
        out_file = f"{args.out}/{args.input.split('/')[-1].split('.')[0]}_{m}_{args.flow.split('/')[-1].split('.')[0]}.json"
        os.makedirs("data/results", exist_ok=True)
        if not os.path.exists(out_file):
            res = example_test(args.input, args.flow, m, args.topk)
            # print(res)

            try:
                with open(out_file, 'w') as f:
                    # print(res)
                    f.write(json.dumps(res, indent=4))
            except Exception as e:
                print(e)
                with open(out_file.split('.')[0] + '.pkl', 'wb') as f:
                    pkl.dump(res, f)
                with open(out_file.split('.')[0] + '.log', 'w', encoding="utf-8") as f: # Human-readable version
                    f.write(format_flowchart_results(res))
                print("Wrote pkl and log files instead!")

            # time.sleep(600)
        else:
            print(f"{out_file} already exists! Skipping this test...")