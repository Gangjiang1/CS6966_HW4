'''
Code source (with some changes):
https://levelup.gitconnected.com/huggingface-transformers-interpretability-with-captum-28e4ff4df234
https://gist.githubusercontent.com/theDestI/fe9ea0d89386cf00a12e60dd346f2109/raw/15c992f43ddecb0f0f857cea9f61cd22d59393ab/explain.py
'''

import torch
import pandas as pd

from torch import tensor 
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr import IntegratedGradients, Saliency
import matplotlib.pyplot as plt

import argparse 
import jsonlines
import os 

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    
    
    def forward_func(self, inputs: tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
        
    def visualize(self, inputs: list, attributes: list, outfile_path: str):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        #import pdb; pdb.set_trace()
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        
        a = pd.Series(attr.cpu().numpy()[0][::-1], 
                         index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1])
        
        a.plot.barh(figsize=(10,20))
        plt.savefig(outfile_path)
    # Calculate negtive and positive words parse below
    def count_negative_words:

    def count_positive_words:

    def explain(self, text: str, outfile_path: str):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])

        # Saliency method
        saliency_attributor = Saliency(self.forward_func)
    
        saliency_attributes, _ = saliency_attributor.attribute(inputs=inputs, target=self.__pipeline.model.config.label2id[prediction[0]['label']])

        # Integrated Gradients
        ig_attributor = IntegratedGradients(self.forward_func, getattr(self.__pipeline.model, 'deberta').embeddings)
        ig_attributes, delta_ig = lig.attribute(inputs=inputs,
                                  baselines=baseline,
                                  target = self.__pipeline.model.config.label2id[prediction[0]['label']], 
                                  return_convergence_delta = True)
        # Set weights
        # Calculate text length, negative words and positive words counts
        text_length = len(text.split())
        negative_word_count = count_negative_words(text)
        positive_word_count = count_positive_words(text)

        # Define initial weights
        weight_saliency = 0.5
        weight_integrated_gradients = 0.5

        # Adjust weights based on text length and word counts
        if text_length > threshold_length:
            # If the text is long, prioritize Saliency
            weight_saliency = 0.7
            weight_integrated_gradients = 0.3
        elif negative_word_count > threshold_length and positive_word_count > threshold_length:
            # If there are more parse of words, prioritize Saliency
            weight_saliency = 0.7
            weight_integrated_gradients = 0.3

        # Combined those two methods
        combined_attributes = weight_saliency * saliency_attributes + weight_integrated_gradients * ig_attributes

        self.visualize(inputs, combined_attributes, outfile_path)

    
    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        input_ids = self.__pipeline.tokenizer.encode(text, add_special_tokens=False)
        #inputs = self.__pipeline.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True, truncation=True).to(self.__device, dtype=torch.float32)
        inputs = self.__pipeline.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True, truncation=True).to(self.__device, dtype=torch.long)

        inputs.requires_grad = True

        #inputs = torch.tensor(input_ids, dtype=torch.float32, device=self.__device, requires_grad=True).unsqueeze(0)
        return inputs
    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        baseline_ids = [self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id]
        
        baseline = torch.tensor(baseline_ids, dtype=torch.float32, device=self.__device).unsqueeze(0)
        
        return baseline
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint) 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=device
                                )
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, device)

    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain(obj["review"], os.path.join(args.output_dir,f'example_{idx}'))
            idx+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analsis_dir', default='out_hw4', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='out/a1_analysis_data.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out_hw4', type=str, help='Directory where model checkpoints will be saved')    
    args = parser.parse_args()
    main(args)
