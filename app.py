from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re

tkn = AutoTokenizer.from_pretrained('tokenizer')
pipe = pipeline('summarization', model='pegasus-samsum-model', tokenizer=tkn)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text_summerization',methods=['POST'])

def summerizer():
	input_text = request.form.get('input_text_html')


	my_list = input_text.split()
	chunk_size = 500
	chunks = []
	for i in range(0, len(my_list), chunk_size):
		chunks.append(my_list[i:i + chunk_size])

	sample_text_split = chunks
	summarized = ''
	for chunck in range(len(sample_text_split)):
		gen_kwargs = {"length_penalty": 0.8, "num_beams": 8,
					  "max_length": 200}
		summarized += pipe(' '.join(sample_text_split[chunck]), **gen_kwargs)[0]['summary_text']

	summarized_text = summarized.replace('<n>', '').replace('.', '. ')

	return render_template('index.html', result = summarized_text)
	#
	# def sample_text_limit(my_list):
	# 	chunk_size = 500
	# 	chunks = []
	# 	for i in range(0, len(my_list), chunk_size):
	# 		chunks.append(my_list[i:i + chunk_size])
	# 	return chunks
	#
	# def predictor(sample_text):
	# 	sample_text_split = sample_text_limit(sample_text.split())
	# 	summarized = ''
	# 	for chunck in range(len(sample_text_split)):
	# 		gen_kwargs = {"length_penalty": 0.8, "num_beams": 8,
	# 					  "max_length": int(len(sample_text_split[chunck]) / 1.5)}
	# 		summarized += pipe(' '.join(sample_text_split[chunck]), **gen_kwargs)[0]['summary_text']
	# 	return summarized
	#
	# return render_template('index.html', result = predictor(input_text))

if __name__ == '__main__':
    app.run(host = '0.0.0.0' , port = 8000 , debug=True)