from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer

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

if __name__ == '__main__':
    app.run(host = '0.0.0.0' , port = 8000 , debug=True)