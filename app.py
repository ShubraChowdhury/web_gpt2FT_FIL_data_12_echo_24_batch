
import numpy as np
from flask import Flask, request, render_template
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

#Create an app object using the Flask class. 
app = Flask(__name__)


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(sequence, max_length):
    model_path = 'outputDir/'
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    #print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))



#Load the trained model. (Pickle file)

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
    
def predict():

    #int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    #features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    #prediction = model.predict(features)  # features Must be in the form [[a, b]]
    input_text = [x for x in request.form.values()] 
    model_path = 'outputDir/'
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{input_text[0]}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=int(input_text[1]),
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )

    #print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
    

    output = generate_text(input_text[0],int(input_text[1]))

    return render_template('index.html', prediction_text='Generated text is...............:  {}'.format(tokenizer.decode(final_outputs[0], skip_special_tokens=True)))


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5011)
