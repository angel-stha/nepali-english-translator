
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
app = Flask(__name__)

model = load_model('model')
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])

def predict():
     '''
     For direct API calls trought request
     '''
    int_features = request.form.values()
    lines = int_features.strip().split('\n')
    list1 = [] 
    for i in range(len(lines)):
        a=(len(lines[i].split(' ')))
        list1.append(a)
      
    tokenizer = Tokenizer()
    token=tokenizer.fit_on_texts(lines)
    
     prediction = model.predict_classes(final_features)
     def get_word(n, tokenizer):
          for word, index in tokenizer.word_index.items():
              if index==n:
                 return word
          return None
    def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq
     preds_text = []
     for i in prediction:
         temp = []
         for j in range(len(i)):
             t = get_word(i[j], eng_tokenizer)
             if j > 0:
                 if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                     temp.append('')
                 else:
                     temp.append(t)
              
             else:
                 if(t == None):
                     temp.append('')
                 else:
                     temp.append(t)            
         
     preds_text.append(' '.join(temp))
     output = preds_text
     return render_template('index.html', prediction_text='Translation {}'.format(output))


    
 

   
if __name__ == "__main__":
    app.run(debug=True)