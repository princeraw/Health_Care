import pickle
import pandas as pd
from flask import Flask , render_template , request

model = pickle.load(open('health_care_1.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def index():
    if request.method =='GET':
        return render_template('index.html')
    else:
        form_inputs = pd.DataFrame(request.form.to_dict(),index = [0])
        prediction = model.predict(form_inputs.astype(float))
        return str(prediction)



if __name__ == "__main__":
    app.run(debug = True)