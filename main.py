# -*- coding: utf-8 -*-

from topic_modelling import get_topics
from visualization import get_barplot, get_wordcloud
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, json,render_template,redirect,url_for,jsonify,json



app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/")        # Standard Flask endpoint
def homepage():
    return render_template("user_form.html")

@app.route("/get_plot", methods = ['GET','POST'])        # Standard Flask endpoint
def get_plot():
    data = request.form
    url = data['url']
    print(url)
    try:
        output = get_topics(url)
        lda_model = output[0]
        doc_list = output[1]
    except ValueError:
        return render_template('Prototype.html')
    if request.form['btn_identifier'] == 'get_wordcloud':
        print("wordcloud button is hit")
        try:
            get_wordcloud(STOP_WORDS,lda_model)
            return render_template('untitled1.html', name = 'Topic Wordcloud', url ='/static/images/wordcloud_topicwise.png')
        except ValueError:
            return render_template('Prototype.html')
        
    if request.form['btn_identifier'] == 'get_topics':
        print("Topic Button is hit")
        try:
            get_barplot(lda_model,doc_list)
            return render_template('untitled1.html', name = 'Topic_Barplot', url ='/static/images/topic_barplot.png')
        except ValueError:
            return render_template('Prototype.html')
        
        
    
    

@app.route("/goback", methods = ['POST'])
def go_back():
    return render_template("user_form.html")    
    
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0',debug=True,port=5003)
