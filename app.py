from flask import Flask,render_template,redirect,request, Response
import pandas as pd
import re
from ftfy import fix_text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from spacy.matcher import Matcher
import spacy
##############################################################################
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import BytesIO
#####################################################################################
import pymongo
import json

#######################################################################
#connecting to MongoDb
try:
    mongo=pymongo.MongoClient(host="localhost",
                              port=27017,
                              serverSelectionTimeoutMS=1000
                              )
    db=mongo.student #name of the database
    mongo.server_info() #triggers an exception if we cannot connect to db
except:
    print("cannot connect to db")
########################################################################

def extract_text_from_pdf(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = BytesIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    for page in PDFPage.get_pages(file, check_extractable=True):
        page_interpreter.process_page(page)

    extracted_text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()

    return extracted_text.decode('utf-8')


stopw  = set(stopwords.words('english'))

df =pd.read_csv('HRAnalytics-Project/job_final.csv') 
df['test']=df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stopw)]))

app=Flask(__name__)



@app.route('/')
def hello():
    return render_template("model.html")



@app.route("/home")
def home():
    return redirect('/')

@app.route('/submit',methods=['POST'])
def submit_data():
    if request.method == 'POST':
        
        f = request.files['userfile']
        f.save(f.filename)
        text = extract_text_from_pdf(f)
        print(text)
  
        extracted_text = {}

        #function to extract email adress
        def get_email_addresses(string):
            r = re.compile(r'[\w\.-]+@[\w\.-]+')
            return r.findall(string)
        
        email = get_email_addresses(text)
        extracted_text["E-Mail"] = email[0]
        # print(email)
        #function to get phone number
        def get_phone_numbers(string):
            r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
            phone_numbers = r.findall(string)
            return [re.sub(r'\D', '', num) for num in phone_numbers]

        phone_number= get_phone_numbers(text)
        extracted_text["Phone Number"] = phone_number
        # print(phone_number)
        # load pre-trained model

        nlp = spacy.load('en_core_web_sm')

        # initialize matcher with a vocab
        matcher = Matcher(nlp.vocab)

        def extract_name(resume_text):
            nlp_text = nlp(resume_text)
            
            # First name and Last name are always Proper Nouns
            pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
            
            matcher.add('NAME', [pattern], on_match = None)
            
            matches = matcher(nlp_text)

            for match_id, start, end in matches:
                span = nlp_text[start:end]
                return span.text

        name = extract_name(text)
        extracted_text["Name"]=name
        # print(name)

        def extract_skills(resume_text):
            nlp_text = nlp(resume_text)

            # removing stop words and implementing word tokenization
            tokens = [token.text for token in nlp_text if not token.is_stop]
            
            skills = ["machine learning",
                    "deep learning",
                    "nlp",
                    "natural language processing",
                    "mysql",
                    "sql",
                    "django",
                    "computer vision",
                    "tensorflow",
                    "opencv",
                    "mongodb",
                    "artificial intelligence",
                    "ai",
                    "flask",
                    "robotics",
                    "data structures",
                    "python",
                    "c++",
                    "matlab",
                    "css",
                    "html",
                    "github",
                    "php",
                    "Neural Networks",
                    "statistics",
                    "Transfer Learning"]
                    
            skillset = []
            
            # check for one-grams (example: python)
            for token in tokens:
                if token.lower() in skills:
                    skillset.append(token)
            
            # check for bi-grams and tri-grams (example: machine learning)
            for token in nlp_text.noun_chunks:
                token = token.text.lower().strip()
                if token in skills:
                    skillset.append(token)
            
            return [i.capitalize() for i in set([i.lower() for i in skillset])]


        skills = []
        skills = extract_skills(text)
        extracted_text['skills']=skills
        # load pre-trained model
        nlp = spacy.load('en_core_web_sm')

        # Grad all general stop words
        STOPWORDS = set(stopwords.words('english'))

        # Education Degrees
        EDUCATION = [
                    'BE','B.E.', 'B.E', 'BS', 'B.S', 
                    'ME', 'M.E', 'M.E.', 'MS', 'M.S', 
                    'B TECH', 'B.TECH', 'M.TECH', 'MTECH', 
                    'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
                ]

        def extract_education(resume_text):
            nlp_text = nlp(resume_text)

            # Sentence Tokenizer
            nlp_text = [sent.text.strip() for sent in nlp_text.sents]

            edu = {}
            # Extract education degree
            for index, text in enumerate(nlp_text):
                for tex in text.split():
                    # Replace all special symbols
                    tex = re.sub(r'[?|$|.|!|,]', r'', tex)
                    if tex.upper() in EDUCATION and tex not in STOPWORDS:
                        edu[tex] = text + nlp_text[index + 1]

            education = []
            for key in edu.keys():
                year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
                if year:
                    education.append((key, ''.join(year[0])))
                else:
                    education.append(key)
            return education

        education = extract_education(text)
        education
        extracted_text["Qualification"] = education
        
        #college/institution
        sub_patterns = ['[A-Z][a-z]* College of Engineering','[A-Z][a-z]* Educational Institute',
                        'University of [A-Z][a-z]*',
                        'Ecole [A-Z][a-z]*',
                        'Indian Institute Of Technology-*[A-Z][a-z]',
                        'National Institute Of Technology-*[A-Z][a-z]']
        
        pattern = '({})'.format('|'.join(sub_patterns))
        matches = re.findall(pattern, text)

        extracted_text["Institute"] = matches
        matches

        # for experience
        sub_patterns = ['[A-Z][a-z]* [A-Z][a-z]* Private Limited','[A-Z][a-z]* [A-Z][a-z]* Pvt. Ltd.','[A-Z][a-z]* [A-Z][a-z]* Inc.', '[A-Z][a-z]* LLC',
                ]
        pattern = '({})'.format('|'.join(sub_patterns))
        Exp = re.findall(pattern, text)
        extracted_text["Experience"] = Exp
        # print(extracted_text)

        ################################################
        # adding users with their data to mongodb
        try:
            user={"name":f"{extracted_text['Name']}",
                  "contact":f"{extracted_text['Phone Number']}",
                   "email":f"{extracted_text['E-Mail']}",
                    "skills":f"{extracted_text['skills']}",
                    "education":f"{extracted_text['Qualification']}",
                    "University":f"{extracted_text['Institute']}",
                    "experience":f"{extracted_text['Experience']}"
                  }
            dbResponse=db.users.insert_one(user)
            print(dbResponse.inserted_id)
             # return Response(response=json.dumps({"message":"user created","id":f"{dbResponse.inserted_id}"}),status=200,mimetype="application/json")
        except Exception as ex:
            print("******************")
            print(ex)

       
        org_name_clean = extracted_text['skills']
        
        def ngrams(string, n=3):
            string = fix_text(string) # fix text
            string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
            string = string.lower()
            chars_to_remove = [")","(",".","|","[","]","{","}","'"]
            rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
            string = re.sub(rx, '', string)
            string = string.replace('&', 'and')
            string = string.replace(',', ' ')
            string = string.replace('-', ' ')
            string = string.title() # normalise case - capital at start of each word
            string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
            string = ' '+ string +' ' # pad names for ngrams...
            string = re.sub(r'[,-./]|\sBD',r'', string)
            ngrams = zip(*[string[i:] for i in range(n)])
            return [''.join(ngram) for ngram in ngrams]
        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
        tfidf = vectorizer.fit_transform(org_name_clean)
        print('Vecorizing completed...')
        
        
        def getNearestN(query):
          queryTFIDF_ = vectorizer.transform(query)
          distances, indices = nbrs.kneighbors(queryTFIDF_)
          return distances, indices
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
        unique_org = (df['test'].values)
        distances, indices = getNearestN(unique_org)
        unique_org = list(unique_org)
        matches = []
        for i,j in enumerate(indices):
            dist=round(distances[i][0],2) 
  
            temp = [dist]
            matches.append(temp)
        matches = pd.DataFrame(matches, columns=['Match confidence'])
        df['match']=matches['Match confidence']
        df1=df.sort_values('match')
        df2=df1[['Position', 'Company','Location']].head(10).reset_index()
        
        
        
        
        
    #return  'nothing' 
    return render_template('model.html',tables=[df2.to_html(classes='job')],titles=['na','Job'])
        
        
        
        
        
if __name__ =="__main__":    
    app.run(port=80,debug=True)