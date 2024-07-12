from flask import Flask, render_template, request, jsonify, redirect, url_for, jsonify, flash, session
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import requests
import torch
from bs4 import BeautifulSoup as bs
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Initializing the database in sqlalchemy by specifying urio location
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Save memory and improve performance
app.secret_key = 'KEY'    #setting up of key
app.config['SESSION_TYPE'] = 'filesystem'  #setting up session

db = SQLAlchemy(app)                #creating db object and initializing SQlAlchemy which is object to database
migrate = Migrate(app, db)          # Using the Migrate function

class User(db.Model):           #class User inheriting from db object
    id = db.Column(db.Integer, primary_key=True)                       #ORM functions
    username = db.Column(db.String(20), unique=True, nullable=False)  
    password = db.Column(db.String(60), nullable=False)
    name = db.Column(db.String(50))
    data_list_1 = db.Column(db.Text)


    def __repr__(self):
        return f"User('{self.username}', '{self.name}', '{self.data_list_1}')"  #Human readable interpretation

 # Example of storing and retrieving a list
list_to_store = ["contructive algorithm", "strings", "dictionary"]

# Serialize the list to JSON before storing it
data_to_store = json.dumps(list_to_store)


admin = Admin(app, name='MyApp', template_mode='bootstrap3')
admin.add_view(ModelView(User, db.session))
with app.app_context():
    db.create_all()

    def create_admin_user():
      admin_user = User.query.filter_by(username='admin').first()
      if not admin_user:
          admin_user = User(username='admin', password='admin', name='Admin', data_list_1 =data_to_store)
          db.session.add(admin_user)
          db.session.commit()
          print("Admin user created.")

    create_admin_user() 
  

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['usrnm']
        password = request.form['pswd']
        print(f"Attempting login with Username: {username}, Password: {password}")
        user = User.query.filter_by(username=username).first()
        if user:
            print(f"User found: {user}")
            if user.password == password:
                session['username'] = username
                flash(f"Login successful, redirecting to dashboard for user: {username}")
                return render_template('chat.html')
            else:
                flash("Incorrect password.")
        flash('Login Unsuccessful. Please check username and password.', 'danger')
    return render_template('home.html')

@app.route('/register', methods=["GET", "POST"])
def register():
  if request.method == 'POST':
    nm = request.form['nm']
    usrnm = request.form['idv']
    pswd = request.form['ps']
    new_user = User(username=usrnm, password=pswd, name=nm)
    db.session.add(new_user)
    db.session.commit()
    flash('Your account has been created! You are now able to log in', 'success')
    return redirect(url_for('login'))
  return render_template('register.html')

@app.route('/chat', methods=["POST", "GET"])
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

#Web scraping
def scrape(url):
  vocab = {"<p>":" ", "</p>":" ", "$$$":'', "\\leq":" <= ", "\\le":" <= ", r'\xa0':" ", "\\cdot":" x ", "\\ldots":"...","\\dots":"...",
         '</span>':' ' , '<span class="tex-font-style-it">':' ', '<span class="tex-font-style-bf">':' ','<span class="tex-font-style-tt">':' ' ,
         '\\ne': '≠', r'\xa0':'', '\\oplus':' xor ', '\\,':'', '&lt':'<', '&gt':'>', "^\\dagger":'', '\\ge':' >= ', '\\operatorname':'',
         "\'":''}

  r = requests.get(url)

  # Parsing the HTML
  soup = bs(r.content, 'html.parser')

  s = soup.find('div', class_='entry-content')
  ques = soup.find('div',{"class":"problem-statement"}).get_text()
  tags = soup.find_all('span', {"class": "tag-box"})
  # print(tags, type(tags))
  tag_lst=[]
  for tag in tags:
    lst = tag.get_text().split('\n')
    tag_lst.append(lst[1].strip())

  Rating = list(map(lambda x: x[1:].isnumeric(), tag_lst))
  Rating = int(list(tag_lst[i] for i in range(len(tag_lst)) if Rating[i])[0][1:])
  # print(Rating)
  # Rating  = int(input("Enter your current rating:"))
  sim = []
  for Tag in tag_lst:
    url = f"https://codeforces.com/problemset?tags={Tag},{Rating+100}-{Rating+200}"
    r = requests.get(url)
    soup = bs(r.content, 'html.parser')
    elems = soup.find_all('div',style="float: left;")
    i = 0
    for elem in elems:
      anchor_tag = elem.find('a')
      href_value = anchor_tag.get('href')
      sim.append(f"https://codeforces.com{href_value}")
      i += 1
      if i == 5:
        break
  
  ques = ques.split("Note")[0]
  ques = "".join(map(str, ques))
  ques = ques.split("standard output")[1]
  ques = "".join(map(str, ques))
  ques = ques.split("ExamplesInput")[0]
  ques = "".join(map(str, ques))
  ques = ques.split("ExampleInput")[0]

  for word in vocab:
    lst = []
    for i in ques.split(word):
      if i!='':
        lst.append(i)
    # lst
    ques = vocab[word].join(map(str, lst))
  return ques, tag_lst, sim

def scrape_leetcode(problem_name):
  data = {"operationName":"questionData","variables":{"titleSlug":f"{problem_name}"},"query":"query questionData($titleSlug: String!) {\n  question(titleSlug: $titleSlug) {\n    questionId\n    questionFrontendId\n    boundTopicId\n    title\n    titleSlug\n    content\n    translatedTitle\n    translatedContent\n    isPaidOnly\n    difficulty\n    likes\n    dislikes\n    isLiked\n    similarQuestions\n    contributors {\n      username\n      profileUrl\n      avatarUrl\n      __typename\n    }\n    langToValidPlayground\n    topicTags {\n      name\n      slug\n      translatedName\n      __typename\n    }\n    companyTagStats\n    codeSnippets {\n      lang\n      langSlug\n      code\n      __typename\n    }\n    stats\n    hints\n    solution {\n      id\n      canSeeDetail\n      __typename\n    }\n    status\n    sampleTestCase\n    metaData\n    judgerAvailable\n    judgeType\n    mysqlSchemas\n    enableRunCode\n    enableTestMode\n    envInfo\n    libraryUrl\n    __typename\n  }\n}\n"}

  r = requests.post('https://leetcode.com/graphql', json = data).json()
  soup = bs(r['data']['question']['content'], 'lxml')
  topic = r['data']['question']['topicTags']
  topic_lst=[]
  for d in topic:
    topic_lst.append(d['name'])

  simques = json.loads(r['data']['question']['similarQuestions'])
  simques_lst = []
  for d in range(len(simques)//2):
    url = "https://leetcode.com/problems/"+ f'{simques[d]["titleSlug"]}' +"/description/"
    simques_lst.append(url)

  question =  soup.get_text().replace('\n',' ')

  question_p1 = question.split('Example')
  question_p2 = question.split('Constraint')
  ques = [question_p1[0], 'Constraint' + question_p2[1]]
  ques = ' '.join(map(str, ques))
  ques = ques.split('\xa0')
  ques = ''.join(map(str, ques))
  return ques,topic_lst,simques_lst

#RAG

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

documents = SimpleDirectoryReader("/home/user/ChatBot/article").load_data()

index = VectorStoreIndex.from_documents(documents)

for doc in documents:
    if "AUTHOR" in doc.text:
        documents.remove(doc)

    if "Author" in doc.text:
        documents.remove(doc)

    if "Preface" in doc.text:
        documents.remove(doc)

# set number of docs to retreive
top_k = 1

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

#Loading the Model
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", quantization_config=config, trust_remote_code=True,torch_dtype=torch.bfloat16)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

def get_Chat_response(text):
  text = text.split(' ')
  if text[0].lower() in ['c', 'cf', 'codeforces']:
      url = r"https://codeforces.com/problemset/problem/" + f"{text[1]}" + r"/" + f"{text[2]}"
      ques = scrape(url)[0]
      tags = scrape(url)[1]
      sim = scrape(url)[2]
  elif text[0].lower() in ['l', 'leetcode']:
      problem_name = text[1]
      ques = scrape_leetcode(problem_name)[0]
      tags = scrape_leetcode(problem_name)[1]
      sim = scrape_leetcode(problem_name)[2]

  query = ques
  response = query_engine.query(query)
  context = "Context:\n"
  for i in range(top_k):
      context = context + response.source_nodes[i].text + "\n\n"

  reflection_prompt=""" Your task is to think upon and reflect on this problem so,
  Describe the problem, in bullet
  points, while addressing the problem goal, inputs, outputs,
  rules, constraints, and other relevant details that appear in
  the problem description."""
  test_reasoning= """Explain the reasoning why each test input leads
  to the output."""
  possible_solution="""explain some ways to solve the question in bullet points"""
  ai_tests= """Generate an additional 6-
  8 diverse input-output tests for the problem. Try to cover
  cases and aspects not covered by the original public tests."""
  initial_prompt= """I am going to give you a question from codeforces and than ask you to tell me something about the question through a series of questions
  reply to each question in order
  QUESTION-"""+query+"""
  REFLECTION-"""+reflection_prompt+"""
  TEST CASES-"""+test_reasoning+"""
  POSSIBLE SOLUTION-"""+possible_solution+"""
  ADDITIONAL TESTS-"""+ai_tests

  prompt_template_w_context = lambda context, comment: f"""You always output C++ code for the problem asked to you
  {context[:200]}
  Please give code for the following question in C++ also give the explanation for your answer. Use the context above if it is helpful.
  {initial_prompt}
  """

  prompt = prompt_template_w_context(context, initial_prompt)

  # messages=[
  #     { 'role': 'user', 'content': prompt}
  # ]

  # # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
  # # initial_output= model.generate(inputs, max_new_tokens=1024, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

  # # initial_output = tokenizer.decode(initial_output[0][len(inputs[0]):], skip_special_tokens=True)
  final_prompt=prompt+"using all this knowledge generate a solution code to the question in c++" + "Only output code"
  messages=[
      { 'role': 'user', 'content': final_prompt}
  ]

  inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
  # tokenizer.eos_token_id is the id of <|EOT|> token
  outputs = model.generate(inputs, max_new_tokens=4096, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
  # return 
  # print(tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n'))
  data_lst = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('\n')
  data_lst.append("")
  data_lst.append('Recommended Questions to Solve Based on your Rating')
  # for i in sim: data_lst.append(i)
  # print(data_lst)
  data = create_tag(data_lst) # this is string
  sim_ques_lst = simQues(sim)
  for i in sim_ques_lst: data+=i
  print(data)
  
  user = User.query.filter_by(username=session['username']).first()
  if user:
    if user.data_list_1 is None:
      setattr(user, "data_list_1", json.dumps(tags))
      db.session.commit()
      print(user.data_list_1)
    else:
      lst = json.loads(user.data_list_1)
      lst = list(map(lambda x:x.lower(), lst))
      for t in range (len(tags)):
          if tags[t].lower() not in lst:
            lst.append(tags[t])
      setattr(user, "data_list_1", json.dumps(lst))
      db.session.commit()
      print(user.data_list_1)

  return data


def create_tag(output_lst):
    str_tag = '<div class="d-flex justify-content-start mb-4"><div class="img_cont_msg"><img src="https://i.ibb.co/fSNP7Rz/icons8-chatgpt-512.png" class="rounded-circle user_img_msg"></div><div class="msg_cotainer">'
    for i in range(len(output_lst)):
        # $nbsp is one space
        i_new = ""
        for j in output_lst[i]:
            if j==" ":
               i_new += "&nbsp;"
            elif j=="<":
               i_new += "&lt;"
            elif j==">":
               i_new += "&gt;"
            elif j=="&":
               i_new += "&amp;"
            else:
               i_new += j
        str_tag += i_new + "<br>"
    return str_tag

def simQues(sim):
  sim_ques_links = []
  for i in sim:
    link = f'<p><a href="{i}" target = "_blank">Question {sim.index(i)+1}</a></p><br>'
    sim_ques_links.append(str(link))
  return sim_ques_links


if __name__ == '__main__':
    app.run()