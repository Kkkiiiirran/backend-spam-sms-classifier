# import pickle
# import nltk
# from nltk.corpus import stopwords
# import string
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# nltk.download('stopwords')


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
    
#     data = []
#     for word in text:
#         if word.isalnum():
#             data.append(word)
#     text=data[:]
#     data.clear()
    
#     for word in text:
#         if word not in stopwords.words("english") and word not in string.punctuation:
#             data.append(word)
#     text = data[:]
#     data.clear()
    
#     for word in text:
#         data.append(ps.stem(word))
    
#     return " ".join(data)

# tfidf = pickle.load(open("vectorizer.pkl","rb"))
# model = pickle.load(open("model.pkl", "rb"))

# inputsms = "Hurrayyy You have won $100000"
# # textbox
# # precprocess
# sentences = [
#     "Congratulations! You've won a $1000 gift card. Click here to claim: www.fakeprize.com",
#     "Hey, are we still on for dinner tonight?",
#     "You've been selected for a limited-time offer. Get a loan at 0% interest now!",
#     "Don't forget to submit the report by 5 PM.",
#     "Urgent! Your account has been compromised. Verify your info at: www.fakebank.com",
#     "Can you pick up some groceries on your way home?",
#     "Earn $5000 per week from home! Sign up now: www.easycash.com",
#     "Happy Birthday! Hope you have a great day.",
#     "Get a free iPhone now! Limited stocks available. Visit: www.freeiphone.com",
#     "Let's meet at the park for a walk this evening."

# ]



# for inputsms in sentences:
#     transform_sms = transform_text(inputsms)
#     # vectorize
#     vector_input = tfidf.transform([transform_sms])
#     # predict

#     result = model.predict(vector_input)[0]
#     # display

#     # print(result)
#     if result==1:
#         print("Spam")
#     else:
#         print("Not Spam")

# # so I have a python file which doessomeproeprocessingand build a prediciton model.

# # and I have react file with textarea component 

# # I want when the user enters text there and hits the "Check"button the data is recived by my python file using FastAPI.

# # I will share the code with you, the file has an inputsms varibale where I have written a dummy string I 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import uvicorn

ps = PorterStemmer()
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    data = []
    for word in text:
        if word.isalnum():
            data.append(word)
    text = data[:]
    data.clear()

    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            data.append(word)
    text = data[:]
    data.clear()

    for word in text:
        data.append(ps.stem(word))

    return " ".join(data)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

class SMSInput(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.post("/predict/")
async def predict_spam(input: SMSInput):
    transform_sms = transform_text(input.text)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]
    return {"prediction": "Spam" if result == 1 else "Not Spam"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
