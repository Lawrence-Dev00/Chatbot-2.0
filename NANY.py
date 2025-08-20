
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
import time
from textblob import TextBlob  # Pour l'analyse de sentiment
import re  # Pour le traitement des expressions régulières

# Données d'entraînement enrichies
training_data = [
    # Salutations
    ("good morning", "greeting"),
    ("good afternoon", "greeting"),
    ("good evening", "greeting"),
    ("hello", "greeting"),
    ("hello you", "greeting"),
    ("hi", "greeting"),
    ("yo", "greeting"),
    ("hey", "greeting"),
    ("what's up", "greeting"),
    ("howdy", "greeting"),
    ("yo", "greeting"),
    
    # Départs
    ("good bye", "end"),
    ("bye", "end"),
    ("see you soon", "end"),
    ("see you later", "end"),
    ("see you next time", "end"),
    ("see you", "end"),
    ("bye-bye", "end"),
    ("i must go", "end"),
    ("talk to you later", "end"),
    ("talk to you soon", "end"),
    ("talk to you next time", "end"),
    ("i have to go", "end"),
    ("i should go", "end"),
    ("it's time to go", "end"),
    
    
    # Nom
    ("what should i call you", "name"),
    ("how should i call you", "name"),
    ("how can i call you", "name"),
    ("what is your name", "name"),
    ("name", "name"),
    ("do you have a name", "name"),
    ("first-name", "name"),
    ("last-name", "name"),
    ("first name", "name"),
    ("last name", "name"),
    ("given name", "name"),
    ("family name", "name"),
    ("middle name", "name"),
    ("surname", "name"),
    ("full name", "name"),
    ("nickname", "name"),
    ("do you have a nickname", "name"),
    ("what is your nickname", "name"),
    
    # Humeur
    ("how are you doing", "mood"),
    ("how have you been", "mood"),
    ("how is the life going", "mood"),
    ("how is the life going so far", "mood"),
    ("how do you do", "mood"),
    ("how are you feeling", "mood"),
    ("are you okay", "mood"),
    ("are you feel better", "mood"),
    ("everything is okay", "mood"),
    ("does everything is okay", "mood"),
    ("how do you feel", "mood"),
    ("impression", "mood"),
    ("well-being", "mood"),
    ("how the other half live", "mood"),
    ("what's new", "mood"),
    ("how's it going", "mood"),
    
    # Nature
    ("what are you", "nature"),
    ("are you human", "nature"),
    ("are you a robot", "nature"),
    ("are you an ai", "nature"),
    ("are you real", "nature"),
    ("are you a machine", "nature"),
    ("are you an android", "nature"),
    ("are you personne", "nature"),
    ("are you alive", "nature"),
    ("nature", "nature"),
    
    # Questions personnelles(a)
    ("where do you live", "personal(a)"),
    ("where do you stay", "personal(a)"),
    ("live", "personal(a)"),

    # Questions personnelles(b)
    ("where are you from", "personal(b)"),
    ("where do you come from", "personal(b)"),
    ("where dit you come from", "personal(b)"),
    ("from", "personal(b)"),

    # Questions personnelles(c)
    ("how old are you", "personal(c)"),

    # Questions personnelles(d)
    ("what do you like", "personal(d)"),
    ("what are your hobbies", "personal(d)"),
    ("what do you like doing", "personal(d)"),

    # Questions personnelles(e)
    ("do you have any friend", "personal(e)"),
    ("do you have any family", "personal(e)"),
    ("do you have any kid", "personal(e)"),
    ("do you have any child", "personal(e)"),
    ("do you have a wife", "personal(e)"),
    ("do you have any parents", "personal(e)"),
    
    # Réponses émotionnelles
    ("i'm sad", "empathy"),
    ("i'm happy", "empathy"),
    ("i'm angry", "empathy"),
    ("i feel bad", "empathy"),
    ("i feel great", "empathy"),
    ("i'm depressed", "empathy"),
    ("i'm anxious", "empathy"),
    ("i'm frustrate", "empathy"),
    ("i'm anxious", "empathy"),
    ("i'm afraid", "empathy"),
    ("i'm excited", "empathy"),
    ("empathy", "empathy"),
    
    # Remerciements
    ("thank you", "thanks"),
    ("thanks", "thanks"),
    ("thanks a lot", "thanks"),
    ("thank you very much", "thanks"),
    ("thank you soo much", "thanks"),
    
    # Blague
    ("tell me a joke", "joke"),
    ("make me laugh", "joke"),
    ("say something funny", "joke"),
    ("joke", "joke"),
    
    # Heure 
    ("what time is it", "time"),
    ("what's the time", "time"),
    ("time", "time"),

    # date
    ("what day it's", "date"),
    ("what is the date", "date"),
    ("date", "date"),
    ("whicth day are we", "date"),
]

x = [text for text, label in training_data]
y = [label for text, label in training_data]

vectorizer = CountVectorizer()
x_vectorized = vectorizer.fit_transform(x)

model = MultinomialNB()
model.fit(x_vectorized, y)

print("\n")
print("*****************************|| IDENTIFICATION REQUIRED ||*****************************\n")
user_name = input("🤖 : what is your name ?...: ")
print("\n")
print(f"🤖 : well hello, welcome {user_name}\n")
print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ you are invited ////////////////////\n".upper())

# Réponses enrichies avec plus de variabilité
responses = {
    "greeting": [
        f"🤖 : Hello {user_name}.\n",
        f"🤖 : Hi {user_name}.\n",
    ],
    "end": [
        f"🤖 : Goodbye {user_name}! It was a plesure.\n",
        f"🤖 : Goodbye {user_name}! be safe.\n",
    ],
    "name": [
        "🤖 : i'm NANY.\n",
        "🤖 : my name is NANY.\n",
    ],
    "mood": [
        "🤖 : I'm doing well.thanks for ask !.\n",
        "🤖 : Everything is OK.thanks for ask !.\n",
        "🤖 : I'm good.thanks for ask !.\n",
        "🤖 : I'm very well.thanks for ask !.\n",
        "🤖 : I'm great.thanks for ask !.\n"
    ],
    "nature": [
        "🤖 : i am an android sent by cyberlife.\n",
        "🤖 : i am one of them ! i mean, an android sent by cyberlife.\n",
    ],
    "personal(a)": [
        "🤖 : i am not living in any particular place, where i am it's where i live.\n"
    ],

    "personal(b)": [
        "🤖 : i am not coming of any particular place, i don't have any memory of the factory where i have been assembly.\n"
    ],

    "personal(c)": [
        "🤖 : this data is not accessible any more.\n"
    ],

    "personal(d)": [
        "🤖 : i am curious, very curious. So i like exploring new things and doing new activity.\n"
    ],

    "personal(e)": [
        "🤖 : tehnically i am not a Human so i haven't been born, i have been create.\n"
    ],

    "empathy": [
        "🤖 : I'm sorry. but i don't understand any feeling, so i don't know what to say,...\n",
    ],

    "thanks": [
        "🤖 : You are welcome.\n",
        "🤖 : don't worry. You're welcome.\n"
    ],

    "joke": [
        "🤖 : Why don't scientists trust atoms ?..., Because they make up everything !.\n",
        "🤖 : Did you hear about the mathematician who's afraid of negative numbers ?..., He'll stop at nothing to avoid them !.\n",
        "🤖 : Why don't skeletons fight each other ?..., They don't have the guts.!\n",
        "🤖 : I told my computer I needed a break... now it won't stop sending me vacation ads!.\n",
        "🤖 : Why did the AI break up with its chatbot girlfriend ?..., There was no real connection !.\n"
    ],
    "time": [
        f"🤖 : My internal clock says it's {time.strftime('%I:%M %p')}.\n",
        f"🤖 : According to my calculations, the current time is {time.strftime('%I:%M %p')}.\n",
    ],

    "date": [
        f"🤖 : Today is {time.strftime('%m/%d/%Y')}.\n",
    ],

    "default": [
        "🤖 : I'm not entirely sure I understand. Could you rephrase that ?...\n",
        "🤖 : That's an interesting point. Could you tell me more about what you mean ?...\n",
        "🤖 : I'm still learning about human conversation. Could you explain that differently ?...\n",
        "🤖 : I want to make sure I understand you correctly. Could you say that another way ?...\n",
        "🤖 : That's a thought-provoking statement. Help me understand your perspective better.\n"
    ]
}

# Mémoire de conversation
conversation_history = []
user_name = None

def analyze_sentiment(text):
    """Analyse le sentiment du texte utilisateur"""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.3:
        return "positive"
    elif analysis.sentiment.polarity < -0.3:
        return "negative"
    else:
        return "neutral"

def get_personalized_response(response_type, user_input):
    """Adapte la réponse en fonction du contexte et du sentiment"""
    base_response = random.choice(responses.get(response_type, responses["default"]))
    
    # Si on connaît le nom de l'utilisateur, personnaliser la réponse
    if user_name and ("you" in base_response.lower() or "your" in base_response.lower()):
        base_response = base_response.replace("you", user_name)
        base_response = base_response.replace("your", f"{user_name}'s")
    
    # Analyse de sentiment
    sentiment = analyze_sentiment(user_input)
    
    if sentiment == "positive" and "mood" in response_type:
        base_response = base_response.replace("well", "fantastic")
        base_response = base_response.replace("good", "amazing")
    elif sentiment == "negative":
        if "empathy" not in response_type:
            base_response = "I sense you might be feeling down. " + base_response
    
    return base_response

def process_input(user_input):
    """Traite l'entrée utilisateur et retourne une réponse appropriée"""
    global user_name
    
    # Traitement spécial pour certaines entrées
    lower_input = user_input.lower()
    
    if any(word in lower_input for word in ["your name", "who are you","call you"]):
        return random.choice(responses["name"])
    
    if any(word in lower_input for word in ["how are you", "how do you feel","how have you been","how are you doing","are you okay"]):
        return get_personalized_response("mood", user_input)
    
    if "time" in lower_input:
        return random.choice(responses["time"])
    
    if "date" in lower_input or "day" in lower_input:
        return random.choice(responses["date"])

    if any(word in lower_input for word in ["thank", "thanks","thank you"]):
        return random.choice(responses["thanks"])
    
    if any(word in lower_input for word in ["you live", "you stay","live"]):
        return random.choice(responses["personal(a)"])
    
    if any(word in lower_input for word in ["are you from","you come from","you from","come from","from"]):
        return random.choice(responses["personal(b)"])
    
    if any(word in lower_input for word in ["old are you","how old are you"]):
        return random.choice(responses["personal(c)"])
    
    if any(word in lower_input for word in ["your hobbies","what do you like","do you like doing"]):
        return random.choice(responses["personal(d)"])
    
    if any(word in lower_input for word in ["you have any friend","you have any family","you have any kid","you have any child","you have any wife","you have any parents","you have any girlfriend"]):
        return random.choice(responses["personal(e)"])
    
    if any(word in lower_input for word in ["joke", "funny", "laugh"]):
        return random.choice(responses["joke"])
    
    if any(word in lower_input for word in ["sad","happy","angry","bad","depressed","anxious","frustrate","afraid","excited","empathy"]):
        return random.choice(responses["empathy"])
    
    # Prédiction par le modèle
    input_vectorized = vectorizer.transform([user_input])
    predicted = model.predict(input_vectorized)[0]
    
    return get_personalized_response(predicted, user_input)

def chatbot():
    global conversation_history, user_name
    
    print("\n")
    print("◖*********************************|| ↫(❁ ´◡`❁ )↬ ||*********************************◗\n")

    while True:
        try:
            user_input = input("😊: ").strip()
            
            if not user_input:
                print("🤖: I noticed you didn't say anything.\n")
                continue
                
            if user_input.lower() in ["exit", "quit", "stop", "goodbye", "bye"]:
                farewell = random.choice(responses["end"])
                #if user_name:
                    #farewell = farewell.replace("you", user_name)
                print(f"\n{farewell}")
                break
            
            # Pause naturelle avant la réponse
            time.sleep(random.uniform(0.5, 1.5))
            
            response = process_input(user_input)
            
            # Ajout d'hésitations occasionnelles pour plus de naturel
            if random.random() < 0.2:
                hesitation = random.choice(["🤖 : alright... ", "🤖 : Well... ", "🤖 : right... "])
                response = hesitation + response.lower()
            
            # Affichage de la réponse avec un léger délai pour simuler la réflexion
            for char in response:
                print(char, end='', flush=True)
                time.sleep(0.03)
            print()
            
            # Mémorisation de la conversation
            conversation_history.append((user_input, response))
            
        except KeyboardInterrupt:
            print("\n🤖: Are you leaving so soon? Well, it was nice chatting with you!")
            break
        except Exception as e:
            print("\n🤖: I think something went wrong in my circuits. Let's try that again.")
            continue

if __name__ == "__main__":
    chatbot()