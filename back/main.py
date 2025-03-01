import firebase_admin
from firebase_admin import credentials  # Importation du module pour gérer les identifiants Firebase
from firebase_admin import db           # Importation du module pour interagir avec la base de données Firebase
import re                             # Module pour utiliser les expressions régulières
import openai                         # Module pour interagir avec l'API OpenAI

# Initialisation de Firebase avec le fichier de clé JSON
cred = credentials.Certificate("YOUR JSON")  # Remplacez "YOUR JSON" par le chemin vers votre fichier de clé Firebase
firebase_admin.initialize_app(cred, {
    'URL': 'HTTPS'  # Remplacez 'HTTPS' par l'URL de votre base de données Firebase
})

# Référence à la branche 'prompt' dans la base de données Firebase
ref = db.reference('prompt')

# Variables globales
ignore_existing_data = True  # Permet d'ignorer les anciennes données au démarrage de l'écoute
first_idea = True            # Indique s'il s'agit de la première idée traitée
openai.api_key = 'KEY'       # Remplacez 'KEY' par votre clé d'API OpenAI
ASK_SUGGESTIONS = "Give possibilities to accomplish the category : "  # Message de base pour demander des suggestions
step = "Idea and research"   # Étape actuelle du processus

def generate_description(title, concept, features, step="Idea and research"):
    """
    Génère une description complète à partir du titre, du concept et des fonctionnalités fournies.
    """
    features_str = ""
    if features:
        # Concatène chaque fonctionnalité dans une chaîne de caractères
        for feature in features:
            features_str += str(feature)
        # Ajoute la liste des fonctionnalités à la description
        features_str += "Features that need to be in it: " + features_str + "\n"
    print(features_str)
    
    # Construction du prompt initial destiné à OpenAI
    initialization = (
        "I want you to behave as a start-up incubator advisor. Description idea: " + concept + "\n" +
        features_str +
        "Project title: " + title +
        " Build a hierarchical tree structure for developing the concept in the most efficient way, it will be divided by a regular business development step" +
        generate_first_prompt(step)
    )
    return initialization

def generate_first_prompt(step):
    """
    Génère un sous-prompt demandant d'afficher uniquement la catégorie actuelle,
    avec 5 points maximum (5 mots max chacun) précédés de leur numéro.
    """
    return "for now display only the category" + str(step) + \
           ". give me only 5 points (5 words max each), they" \
           "must be precede by their number"

def restructure_list(lst):
    """
    Réorganise une liste de chaînes en retirant les numéros et la ponctuation,
    pour ne conserver que le texte descriptif.
    """
    def extract_text_from_string(input_string):
        pattern = r'\d+\.\s(.*)'  # Expression régulière pour extraire le texte après le numéro suivi d'un point
        match = re.search(pattern, input_string)  # Recherche de la correspondance dans la chaîne
        if match:
            text = match.group(1)  # Extraction du texte correspondant
            return text

    new_lst = []
    for sentence in lst:
        new_lst.append(extract_text_from_string(sentence))
    return new_lst

def generate_summary(messages, step):
    """
    Génère un résumé pour l'étape donnée en ajoutant un prompt à la conversation,
    puis en récupérant la réponse d'OpenAI.
    """
    prompt = "Generate a summary for this step: " + step
    messages.append({'role': 'user', 'content': prompt})
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    summury = response['choices'][0]['message']['content']
    messages.append({'role': 'assistant', 'content': summury})
    return summury, messages

def generate_chat_response_first_time(description):
    """
    Envoie le prompt initial à OpenAI et retourne la réponse découpée par ligne
    ainsi que le fil de conversation complet.
    """
    messages = [{"role": "user", "content": message} for message in [description]]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    assistant_reply = response['choices'][0]['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_reply})
    # La réponse est découpée en lignes pour faciliter le traitement ultérieur
    return assistant_reply.split("\n"), messages

def get_sub_categories(category, messages):
    """
    Demande à OpenAI de générer des suggestions de sous-catégories pour la catégorie donnée.
    """
    messages.append({'role': 'user', 'content': ASK_SUGGESTIONS + category})
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    assistant_reply = response['choices'][0]['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_reply})
    # Sépare la réponse en une liste de suggestions
    suggestions = assistant_reply.split("\n")
    return suggestions, messages

def on_event_added(event):
    """
    Fonction appelée lors de l'ajout d'un nouvel événement dans Firebase.
    Traite l'événement en fonction des données reçues et met à jour la base de données avec la réponse générée.
    """
    global ignore_existing_data
    global first_idea
    global messages
    global step
    
    if ignore_existing_data:
        # Ignore les données existantes au démarrage
        ignore_existing_data = False
        return

    print("New event added:")
    print(event.path)
    print(event.data)
    
    # Si l'événement contient une nouvelle étape, met à jour la variable 'step'
    if "step" in event.data:
        step = event.data['step']
    # Si c'est la première idée ou si l'événement contient un titre, traite la nouvelle idée
    elif first_idea or "title" in event.data:
        description = generate_description(
            event.data['title'], 
            event.data['ideaDescription'],
            event.data['attributes'], 
            step
        )
        assistant_reply, messages = generate_chat_response_first_time(description)
        # Restructure la réponse pour ne garder que le texte descriptif
        assistant_reply = restructure_list(assistant_reply)
        print(assistant_reply)
        # Récupère la clé de l'événement pour la mise à jour dans Firebase
        prompt_key = event.path.split('/')[-1]
        event_data = event.data.copy()
        event_data['response'] = assistant_reply
        # Mise à jour de la branche 'promptOutput' dans Firebase
        ref_output = db.reference('promptOutput')
        event_ref = ref_output.child(prompt_key)
        event_ref.update(event_data)
        first_idea = False
    # Si l'événement contient une action de clic pour obtenir des sous-catégories
    elif "click" in event.data:
        category = event.data['click']
        sub_categories, messages = get_sub_categories(category, messages)
        sub_categories = restructure_list(sub_categories)
        print(sub_categories)
        prompt_key = event.path.split('/')[-1]
        event_data = event.data.copy()
        event_data['response'] = sub_categories
        ref_output = db.reference('promptOutput')
        event_ref = ref_output.child(prompt_key)
        event_ref.update(event_data)

# Démarre l'écoute des événements sur la référence 'prompt'
event_added_listener = ref.listen(on_event_added)
