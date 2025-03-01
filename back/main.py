import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import re


import openai
cred = credentials.Certificate("YOUR JSON")
firebase_admin.initialize_app(cred, {
    'URL': 'HTTPS'
})

ref = db.reference('prompt')
ignore_existing_data = True
first_idea = True
openai.api_key = 'KEY'
ASK_SUGGESTIONS = "Give possibilities to accomplish the category : "
step = "Idea and research"

def generate_description(title, concept, features, step="Idea and research"):
    features_str = ""
    if features:
        for feature in features:
            features_str += str(feature)
        features_str += "Features that need to be in it: " + features_str + "\n"
    print(features_str)
    initialization = "I want you to behave as a start-up incubator advisor. Description idea: " + concept + "\n" + features_str + "Project title: " + title + \
                     " Build a hierarchical tree structure for developing the concept in the most efficient way, it will be divided by a regular business development step" \
                     + generate_first_prompt(step)

    return initialization


def generate_first_prompt(step):
    return "for now display only the category" + str(step) + \
           ". give me only 5 points (5 words max each), they" \
           "must be precede by their number"


def restructure_list(lst):
    def extract_text_from_string(input_string):
        pattern = r'\d+\.\s(.*)'  # Regular expression pattern
        match = re.search(pattern, input_string)  # Find the match in the input string

        if match:
            text = match.group(1)  # Extract the text portion
            return text
        # else:
        #     return

    new_lst = []
    for sentance in lst:
        new_lst.append(extract_text_from_string(sentance))
    return new_lst


def generate_summary(messages, step):
    prompt = "Generate a summary for this step: " + step
    messages.append({'role': 'user', 'content': prompt})
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages)
    summury = response['choices'][0]['message']['content']
    messages.append({'role': 'assistant', 'content': summury})

    return summury, messages


def generate_chat_response_first_time(description):
    messages = [{"role": "user", "content": message} for message in [description]]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    assistant_reply = response['choices'][0]['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_reply})
    return assistant_reply.split("\n"), messages


def get_sub_categories(category, messages):
    """

    :param category:
    :param messages:
    :return:
    """
    messages.append({'role': 'user', 'content': ASK_SUGGESTIONS + category})
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages)
    assistant_reply = response['choices'][0]['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_reply})
    suggestions = assistant_reply.split("\n")

    return suggestions, messages


def on_event_added(event):
    global ignore_existing_data
    global first_idea
    global messages
    global step
    if ignore_existing_data:
        ignore_existing_data = False
        return
    print("New event added:")
    print(event.path)
    print(event.data)
    if "step" in event.data:
        step = event.data['step']
    elif first_idea or "title" in event.data:
        description = generate_description(event.data['title'], event.data['ideaDescription'],
                                           event.data['attributes'], step)
        assistant_reply, messages = generate_chat_response_first_time(description)
        assistant_reply = restructure_list(assistant_reply)
        print(assistant_reply)
        prompt_key = event.path.split('/')[-1]
        event_data = event.data.copy()
        event_data['response'] = assistant_reply
        ref_output = db.reference('promptOutput')
        event_ref = ref_output.child(prompt_key)
        event_ref.update(event_data)
        first_idea = False
    # elif "clickPdf" in event.data:
    #     generate_pdf.generate_pdf("Unicorn")
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


event_added_listener = ref.listen(on_event_added)

