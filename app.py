import random
import string
import streamlit as st
import wikipediaapi
from funcs import *
from bot import *
from plots import *

# doing this first to ensure things will work
def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# connects to Wikipedia API and defines game log
random_string = generate_random_string(10)
wiki_wiki = wikipediaapi.Wikipedia(
    f'WikiBot-{random_string} (https://www.linkedin.com/in/kmaurinjones/)',
    'en',
    timeout = 30
    )

# Setting the page title
st.set_page_config(
    page_title = "WikiGameBot",
    layout = "centered"
)

# Title of the web app
st.title("WikiGameBot")

submitted = None

# Form for user inputs
with st.form("wiki_game_form"):
    st.write("Enter a topic to start on and/or a topic to end on. If left blank, a topic will be chosen randomly.")
    
    # User input for Start Topic
    start_topic = st.text_input(
        label="Start Topic",
        placeholder="Leave this blank to randomly select a start topic"
    )
    
    # User input for Target Topic
    target_topic = st.text_input(
        label="Target Topic",
        placeholder="Leave this blank to randomly select a target topic"
    )
    
    # Submit button for the form
    submitted = st.form_submit_button("Begin")
    
if submitted:

    # if start not passed, get a random one
    if not start_topic:
        start_topic = get_random_wiki_page(wiki_wiki)

    # if target topic not passed, get a random one
    if not target_topic:
        target_topic = get_random_wiki_page(wiki_wiki)

    game = WikiGameBot(wiki_wiki = wiki_wiki, start_topic = start_topic, target_topic = target_topic)

    # Displaying the start and target topics and their repsective summaries
    with st.expander(f"**Wiki Results for '{game.start_topic)}'**"):
        st.markdown(f'- *"{game.starting_url}"*')
        st.markdown(f'- *"{game.current_summary}"*')

    with st.expander(f"**Wiki Results for '{game.target_topic}'**"):
        st.markdown(f'- *"{game.target_url}"*')
        st.markdown(f'- *"{game.target_summary}"*')

    st.divider()

    game.play_game()

outro_message = """
Thanks for checking out this app. If you have any questions or comments or would like to connect for any reason, you can reach me at:
- LinkedIn: https://www.linkedin.com/in/kmaurinjones/
""".strip()
st.write(outro_message)

disclaimer = """
Disclaimer: The creator of this web app and author of this code is not responsible for the content produced by the app. The content produced by this app is a result of content sourced from Wikipedia and is property of Wikipedia.
""".strip()
st.write(disclaimer)
