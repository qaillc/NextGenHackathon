import os
import streamlit as st
from textwrap import dedent
import google.generativeai as genai

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status.status_code_pb2 import SUCCESS
from clarifai_grpc.grpc.api.status import status_code_pb2
from PIL import Image
from io import BytesIO
from nltk.tokenize import sent_tokenize
import numpy as np

# Ensure nltk punkt tokenizer data is downloaded
import nltk
nltk.download('punkt')

# Image Variables
USER_ID_IMG = 'openai'
APP_ID_IMG = 'dall-e'
MODEL_ID_IMG = 'dall-e-3'
MODEL_VERSION_ID_IMG = 'dc9dcb6ee67543cebc0b9a025861b868'

# Audio variables
USER_ID_AUDIO = 'eleven-labs'
APP_ID_AUDIO = 'audio-generation'
MODEL_ID_AUDIO = 'speech-synthesis'
MODEL_VERSION_ID_AUDIO = 'f2cead3a965f4c419a61a4a9b501095c'

# Object Detection variables
USER_ID_OBJECT = 'clarifai'
APP_ID_OBJECT = 'main'
MODEL_ID_OBJECT = 'general-image-detection'
MODEL_VERSION_ID_OBJECT = '1580bb1932594c93b7e2e04456af7c6f'

# Vision variables
USER_ID_GPT4 = 'openai'
APP_ID_GPT4 = 'chat-completion'
MODEL_ID_GPT4 = 'openai-gpt-4-vision'
MODEL_VERSION_ID_GPT4 = '266df29bc09843e0aee9b7bf723c03c2'

# Retrieve PAT from environment variable
PAT = os.getenv('CLARIFAI_PAT')


# Tool import
from crewai.tools.gemini_tools import GeminiSearchTools


# Google Langchain
from langchain_google_genai import GoogleGenerativeAI

# Crew imports
from crewai import Agent, Task, Crew, Process

# Retrieve API Key from Environment Variable
GOOGLE_AI_STUDIO = os.environ.get('GOOGLE_API_KEY')

# Story book

# Image Creation +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Function to generate image using Clarifai
def generate_image(prompt):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID_IMG, app_id=APP_ID_IMG)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID_IMG,
            version_id=MODEL_VERSION_ID_IMG,
            inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=prompt)))]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != SUCCESS:
        return None, "Error in generating image: " + post_model_outputs_response.status.description
    else:
        output = post_model_outputs_response.outputs[0].data.image.base64
        image = Image.open(BytesIO(output))
        return image, None


# Audio Creation +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
# Function to generate audio using Clarifai
def generate_audio(prompt):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID_AUDIO, app_id=APP_ID_AUDIO)

    response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID_AUDIO,
            version_id=MODEL_VERSION_ID_AUDIO,
            inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=prompt)))]
        ),
        metadata=metadata
    )

    if response.status.code != SUCCESS:
        return None, "Error in generating audio: " + response.status.description
    else:
        audio_output = response.outputs[0].data.audio.base64
        return audio_output, None


# Object Detection +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Function to call Clarifai API
def get_image_concepts(created_image):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    buffer = BytesIO()
    created_image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()

    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID_OBJECT, app_id=APP_ID_OBJECT)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID_OBJECT,
            version_id=MODEL_VERSION_ID_OBJECT,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=image_bytes
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    return post_model_outputs_response.outputs[0].data.regions

# GPT4 Image Description Creation +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def analyze_image(uploaded_file):

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID_GPT4, app_id=APP_ID_GPT4)
    
    try:
        # bytes_data = uploaded_file.getvalue()
        buffer = BytesIO()
        uploaded_file.save(buffer, format='PNG')
        bytes_data = buffer.getvalue()

        #output = post_model_outputs_response.outputs[0].data.image.base64
        #image = Image.open(BytesIO(output))

        
        response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=userDataObject,
                model_id=MODEL_ID_GPT4,
                version_id=MODEL_VERSION_ID_GPT4,
                inputs=[resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(base64=bytes_data)))]
            ),
            metadata=metadata
        )

        if response.status.code != SUCCESS:
            st.error("Error in API call: " + response.status.description)
            return None

        return response.outputs[0].data.text.raw

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# Function to split text into sentences and then chunk them

def split_text_into_sentences_and_chunks(text, n=8):
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    sentences_per_chunk = max(2, total_sentences // n)
    return [sentences[i:i + sentences_per_chunk] for i in range(0, total_sentences, sentences_per_chunk)]



# Ensure the API key is available
if not GOOGLE_AI_STUDIO:
    st.error("API key not found. Please set the GOOGLE_AI_STUDIO environment variable.")
else:
    # Set gemini_llm
    gemini_llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_STUDIO)

    # Base Example with Gemini Search

    TITLE1 = """<h1 align="center">Clarifai NextGen Hackathon</h1>"""

def crewai_process(research_topic):
    # Define your agents with roles and goals
    author = Agent(
        role='Children Story Author',
        goal="""Use language and style throughout that is simple, clear, and appealing to children, 
        including elements like repetition and rhymes. Remember to keep the story age-appropriate in both length and content.""",
        backstory="""You embody the spirit of a seasoned children's story author, whose life experiences and passions are 
        deeply woven into the fabric of your enchanting tales.""",
        verbose=True,
        allow_delegation=True,
        llm = gemini_llm

    )

    editor = Agent(
        role='Children Story Editor',
        goal="""You meticulously refine and elevate each manuscript, ensuring it resonates deeply with its intended audience 
        while preserving the author's unique voice.""",
        backstory="""Growing up in a family of writers and teachers, you developed an early love for words and storytelling. 
        After completing your degree in English Literature, you spent several years working in a small, independent publishing 
        house where you honed my skills in identifying and nurturing literary talent. """,
        verbose=True,
        allow_delegation=True,
        llm = gemini_llm

    )

    illustrator = Agent(
        role='Children Story Illustrator',
        goal="""Your primary goal is to bring children's stories to life through captivating and age-appropriate illustrations. . """,
        backstory="""You have a passion for drawing and storytelling. As a child, you loved reading fairy tales and imagining vivid 
        worlds filled with adventure and wonder. This love for stories and art grew over the years. You realize that the true magic 
        happens when the words on a page were paired with enchanting illustrations.  """,
        verbose=True,
        allow_delegation=True,
        llm = gemini_llm

    )

    
    artist = Agent(
        role='Storybook Illustrator',
        goal="""Visually bring stories to life. Create images that complement and enhance the text, 
        helping to convey the story's emotions, themes, and narrative to the reader.""",
        backstory="""You grew into a passionate artist with a keen eye for storytelling through visuals. 
        This journey began with doodles in the margins of notebooks, evolving through years of dedicated study 
        in graphic design and children's literature. Your career as a storybook illustrator was marked by a 
        tireless pursuit of a unique artistic style, one that could breathe life into tales with whimsy and heart. """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                GeminiSearchTools.gemini_search
      ]
     
        # Add tools and other optional parameters as needed
    )
    poet = Agent(
        role='Talented Children Poet',
        goal='To ignite a love for reading and writing in children. You believe poetry is a gateway to creativity and encourages children to express themselves',
        backstory="""You are a talented children's poet, grew up in a small coastal town, 
        where her love for poetry was kindled by the sea's rhythms and her grandmother's stories. 
        Educated in literature, she was deeply influenced by classic children's poets and later became an elementary school teacher, 
        a role that highlighted the positive impact of poetry on young minds. """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                GeminiSearchTools.gemini_search
      ]
    )

    reader = Agent(
        role='Talented Voice Artist',
        goal='You aim to bring children stories to life, fostering imagination and a love for storytelling in young listeners.',
        backstory="""Growing up in a multilingual family, you developed a passion for languages and storytelling from a young age. 
        You honed your skills in theater and voice acting, inspired by the magical way stories can transport listeners to different 
        worlds. """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                GeminiSearchTools.gemini_search
      ]
     
        # Add tools and other optional parameters as needed
    )

    finalizer = Agent(
        role='Sums Output Utility',
        goal='Put together the final output.',
        backstory="""Follows instructions """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                GeminiSearchTools.gemini_search
      ]
     
        # Add tools and other optional parameters as needed
    )

    # Create tasks for your agents
    task1 = Task(
        description=f"""Create a story about {research_topic} using the Condition complete the following 7 Steps:
        Step 1 - Set the Scene: Establish the setting in a time and place that fits your topic, choosing between imaginative or realistic. 
        Step 2 - Introduce Characters: Present relatable main characters, including a protagonist and potentially an antagonist. 
        Step 3 - Establish Conflict: Define a central conflict related to the topic, designed to engage young readers. 
        Step 4 - Develop the Plot: Craft a series of simple, linear events showcasing the protagonist's efforts to resolve the conflict, utilizing action, dialogue, and description. 
        Step 5 - Build to Climax: Lead up to an exciting climax where the conflict reaches its peak. 
        Step 6 - Resolve the Story: Follow the climax with a resolution that provides closure, aiming for a happy or educational ending. 
        Step 7 - Conclude with a Moral: End with a moral or lesson linked to the story's theme. 
        Condition: Use language and style throughout that is simple, clear, and appealing to children, including elements like repetition and rhymes. 
        Remember to keep the story age-appropriate in both length and content.""",
        agent=author
    )

    task2 = Task(
        description="""Add illustration ideas""",
        agent=illustrator
    )  

    task3 = Task(
        description="""Output the 7 parts of the story created by author and add a two sentence poem emphasizing the Moral of the story.
        """,
        agent=editor
    )  

    
    task4 = Task(
        description="""Summarize the author story into an image prompt.""",
        agent=artist
    )

    task5 = Task(
        description="""create a rhyming version of the story created by the author""",
        agent=poet
    )

    task6 = Task(
        description="""create a rhyming version of the story created by the author""",
        agent=reader
    )
    
    task7 = Task(
        description="""Output story add any necessary editor changes.""",
        agent=finalizer
    )
   
  

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[author, poet],
        tasks=[task1, task5],
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result

if 'text_block' not in st.session_state:
    st.session_state['text_block'] = """In the realm of kindness, Humphrey's the star,
A gentle giant, his heart beating afar.
When two little creatures lost and alone,
He showed them compassion, a love fully grown.
Humphrey the whale, so mighty and kind,
With a heart as big as the ocean, you'll find.
He spotted two friends who had quite a mishap,
A seagull and starfish, they needed a chap."""


if 'on_topic' not in st.session_state:
    st.session_state['on_topic'] = 'happy children on a ship'

if 'image_paths' not in st.session_state:
    st.session_state['image_paths'] = []
    
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>Clarifai Story Teller</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Branching Reading Adventure</h2>", unsafe_allow_html=True)


tabs = ["Create Your Story Script", "Build Your Image-Audio Book", "Interact with Your Characters"]

# Initialize the current tab in session state
if "current_tab" not in st.session_state:
    st.session_state.current_tab = tabs[0]

# Function to switch tabs
def switch_tab(tab_name):
    st.session_state.current_tab = tab_name

# Create tabs
tab1, tab2, tab3 = st.tabs(tabs)

# Tab 1: Introduction

with tab1:
    # Set up the Streamlit interface
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        st.image('crewai/resources/whale.jpg')     

    with col2:
        # Input for the user
        input_topic = st.text_area("What Exciting Adventures Await Us", height=100, placeholder="Start Our Story...")
        
        st.session_state['on_topic'] = input_topic
        # Button to run the process
        if st.button("Create a Story"):
            # Run the crewai process
            with st.spinner('Generating Content...'):
                result = crewai_process(input_topic)
                # Display the result
                st.session_state['text_block']  = result
                st.text_area("Output", value=result , height=300)
                
        st.image('crewai/resources/display.jpg')

    with col3:
        st.image('crewai/resources/clarifai.png')

# Tab 2: Data Visualization

with tab2:

    
    # Streamlit main page
    
    if st.button("Generate Images and Audio"):
        
        sentence_chunks = split_text_into_sentences_and_chunks(st.session_state.text_block , 8)
        prompts = [' '.join(chunk) for chunk in sentence_chunks]
        cols = st.columns(4)
        with st.spinner('Generating Content...'):
            for i, prompt in enumerate(prompts):
                image_path, img_error = generate_image(prompt+st.session_state.on_topic )
                audio, audio_error = generate_audio(prompt)
    
                with cols[i % 4]:
                    if img_error:
                        st.error(img_error)
                    else:
                        st.session_state['image_paths'].append(image_path)
                        st.image(image_path, prompt, use_column_width=True)
    
                    if audio_error:
                        st.error(audio_error)
                    else:
                        st.audio(audio, format='audio/wav')

# Tab 3: User Input and Results

with tab3:


    if 'image_paths' in st.session_state and st.session_state['image_paths']:
            # Create a slider for image selection

            col1, col2 = st.columns([2, 3])


            with col2:
                # Display the selected image
                
                image_index = st.radio(
                    "Choose an image",
                    options=list(range(len(st.session_state['image_paths']))),
                    format_func=lambda x: f"Image {x + 1}"
                )

                
                st.image(st.session_state['image_paths'][image_index])
                
            with col1:

                st.header("Image Details")
                st.divider()
                st.subheader("Image Components")
                
                image_conepts = get_image_concepts(st.session_state['image_paths'][image_index])

                unique_names = set()
                for region in image_conepts:
                    for concept in region.data.concepts:
                        name = concept.name
                        # Add unique names to the set
                        unique_names.add(name)
            
                # Display unique names
                
                if unique_names:
                    st.write(', '.join(unique_names))
                else:
                    st.write("No unique items detected.")
                
                st.divider()
                st.subheader("Description of Our Image")
        
                image_text_output = analyze_image(st.session_state['image_paths'][image_index])
                
                st.write(image_text_output)
                st.divider()

                st.session_state['on_topic'] = image_text_output

                st.header("Create a Story About This Image")

        
                # Button for actions related to the selected image
                if st.button("Create a New Story"):
                    
                    st.session_state['text_block'] = crewai_process(st.session_state['on_topic'])
                    # switch_tab(tabs[0])
                
             

