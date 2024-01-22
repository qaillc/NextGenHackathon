# clarify tools

import os
import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from PIL import Image
from io import BytesIO
from base64 import b64decode

import json
import os
from google.api_core import exceptions

#Constants

PAT = os.getenv('CLARIFAI_PAT')
USER_ID = 'openai'
APP_ID = 'dall-e'
MODEL_ID = 'dall-e-3'
MODEL_VERSION_ID = 'dc9dcb6ee67543cebc0b9a025861b868'

# Ensure the API key is available
if not PAT:
    raise ValueError("API key not found. Please set the CLARIFAI_PAT environment variable.")

import requests
from langchain.tools import tool


class ClarifaiTools():
  @tool("Clarifai Image Tool")
  def clarifai_image(prompt):

    """
    Draws images from prompts
    Args:
        query (str): Prompt
    Returns:
        str: Image
    """  

    # Clarifai gRPC setup
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(raw=prompt)
                    )
                )
            ]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        return None, "Error in generating image: " + post_model_outputs_response.status.description
    else:
        output = post_model_outputs_response.outputs[0].data.image.base64
        image = Image.open(BytesIO(output))
        return image, None

        

