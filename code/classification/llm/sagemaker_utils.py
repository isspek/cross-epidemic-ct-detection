import json
from typing import Dict

import boto3
import sagemaker
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
import time

from langchain import PromptTemplate, SagemakerEndpoint

from code.pysettings import AWS_LOCAL_PROFILE, AWS_ROLE, AWS_REGION


def create_endpoint_with_hf_model(model, endpoint_type='ml.g4dn.12xlarge'):
    session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=session)
    region = sagemaker_session.boto_region_name
    role = AWS_ROLE
    print(f"Region: {region}")
    image_uri = get_huggingface_llm_image_uri(
        backend="huggingface",  # or lmi
        region=region
    )
    print(f"Image URI: {image_uri}")
    if model == 'llama':
        model_label = 'llama-65b'
        #model_id = 'decapoda-research/llama-65b-hf'
        model_id = 'huggyllama/llama-65b'
        #model_label = 'llama-30b'
        #model_id = 'decapoda-research/llama-30b-hf'
    elif model == 'flan-xxl':
        model_label = 'flan-t5-xxl'
        model_id = 'google/flan-t5-xxl'
    elif model == 'flan-small':
        model_label = 'flan-t5-small'
        model_id = 'google/flan-t5-small'
    elif model == 'eleuther': # bitsandbytes quantization was used ...
        model_label = 'eleuther-20b'
        model_id = 'EleutherAI/gpt-neox-20b'
    model_name = model_label + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print(f"Model name: {model_name}")
    hub = {
        'HF_MODEL_ID': model_id,
        'HF_TASK': 'text-generation',
        'SM_NUM_GPUS': '8', # 8 for llama 65b on ml.g5.48xlarge
        #'HF_MODEL_QUANTIZE': 'gptq', #'bitsandbytes', #gptq
    }
    model = HuggingFaceModel(
        name=model_name,
        env=hub,
        role=role,
        image_uri=image_uri
    )
    print('Deploying model...')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=endpoint_type,
        endpoint_name=model_name
    )
    print(f"Created endpoint, name: {predictor.endpoint_name}")
    return predictor

class SagemakerContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]

def create_sagemaker_langchain_llm(endpoint_name):
    #predictor = create_endpoint_with_hf_model(model)
    print(f"Creating langcain adapter for Sagemaker, endpoint name: {endpoint_name}")
    content_handler = SagemakerContentHandler()
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        credentials_profile_name=AWS_LOCAL_PROFILE,
        region_name=AWS_REGION,
        model_kwargs={"temperature": 0, "max_new_tokens": 5, "watermark": False},
        content_handler=content_handler,
    )
    return llm

def test_lagchain_llm(endpoint_name='flan-t5-xxl2023-06-22-16-15-52'):
    predictor = None
    try:
        #predictor = create_endpoint_with_hf_model('flan-xxl')
        #endpoint_name = predictor.endpoint_name
        #session = boto3.Session()
        #sagemaker_session = sagemaker.Session(boto_session=session)
        #region = sagemaker_session.boto_region_name
        llm = create_sagemaker_langchain_llm(endpoint_name)
        res = llm('Who was Abraham Lincoln?')
        print(res)
    except:
        import traceback
        traceback.print_exc()
    finally:
        pass
        # if predictor:
        #     predictor.delete_endpoint()
        #     predictor.delete_model()

if __name__ == '__main__':
    test_lagchain_llm(endpoint_name='llama-65b2023-07-26-11-17-15')
    #create_endpoint_with_hf_model('llama', endpoint_type='ml.g5.48xlarge')