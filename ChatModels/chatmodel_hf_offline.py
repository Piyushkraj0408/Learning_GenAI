from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggongface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=0,
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True,
    },
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of india")
print(result.content)