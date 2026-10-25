from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',temperature=0.5)

result = model.invoke("Who is the best ai company in world?")
print(result.content[0]['text'])