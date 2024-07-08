import dotenv
import os
import dspy
dotenv.load_dotenv()

wrkr = dspy.GROQ(
    model="mixtral-8x7b-32768",
    api_key=os.getenv("GROG_API_KEY"),
    max_tokens=500,
    temperature=0.1
)
bss = dspy.GROQ(
    model="mixtral-8x7b-32768",
    api_key=os.getenv("GROG_API_KEY"),
    max_tokens=500,
    temperature=0.1
)

dspy.configure(lm=wrkr)
