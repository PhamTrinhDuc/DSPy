import dspy
from dspy.signatures.signature import signature_to_template
import os
import re
import dotenv
from build_dataset import prepare_dataset

dotenv.load_dotenv()

MODEL_NAME = "mixtral-8x7b-32768"
turbo = dspy.GROQ(
    model=MODEL_NAME,
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=500
)
dspy.settings.configure(lm=turbo)

# Now conceptually Signatures are a little like prompt 
class GrugTranslate(dspy.Signature):
    "Dịch đoạn tiếng Việt đơn giản sang Grug text"

    plain_vietnamese = dspy.InputField()
    grug_text = dspy.OutputField()

# Defining a module
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(GrugTranslate)
    
    def forward(self, plain_vietnamese):
        return self.prog(plain_vietnamese=plain_vietnamese)
    
# Measuring and Optimizing Performance
# Metric 1: https://apps.dtic.mil/sti/tr/pdf/AD0667273.pdf
def automated_readability_index(text):
    import re
    characters = len(re.sub(r'\s+', '', text)) # Count characters (ignoring whitespace)
    words = len(text.split()) # Count words by splitting the text
    # Count sentences by finding period, exclamation, or question mark
    sentences = len(re.findall(r'[.!?\n]', text))
    # small change is to add a new line character as grug doesn't seem to use punctuation.
    if words == 0 or sentences == 0:  # Prevent division by zero
        return 0
    # Calculate the Automated Readability Index (ARI)
    ari = (4.71 * (characters / words)) + (0.5 * (words / sentences)) - 21.43
    
    return round(ari, 2)

# Metric 2: AI Feedback
class AssessBasedOnQuestion(dspy.Signature):
    """Đưa ra văn bản được đánh giá, hãy đưa ra câu trả lời có hoặc không cho câu hỏi đánh giá."""

    assessed_text = dspy.InputField(format=str)
    assessment_question = dspy.InputField(format=str)
    assessment_answer = dspy.OutputField(desc="Yes or No")

if __name__ == "__main__":
    ##################### Explain GrugTranslate ################################
    # grug_translate_as_template = signature_to_template(GrugTranslate)
    train_dataset, test_dataset = prepare_dataset()

    # print(str(grug_translate_as_template))
    # print(train_dataset[0])
    # print(grug_translate_as_template.query(example=train_dataset[0]))
    # print(GrugTranslate.signature)
    # print(GrugTranslate.with_instructions)

    ##################### Explain CoT #########################
    # cot = CoT()
    # pred = cot.forward("Bạn không nên xây dựng những hệ thống phức tạp")
    # print(pred['rationale'])
    # print(pred['grug_text'])

    ##################### Explain metric optimize1 #################

    for example in train_dataset:
        source_ari = automated_readability_index(example['plain_Vietnamese'])
        grug_ari = automated_readability_index(example['grug_text'])
        print(f"ARI {source_ari} => {grug_ari}")

        ##################### Explain metric optimize2 #################
