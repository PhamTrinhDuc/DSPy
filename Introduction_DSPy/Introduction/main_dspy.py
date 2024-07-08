import dspy
from dspy.signatures.signature import signature_to_template
import os
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

############################## Now conceptually Signatures are a little like prompt #######################
class GrugTranslate(dspy.Signature):
    "Dịch đoạn tiếng Việt đơn giản sang Grug text"

    plain_vietnamese = dspy.InputField()
    grug_text = dspy.OutputField()

# grug_translate_as_template = signature_to_template(GrugTranslate)
# train_dataset, test_dataset = prepare_dataset()

# print(str(grug_translate_as_template))
# print(train_dataset[0])
# print(grug_translate_as_template.query(example=train_dataset[0]))
# print(GrugTranslate.signature)
# print(GrugTranslate.with_instructions)

############################### Defining a module ###################################
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(GrugTranslate)
    
    def forward(self, plain_vietnamese):
        return self.prog(plain_vietnamese=plain_vietnamese)

# cot = CoT()
# pred = cot.forward("Bạn không nên xây dựng những hệ thống phức tạp")
# print(pred['rationale'])
# print(pred['grug_text'])

########################## Measuring and Optimizing Performance ######################