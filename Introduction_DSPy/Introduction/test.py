from main_dspy import *



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

# for example in train_dataset:
#     source_ari = automated_readability_index(example['plain_Vietnamese'])
#     grug_ari = automated_readability_index(example['grug_text'])
#     print(f"ARI {source_ari} => {grug_ari}")

##################### Explain metric optimize2 #################
example_question_assessment = dspy.Example(
    assessed_text="This is a test",
    assessment_question="Is this a test?", assessment_answer="Yes"
).with_inputs("assessed_text", "assessment_question")

print(signature_to_template(AssessBasedOnQuestion).query(example_question_assessment))
print(similarity_metric())
      