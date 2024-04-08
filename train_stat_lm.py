from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from stat_lm import StatLM, Tokenizer, remove_punct


save_path = Path("models/stat_lm/")
save_path.mkdir(exist_ok=True, parents=True)
dataset = load_dataset("SiberiaSoft/SiberianPersonaChat")

texts = []
for sample in tqdm(dataset["train"], desc="preprocess dataset"):
    if sample["name"] == "dialog_personal_context":
        texts.append(remove_punct(sample["input"] + sample["output"]))

tokenizer = Tokenizer().build_vocab(texts)

model = StatLM(tokenizer, context_size=5)

model.train(texts)

stat_lm_path = 'models/stat_lm/stat_lm.pkl'
tokenizer_path = 'models/stat_lm/tokenizer.pkl'

tokenizer.save(tokenizer_path)
model.save_stat(stat_lm_path)


from stat_lm import GenerationConfig
model.generate_response("собеседник: привет, как дела ты: ", GenerationConfig())

