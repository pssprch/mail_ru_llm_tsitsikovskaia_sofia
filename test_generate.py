from stat_lm import GenerationConfig, construct_model


if __name__ == "__main__":
    model, tokenizer = construct_model()

    model.generate_response("привет, как дела", GenerationConfig())