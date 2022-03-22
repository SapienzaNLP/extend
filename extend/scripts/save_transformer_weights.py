import sys

from transformers import PreTrainedModel

from extend.esc_ed_module import ESCModule


def save_transformer_weights(checkpoint_path: str, output_dir: str) -> None:

    esc_ed_module = ESCModule.load_from_checkpoint(checkpoint_path)
    qa_model: PreTrainedModel = esc_ed_module.qa_model

    qa_model.save_pretrained(output_dir)


def main():
    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2]
    save_transformer_weights(checkpoint_path, output_dir)


if __name__ == "__main__":
    main()
