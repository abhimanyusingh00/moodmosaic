import torch
from src.models.personality_big5 import PersonalityRegressorBig5, BIG5_TRAITS


def load_big5_model(ckpt_path="experiments/checkpoints/personality_big5/best_big5.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    encoder_name = ckpt.get("encoder_name", "sentence-transformers/all-mpnet-base-v2")
    model = PersonalityRegressorBig5(encoder_name=encoder_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def pretty_print_scores(texts, preds):
    for text, scores in zip(texts, preds):
        print("Text:")
        print("  ", text)
        print("Scores (approx, in [-1, 1]):")
        for trait, v in zip(BIG5_TRAITS, scores):
            print(f"  {trait}: {v:.3f}")
        print()


def main():
    model, device = load_big5_model()

    sample_texts = [
        "I love learning new ideas and exploring different research topics. I enjoy reading, reflecting, and trying creative approaches in my projects.",
        "I always plan my work carefully, make detailed schedules, and double check everything before I submit.",
        "I feel energized when I am around people and I like being the one who starts conversations in group projects.",
        "I try to be patient and supportive with my teammates, and I care a lot about keeping a friendly atmosphere.",
        "I often worry about results and deadlines, and sometimes I feel stressed or anxious when things are uncertain.",
    ]

    with torch.no_grad():
        preds = model(sample_texts).cpu().numpy()

    pretty_print_scores(sample_texts, preds)


if __name__ == "__main__":
    main()
