import json
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Path to the artifacts folder
ART_DIR = Path(r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Topic Modeling\artifacts")

# Load topics file
with open(ART_DIR / "topic_top_words.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

# Create output folder for wordclouds
output_dir = ART_DIR / "wordclouds"
output_dir.mkdir(parents=True, exist_ok=True)

# Generate a WordCloud for each topic
for t in topics:
    topic_id = t["topic"]
    words = " ".join(t["top_words"])
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="plasma",
        max_words=100
    ).generate(words)

    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Topic {topic_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"topic_{topic_id}_wordcloud.png")
    plt.close()

print(f"✅ WordClouds saved in: {output_dir}")
